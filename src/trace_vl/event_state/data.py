"""JoyAI response-time adapter, selected extraction, immutable media audit.

Annotation response times are NOT earliest-knowable event times. Only a single
request / single response record is supported in this initial pilot. Original
archives and benchmark media are never modified.
"""
from __future__ import annotations

import csv
import math
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .common import bind, digest, read_json, sha, write_json, write_rows

VIDEO_EXT = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}


def video_id(value):
    value = str(value).replace('\\', '/').split('?')[0].rstrip('/')
    name = Path(value).name
    if Path(name).suffix.lower() in VIDEO_EXT:
        name = Path(name).stem
    return name[2:] if name.startswith('v_') else name


def split_for(key, seed):
    n = int(digest([seed, key])[:8], 16) / 2**32
    return 'train' if n < .75 else 'calibration' if n < .85 else 'dev'


def joy_records(raw, cfg):
    rows, rejected = [], Counter()
    seen = set()
    for record in raw:
        try:
            if record.get('task_type') != 'event_grounding':
                rejected['other_task'] += 1
                continue
            questions, responses = record['question'], record['response']
            if len(questions) != 1 or len(responses) != 1:
                rejected['multiple_turns'] += 1
                continue
            name = record['video_name']
            if Path(name).name != name or Path(name).suffix.lower() not in VIDEO_EXT:
                raise ValueError('unsafe or unsupported video name')
            start, target = float(questions[0]['time']), float(responses[0]['time'])
            text = questions[0]['content'].strip()
            if not text or not all(map(math.isfinite, [start, target])) or start < 0 or target <= start:
                raise ValueError('invalid times or question')
            if target-start > cfg['max_wait'] or target+3 > cfg['max_duration']:
                rejected['outside_pilot_horizon'] += 1
                continue
            # Whole integer seconds; both t-.5 and t frames must be available.
            ticks = list(range(max(1, math.ceil(start+.5)), math.ceil(target)+4))
            if not ticks or sum(t < target for t in ticks) < 1:
                rejected['no_wait_supervision'] += 1
                continue
            key = video_id(name)
            row = {'id': digest([name, start, target, text])[:24], 'video_name': name,
                   'source_video_id': key, 'source': record.get('source', 'unspecified'),
                   'split': split_for(key, cfg['seed']), 'condition': text, 'query_s': start,
                   'response_s': target, 'ticks': ticks, 'label_type': 'annotated_response_readiness',
                   'semantics': 'wait before annotated response; ready for at most three seconds after; '
                                'not physical event onset/end or exhaustively labelled surveillance'}
            if row['id'] in seen:
                rejected['duplicate_condition'] += 1
                continue
            seen.add(row['id'])
            rows.append(row)
        except (KeyError, TypeError, ValueError, IndexError):
            rejected['malformed'] += 1
    return rows, dict(rejected)


def collect_ids(obj):
    found = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (str, int)) and (key.lower() in {
                'video', 'video_name', 'video_id', 'video_path', 'youtube_id', 'youtube_url',
                'url', 'videoid', 'vid', 'original_video_id'}):
                m = re.search(r'(?:v=|youtu.be/)([A-Za-z0-9_-]{11})', str(value))
                found.add(m.group(1) if m else video_id(value))
            found.update(collect_ids(value))
    elif isinstance(obj, list):
        for v in obj:
            found.update(collect_ids(v))
    return found


def benchmark_inventory(roots, required=True, progress=None):
    blocked, report = set(), []
    for root in map(Path, roots):
        if progress:
            progress.update('benchmark-inventory', root=str(root))
        ids, files, errors, archives = set(), [], [], []
        if not root.is_dir():
            errors.append('directory missing')
        else:
            for p in sorted(Path(base)/name for base, _, names in os.walk(root) for name in names):
                if p.suffix.lower() not in VIDEO_EXT | {'.zip', '.json', '.jsonl', '.csv', '.parquet'}:
                    continue
                if not p.is_file():
                    continue
                if progress:
                    progress.update(current_file=str(p), ids=len(ids), annotations=len(files), archives=len(archives))
                if p.suffix.lower() in VIDEO_EXT:
                    ids.add(video_id(p.name))
                elif p.suffix.lower() == '.zip':
                    # StreamingBench's official CSVs have question_id, not video IDs.
                    # Read actual media names from the ZIP central directory; never
                    # decompress benchmark videos or infer video IDs from question IDs.
                    try:
                        with zipfile.ZipFile(p) as z:
                            members = sorted((m.filename, m.file_size, m.CRC) for m in z.infolist()
                                             if not m.is_dir() and Path(m.filename).suffix.lower() in VIDEO_EXT
                                             and '__MACOSX' not in Path(m.filename).parts
                                             and not Path(m.filename).name.startswith('._'))
                        archive_ids = {video_id(name) for name, _, _ in members}
                        ids.update(archive_ids)
                        archives.append({'path': str(p), 'bytes': p.stat().st_size,
                                         'video_members': len(members), 'ids': len(archive_ids),
                                         'directory_sha256': digest(members), 'mode': 'central_directory_only'})
                    except (OSError, zipfile.BadZipFile, RuntimeError) as e:
                        errors.append(f'{p}: ZIP directory unreadable: {e}')
                elif p.suffix.lower() in {'.json', '.jsonl', '.csv', '.parquet'}:
                    try:
                        if p.suffix == '.parquet':
                            import pyarrow.parquet as pq
                            table = pq.ParquetFile(p)
                            cols = [n for n in table.schema.names if n.lower() in {
                                'video', 'video_name', 'video_id', 'video_path', 'youtube_id',
                                'youtube_url', 'url', 'videoid', 'vid'}]
                            # Never load video bytes into RAM.
                            for batch in table.iter_batches(columns=cols, batch_size=1024):
                                ids.update(collect_ids(batch.to_pylist()))
                        elif p.suffix == '.csv':
                            with open(p, encoding='utf-8-sig', newline='') as f:
                                ids.update(collect_ids(list(csv.DictReader(f))))
                        elif p.suffix == '.jsonl':
                            import json
                            with open(p) as f:
                                for line in f:
                                    if line.strip():
                                        ids.update(collect_ids(json.loads(line)))
                        else:
                            ids.update(collect_ids(read_json(p)))
                        files.append({'path': str(p), 'sha256': sha(p)})
                    except Exception as e:
                        errors.append(f'{p}: {e}')
        report.append({'root': str(root), 'ids': len(ids), 'files': files, 'archives': archives,
                       'errors': errors, 'scope': 'normalized annotation/media filenames; '
                       'not cross-name perceptual deduplication or a full ZIP CRC/decode check'})
        blocked.update(ids)
        if required and (not files or not ids or errors):
            raise ValueError(f'Benchmark exclusion inventory incomplete for {root}: {errors[:3]}, '
                             f'{len(files)} annotations, {len(archives)} ZIP directories, '
                             f'{len(ids)} IDs. No training started.')
    return blocked, report


def media_index(roots, progress=None):
    index = defaultdict(set)
    scanned = 0
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        if progress:
            progress.update('media-file-index', root=str(root))
        for p in (Path(base)/name for base, _, names in os.walk(root) for name in names):
            scanned += 1
            if progress and scanned % 1000 == 0:
                progress.update(entries=scanned, video_names=len(index))
            if p.suffix.lower() in VIDEO_EXT and p.is_file():
                index[p.name].add(str(p.resolve()))
    return index


def verify_video(path):
    """Decode every frame once; record sha plus count/metadata discrepancy. No silent short clip."""
    import cv2
    if hasattr(cv2, 'CAP_PROP_N_THREADS'):
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    else:
        cap = cv2.VideoCapture(str(path))
    try:
        expected, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError('unreadable metadata')
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ValueError('empty decoded frame')
            count += 1
        if count < 2 or (expected > 0 and count < expected-1):
            raise ValueError(f'truncated decode: {count} / {expected}')
        return {'sha256': sha(path), 'frames': count, 'fps': fps,
                'last_frame_s': (count-1)/fps, 'verification': 'full_decode_and_content_sha256',
                'size': Path(path).stat().st_size, 'mtime_ns': Path(path).stat().st_mtime_ns}
    finally:
        cap.release()


def prepare(annotation, media_roots, out, cfg, code):
    from .preparation import Progress, worker_count
    workers = worker_count()
    with Progress(out, workers) as progress:
        return _prepare(annotation, media_roots, out, cfg, code, workers, progress)


def _prepare(annotation, media_roots, out, cfg, code, workers, progress):
    from trace_vl.v3.ovo_transfer import ensure_media

    from .preparation import reuse_extracted, verify_many
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blocked, bench = benchmark_inventory(cfg['benchmark_roots'], cfg['require_benchmark_inventory'], progress)
    identity = {'annotation': sha(annotation), 'settings': cfg, 'code': code,
                'benchmark_inventory': bench, 'media_roots': list(map(str, media_roots))}
    bind(out/'identity.json', identity)
    progress.update('annotation-conversion', annotation=str(annotation))
    rows, rejected = joy_records(read_json(annotation), cfg)
    by_video = defaultdict(list)
    for r in rows:
        if r['source_video_id'] in blocked:
            rejected['benchmark_overlap'] = rejected.get('benchmark_overlap', 0)+1
        else:
            by_video[r['video_name']].append(r)
    # Fixed hash order. Selection is made before model execution; no accuracy-based refill.
    index = media_index(media_roots, progress)
    names = sorted(by_video, key=lambda s: (s not in index, digest([cfg['seed'], s])))[:cfg['max_videos']]
    originals = out/'original-links'
    originals.mkdir(exist_ok=True)
    duplicates = {}
    progress.update('resolve-originals', selected=len(names))
    for name in names:
        progress.update(current_video=name)
        paths = sorted(index.get(name, []))
        if len(paths) > 1:
            hashes = {sha(p) for p in paths}
            if len(hashes) > 1:
                duplicates[name] = paths
                continue
        if paths:
            dest = originals/name
            if dest.is_symlink():
                dest.unlink()
            dest.symlink_to(paths[0])
    allowed = [n for n in names if n not in duplicates]
    if shutil.disk_usage(out).free < cfg['reserve_gb']*1e9:
        raise ValueError('Insufficient free space before extraction')
    archive_workers = int(os.environ.get('TRACE_ARCHIVE_WORKERS', min(workers, 8)))
    if not 1 <= archive_workers <= 32:
        raise ValueError('TRACE_ARCHIVE_WORKERS must be between 1 and 32')
    progress.update('reuse-extracted-media', selected=len(allowed))
    reuse_extracted(out, identity, allowed)
    progress.update('media-probe-and-extraction', selected=len(allowed), archive_workers=archive_workers)
    media = ensure_media(allowed, str(originals), list(map(str, media_roots)), str(out),
                         progress=lambda **values: progress.update(**values), archive_workers=archive_workers)
    kept = []
    previous = read_json(out/'data-audit.json').get('verified', {}) if (out/'data-audit.json').is_file() else {}
    verified, errors = verify_many(allowed, out, previous, workers, progress)
    progress.update('assemble-manifest', verified=len(verified), decode_errors=len(errors))
    # Worker completion order must not change the source selection or manifest order.
    for name in allowed:
        if name not in verified:
            continue
        p, v = out/'media'/name, verified[name]
        for r in sorted(by_video[name], key=lambda x: x['id'])[:cfg['max_conditions_per_video']]:
            if r['ticks'][-1] > v['last_frame_s']:
                rejected['response_or_followup_outside_media'] = rejected.get(
                    'response_or_followup_outside_media', 0)+1
                continue
            kept.append({**r, 'media_path': str(p.resolve()), 'media_sha256': v['sha256'],
                         'fps': v['fps'], 'annotation_sha256': identity['annotation']})
    # A normalized source ID must not map to two different underlying videos.
    source_hashes = defaultdict(set)
    for r in kept:
        source_hashes[r['source_video_id']].add(r['media_sha256'])
    ambiguous_ids = {k for k, v in source_hashes.items() if len(v) > 1}
    kept = [r for r in kept if r['source_video_id'] not in ambiguous_ids]
    # Content duplicates cannot be split across train / calibration / dev.
    content_splits = defaultdict(set)
    for r in kept:
        content_splits[r['media_sha256']].add(r['split'])
    conflicting = {k for k, v in content_splits.items() if len(v) > 1}
    kept = [r for r in kept if r['media_sha256'] not in conflicting]
    counts = {s: {'videos': len({r['source_video_id'] for r in kept if r['split'] == s}),
                  'conditions': sum(r['split'] == s for r in kept)} for s in cfg['min_videos']}
    problems = []
    for s, c in counts.items():
        for kind, limits in [('videos', cfg['min_videos']), ('conditions', cfg['min_conditions'])]:
            if c[kind] < limits[s]:
                problems.append(f'{s} {kind}: {c[kind]} < {limits[s]}')
    waits = Counter(round(r['response_s']-r['query_s'], 1) for r in kept)
    report = {'counts': counts, 'rejected': rejected, 'media': media, 'ambiguous_originals': duplicates,
              'decode_errors': errors, 'verified': verified, 'cross_split_duplicate_hashes': sorted(conflicting), 'ambiguous_source_ids': sorted(ambiguous_ids),
              'source_counts': dict(Counter(r['source'] for r in kept)), 'wait_seconds_histogram': dict(waits), 'selection': 'existing originals first, then fixed hash order',
              'largest_wait_bin_fraction': max(waits.values(), default=0)/max(1, len(kept)), 'benchmark_inventory': bench,
              'overlap_check_scope': 'metadata IDs and exact content within selected data; not perceptual dedup',
              'problems': problems, 'ready': not problems}
    write_rows(out/'manifest.jsonl', kept)
    write_json(out/'data-audit.json', report)
    print('DATA AUDIT', counts, 'problems:', problems, flush=True)
    if problems:
        raise ValueError('Data gate failed; inspect data-audit.json. No GPU work started.')
    return kept
