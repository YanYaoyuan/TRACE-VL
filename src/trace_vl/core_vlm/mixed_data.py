"""Official training-only temporal QA + existing general QA, frozen before fitting."""
from __future__ import annotations

import csv
import fcntl
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trace_vl.event_state.data import benchmark_inventory
from trace_vl.v3.backbone import frame_size
from trace_vl.v3.ovo_transfer import ensure_media
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .data import read_split, sha256
from .prepare import aliases, check_space, digest, verify_video, write_json
from .transfer_data import NEXTQA_VAL_SHA256, nextqa_rows

NEXTQA_TRAIN_SHA256 = '51ae6e611fa58835f63a0beb325fe647bb197c7e320a31f0bb7a9cf969e082b2'
TEMPORAL_TASKS = {'TN', 'TC', 'TP'}


def resolve_general_data(cfg):
    """Follow the frozen transfer's baseline provenance, never guess a directory suffix."""
    explicit = cfg.get('general_data')
    expected = None
    if explicit:
        directory = Path(explicit).resolve()
    else:
        inputs = json.loads((Path(cfg['external_data']) / 'inputs.json').read_text())
        baseline = Path(inputs['config']['baseline_run'])
        baseline_cfg = json.loads((baseline / 'experiment-config.json').read_text())
        directory = Path(baseline_cfg['data_out']).resolve()
        expected = inputs['training_manifest']
        training = json.loads((baseline / 'train/grid/run.json').read_text())
        if training['manifest_sha256'] != expected:
            raise ValueError('baseline and frozen transfer training manifest differ')
    required = ('manifest.jsonl', 'complete.json', 'verification.json')
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(f'prepared general QA data missing at {directory}: {missing}; '
                         'restore the recorded directory or set TRACE_GENERAL_DATA explicitly')
    actual = sha256(directory / 'manifest.jsonl')
    completed = json.loads((directory / 'complete.json').read_text())
    if actual != completed['manifest_sha256'] or (expected is not None and actual != expected):
        raise ValueError('resolved general QA manifest differs from recorded provenance')
    return directory


def frozen_external(directory):
    directory = Path(directory)
    complete = json.loads((directory / 'complete.json').read_text())
    if set(complete['manifests']) != {'ovo', 'nextqa'}:
        raise ValueError('expected the two completed external development manifests')
    for name, h in {**{f'{k}.jsonl': v for k, v in complete['manifests'].items()},
                    **complete['audit_files']}.items():
        if sha256(directory / name) != h:
            raise ValueError('external population/audit changed')
    for p, h in complete['videos'].items():
        if sha256(p) != h:
            raise ValueError('external media changed')
    return {k: directory / f'{k}.jsonl' for k in complete['manifests']}, complete


def balanced_selection(temporal, general, per_source, seed):
    """Equal question counts, round-robin source videos; no answer-dependent choice."""
    result = []
    for family, rows in [('temporal', temporal), ('general', general)]:
        groups = defaultdict(list)
        for r in rows:
            groups[r['source_group']].append(r)
        order = sorted(groups, key=lambda k: digest([seed, family, k]))
        for k in order:
            groups[k].sort(key=lambda r: digest([seed, r['id']]))
        selected = []
        for i in range(max((len(v) for v in groups.values()), default=0)):
            selected.extend(groups[k][i] for k in order if i < len(groups[k]))
            if len(selected) >= per_source:
                break
        if len(selected) < per_source:
            raise ValueError(f'insufficient {family} training QA: {len(selected)} < {per_source}')
        result.extend({**r, 'mixture_family': family, 'split': 'train'} for r in selected[:per_source])
    if len({r['id'] for r in result}) != len(result):
        raise ValueError('duplicate mixture question IDs')
    return sorted(result, key=lambda r: r['id'])


def disjoint_filter(rows, blocked, forbidden_hashes=None, hashes=None):
    kept, rejected = [], []
    for r in rows:
        keys = set(r.get('source_aliases', [])) | aliases(r['video'])
        reason = 'heldout_source_overlap' if keys & blocked else None
        if hashes is not None and hashes[r['video']] in forbidden_hashes:
            reason = 'heldout_exact_bytes'
        if reason:
            rejected.append({'id': r['id'], 'video': r['video'], 'reason': reason})
        else:
            kept.append(r)
    return kept, rejected


def prepare(cfg, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(cfg, out)


def _prepare(cfg, out):
    check_space(out, 30)
    manifests, external = frozen_external(cfg['external_data'])
    general_root = resolve_general_data(cfg)
    general_manifest = general_root / 'manifest.jsonl'
    completed = json.loads((general_root / 'complete.json').read_text())
    if sha256(general_manifest) != completed['manifest_sha256']:
        raise ValueError('general QA manifest changed')
    train_path, val_path = Path(cfg['nextqa_train']), Path(cfg['nextqa_val'])
    if sha256(train_path) != NEXTQA_TRAIN_SHA256 or sha256(val_path) != NEXTQA_VAL_SHA256:
        raise ValueError('use checksum-pinned official NExT-QA train.csv and val.csv')
    # Full validation IDs, including validation videos outside our 200-video pilot.
    with val_path.open() as f:
        blocked = {r['video'] for r in csv.DictReader(f)}
    with train_path.open() as f:
        all_train_ids = {r['video'] for r in csv.DictReader(f)}
    if all_train_ids & blocked:
        raise ValueError('official NExT-QA train/val source overlap')
    general_train = read_split(general_manifest, 'train')
    general_val = read_split(general_manifest, 'val')
    for r in general_val:
        blocked.update(r.get('source_aliases', []))
        blocked.update(aliases(r['video']))
    # OVO all videos (both halves), StreamingBench and Video-MME IDs only.
    # NExT-QA root is deliberately NOT used here: it contains official train videos.
    bench_ids, bench_report = benchmark_inventory(cfg['benchmark_roots'], required=True)
    blocked.update(bench_ids)
    for manifest in manifests.values():
        for r in read_split(manifest, 'val'):
            blocked.update(r.get('source_aliases', []))
    identity = {'config': cfg, 'train_csv': sha256(train_path), 'val_csv': sha256(val_path),
                'general_manifest': sha256(general_manifest), 'external': external,
                'blocked_ids': digest(sorted(blocked)), 'implementation': sha256(__file__)}
    identity['id'] = digest(identity)
    ensure_run_manifest(out / 'inputs.json', identity)
    if (out / 'complete.json').is_file():
        saved = json.loads((out / 'complete.json').read_text())
        if saved['inputs'] != identity['id']:
            raise ValueError('mixture inputs changed')
        for name, h in saved['files'].items():
            if sha256(out / name) != h:
                raise ValueError('frozen training mixture changed')
        for p, h in saved['videos'].items():
            if sha256(p) != h:
                raise ValueError('training media changed')
        print('REUSED mixture; no archive scan/full decode', flush=True)
        return out / 'manifest.jsonl', manifests
    temporal = nextqa_rows(train_path, videos=len(all_train_ids), seed=cfg['seed'], training_id_collisions=True)
    id_collisions = [{k: r[k] for k in ('annotation_id', 'id', 'task', 'video')}
                     for r in temporal if 'annotation_id' in r]
    write_json(out / 'annotation-audit.json', {'training_rows': len(temporal),
        'disambiguated_rows': id_collisions,
        'policy': 'distinct prompts sharing a source ID retain both with content-derived IDs; '
                  'identical/conflicting prompts fail; external validation IDs unchanged'})
    temporal = [{**r, 'id': 'train-' + r['id']} for r in temporal if r['task'] in TEMPORAL_TASKS]
    temporal, rejected = disjoint_filter(temporal, blocked)
    general_train = [r for r in general_train if 'options' in r]
    general_train, rej = disjoint_filter(general_train, blocked)
    rejected.extend(rej)
    # Freeze selection before checking availability; never substitute a different
    # video because a selected file is missing/corrupt.
    write_json(out / 'capacity.json', {'requested_per_family': cfg['questions_per_family'],
        'available_questions': {'temporal': len(temporal), 'general': len(general_train)},
        'available_videos': {'temporal': len({r['source_group'] for r in temporal}),
                             'general': len({r['source_group'] for r in general_train})},
        'note': 'Counts after source exclusions, before media checks; no automatic downsizing.'})
    rows = balanced_selection(temporal, general_train, cfg['questions_per_family'], cfg['seed'])
    from .scale_repair import exclude_declared_missing
    rows, media_exclusion = exclude_declared_missing(rows, cfg, identity)
    if media_exclusion:
        write_json(out / 'media-exclusion.json', media_exclusion)
    wanted = sorted({r['video'] for r in rows if r['mixture_family'] == 'temporal'})
    roots = [Path(p) for p in cfg['nextqa_roots']]
    originals = out / 'originals'
    originals.mkdir(exist_ok=True)
    found = defaultdict(set)
    for root in roots:
        if root.is_dir():
            for p in root.rglob('*.mp4'):
                if p.name in wanted:
                    found[p.name].add(p.resolve())
    for name, candidates in found.items():
        if len({sha256(p) for p in candidates}) > 1:
            raise ValueError(f'ambiguous source media: {name}')
        dest, chosen = originals / name, min(candidates)
        if dest.exists() or dest.is_symlink():
            if dest.resolve() != chosen:
                raise ValueError('media mapping changed')
        else:
            dest.symlink_to(chosen)
    last = [0.]

    def progress(**status):
        if time.monotonic() - last[0] > 20:
            print('MIXTURE MEDIA ' + json.dumps(status, default=str), flush=True)
            last[0] = time.monotonic()

    media = ensure_media(wanted, originals, roots, out / 'nextqa-media',
                         archive_workers=cfg['archive_workers'], progress=progress)
    write_json(out / 'media.json', media)
    if any(media[k] for k in ('still_missing', 'ambiguous', 'extracted_but_unreadable', 'archives_skipped')):
        raise ValueError(f'selected training media unavailable: inspect {out / "media.json"}')
    for r in rows:
        if r['mixture_family'] == 'temporal':
            r['video'] = str((out / 'nextqa-media/media' / r['video']).resolve())
    # Reuse full-decode records only when byte hashes and frame requirements match.
    general_verified = json.loads((general_root / 'verification.json').read_text())['videos']
    cache = out / 'verified'
    cache.mkdir(exist_ok=True)

    def verify(path):
        key = cache / (digest(path) + '.json')
        h = sha256(path)
        info = json.loads(key.read_text()) if key.exists() else general_verified.get(path, {})
        if (info.get('sha256') != h or info.get('frames', 0) < 2 * cfg['steps']
                or 'minimum_frames_end' not in info):
            info = verify_video(path, 2 * cfg['steps'])
        height, width = frame_size(info['height'], info['width'], cfg['pixels'], 32)
        if height // 32 < cfg['pool_grid'][0] or width // 32 < cfg['pool_grid'][1]:
            raise ValueError(f'media cannot support fixed spatial grid: {path}')
        write_json(key, info)
        return path, info['sha256']

    paths = sorted({r['video'] for r in rows})
    hashes = {}
    with ThreadPoolExecutor(cfg['workers']) as pool:
        for i, (p, h) in enumerate(pool.map(verify, paths), 1):
            hashes[p] = h
            if i % 20 == 0 or i == len(paths):
                print(f'VERIFIED TRAIN MEDIA {i}/{len(paths)}', flush=True)
    forbidden_hashes = set(external['videos'].values()) | {sha256(r['video']) for r in general_val}
    _, content_rejected = disjoint_filter(rows, set(), forbidden_hashes, hashes)
    if content_rejected:
        write_json(out / 'overlap-failure.json', content_rejected)
        raise ValueError('selected training bytes overlap held-out data; no GPU work')
    for family in ('general', 'temporal'):
        n = len({hashes[r['video']] for r in rows if r['mixture_family'] == family})
        if n < cfg['minimum_videos_per_family']:
            raise ValueError(f'insufficient distinct {family} videos: {n}')
    # Keep validation outside the training manifest. No model-based selection.
    atomic_text(out / 'manifest.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
    write_json(out / 'inventory.json', {'questions': len(rows), 'videos': len(set(hashes.values())),
        'families': dict(Counter(r['mixture_family'] for r in rows)),
        'tasks': dict(Counter(r.get('task', 'general') for r in rows)),
        'media_exclusion': media_exclusion,
        'excluded': rejected, 'benchmark_inventory': bench_report, 'source_overlap': 0,
        'exact_byte_overlap': 0, 'note': 'official NExT-QA training partition; no perceptual/pretraining audit'})
    write_json(out / 'complete.json', {'inputs': identity['id'], 'videos': hashes,
        'files': {n: sha256(out / n) for n in ['manifest.jsonl', 'inventory.json', 'media.json', 'annotation-audit.json',
                                             *(['media-exclusion.json'] if media_exclusion else [])]}})
    return out / 'manifest.jsonl', manifests
