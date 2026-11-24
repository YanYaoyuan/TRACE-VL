"""Controlled scene-order retention: real pixels, synthetic ordering, identical suffixes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from trace_vl.streaming_vision.media import VideoProbeCache
from trace_vl.v3.backbone import frame_size
from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream
from .control import checkpoint_identity
from .data import decode_frame_indices, sha256
from .experiment import run_jobs
from .prepare import digest, write_json
from .train import checked_checkpoint

SCOPE = ('Development scene-order retention diagnostic, not natural chronological video. '
         'All eight prior delayed sources, assistant-reviewed scene descriptions; not human certification. '
         'Only the two historical event blocks swap. Each pair has identical real-video filler, '
         'recent suffix, question, options, presentation clock and writer random seed. '
         'Frozen Plain LoRA, 32 historical slots, at most128 read tokens, no training. '
         'No claim of novel memory architecture or natural five-minute event reasoning.')


def select_events(review, annotation):
    """Freeze exact already-viewed source frames; forbid shared boundary observations."""
    if review['source_group'] != annotation['source_group']:
        raise ValueError('review and scene annotation refer to different sources')
    if len(annotation['labels']) != 2 or len(set(annotation['labels'])) != 2:
        raise ValueError('two distinct scene labels required')
    events = []
    for group, steps in zip(annotation['event_groups'], annotation['event_steps'], strict=True):
        if group not in (0, 1) or len(steps) != 2 or len(set(steps)) != 2:
            raise ValueError('exactly two distinct reviewed steps per scene required')
        r = review['groups'][group]
        if any(type(i) is not int or i < 0 or 2*i+1 >= len(r['indices']) for i in steps):
            raise ValueError('reviewed step index outside contact sheet')
        indices = [r['indices'][2*i:2*i+2] for i in steps]
        times = [r['times'][2*i:2*i+2] for i in steps]
        if any(len(p) != 2 for p in indices) or any(b <= a for a, b in times):
            raise ValueError('invalid reviewed source steps')
        if any(b[0] <= a[-1] for a, b in zip(times, times[1:])):
            raise ValueError('each source event must preserve its internal order')
        events.append({'indices': indices, 'times': times})
    flattened = [i for e in events for pair in e['indices'] for i in pair]
    if len(flattened) != 8 or len(set(flattened)) != 8:
        raise ValueError('paired scenes must have eight distinct source frames')
    return events


def pair_queries(annotation, swapped, question, delays):
    if delays != [15, 60, 300]:
        raise ValueError('fixed predeclared delays required')
    reverse = int(digest(['pair-options-v1', annotation['source_group']])[-1], 16) % 2
    options = annotation['labels'][::-1] if reverse else annotation['labels']
    first = annotation['labels'][int(swapped)]
    return [{'id': f'{annotation["source_group"]}-{int(swapped)}-delay{delay}',
             'question': question, 'options': options, 'answer': options.index(first),
             'query_time': float(delay + 7.5), 'additional_wait_seconds': delay,
             'cohort': 'paired_scene_order', 'evidence': [0., 3.5]}
            for delay in delays]


def write_views(paths, event_frames, filler_path, filler_indices, hw):
    """Lossless 2-fps real-frame inputs; both views share every pixel after the first4 steps."""
    h, w = hw
    processes, logs = [], []
    hashes = [hashlib.sha256(), hashlib.sha256()]
    suffix = hashlib.sha256()
    try:
        for path in paths:
            log = path.with_suffix('.encode.log').open('wb')
            logs.append(log)
            processes.append(subprocess.Popen(
                ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-y', '-threads', '1',
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '2', '-i', 'pipe:0',
                 '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', '-threads', '1', str(path)],
                stdin=subprocess.PIPE, stderr=log))
        for view, order in enumerate(((0, 1), (1, 0))):
            for index in order:
                raw = event_frames[index].tobytes()
                processes[view].stdin.write(raw)
                hashes[view].update(raw)
        filler = longstream.stream_pixels(filler_path, filler_indices, h, w)
        try:
            for block in filler:
                raw = block.tobytes()
                suffix.update(raw)
                for process, checksum in zip(processes, hashes, strict=True):
                    process.stdin.write(raw)
                    checksum.update(raw)
        finally:
            filler.close()
        for process in processes:
            process.stdin.close()
        if any(p.wait(timeout=60) != 0 for p in processes):
            raise ValueError('lossless paired media encoding failed; inspect encode.log')
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    return [h.hexdigest() for h in hashes], suffix.hexdigest()


def prepare(args):
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.base / 'inputs.json').read_text())
    plan = json.loads(args.annotations.read_text())
    manifest = json.loads((args.review / 'manifest.json').read_text())
    if base['id'] != plan['base_input_id'] or base['id'] != manifest['input_id']:
        raise ValueError('paired scenes belong to a different frozen pilot')
    if len(plan['rows']) != 8 or len(manifest['rows']) != 8:
        raise ValueError('preserve all eight fixed sources; no success-based exclusions')
    if checked_checkpoint(Path(base['plain']), base['checkpoint']['run']) != base['checkpoint']:
        raise ValueError('Plain checkpoint changed')
    if checkpoint_identity(Path(base['model'])) != base['model_identity']:
        raise ValueError('base model changed')
    previous_path = root / 'inputs.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if (previous['base_input_id'] != base['id'] or previous['annotations_sha256'] != sha256(args.annotations)
                or previous['review_manifest_sha256'] != sha256(args.review / 'manifest.json')
                or previous['source_code_sha256'] != sha256(longstream.__file__)
                or previous['pair_source_code_sha256'] != sha256(__file__)):
            raise ValueError('existing run has different inputs; use a new run directory')
        for source in previous['sources']:
            if sha256(source['video']) != source['video_sha256'] or sha256(source['metadata']) != source['metadata_sha256']:
                raise ValueError('existing prepared media changed')
        return
    train = json.loads((Path(base['plain']).parent / 'run.json').read_text())
    train_hashes = set(train['videos'].values())
    delayed = [s for s in base['sources'] if s['queries'][0]['cohort'] == 'activitynet_delayed']
    fillers = [s for s in base['sources'] if s['queries'][0]['cohort'] == 'ovo_long']
    cache = VideoProbeCache(root / 'metadata', stage_media=False)
    media = root / 'media'
    media.mkdir(exist_ok=True)
    pairs, views = [], [[], []]
    for a, r, source, filler in zip(plan['rows'], manifest['rows'], delayed, fillers, strict=True):
        if sha256(args.review / f'review-{a["number"]}.jpg') != r['image_sha256']:
            raise ValueError('reviewed contact sheet changed')
        for original in (source, filler):
            if sha256(original['video']) != original['video_sha256'] or original['video_sha256'] in train_hashes:
                raise ValueError('original source changed or overlaps Plain training')
            if sha256(original['metadata']) != original['metadata_sha256']:
                raise ValueError('original video metadata changed')
        if source['source_group'] != a['source_group'] or source['video_sha256'] == filler['video_sha256']:
            raise ValueError('event and filler sources are not distinct')
        events = select_events(r, a)
        meta = json.loads(Path(source['metadata']).read_text())
        hw = frame_size(meta['streams'][0]['height'], meta['streams'][0]['width'], base['pixels'], 32)
        flat = [i for e in events for pair in e['indices'] for i in pair]
        ordered = sorted(flat)
        decoded = decode_frame_indices(source['video'], ordered, *hw, 180, stop_after_selection=True)
        lookup = dict(zip(ordered, decoded, strict=True))
        frames = [np.stack([lookup[i] for pair in e['indices'] for i in pair]) for e in events]
        fm = json.loads(Path(filler['metadata']).read_text())
        indices, times, _ = longstream.schedule(fm, 304.)
        if len(indices) != 608:
            raise ValueError('filler must provide304 distinct two-frame steps')
        paths = [media / f'{a["source_group"]}-{int(swapped)}.mkv' for swapped in (False, True)]
        expected, suffix = write_views(paths, frames, filler['video'], indices, hw)
        receipt = {'source_group': a['source_group'], 'annotation': a, 'review_image_sha256': r['image_sha256'],
                   'events': events, 'source_video_sha256': source['video_sha256'],
                   'source_video': source['video'], 'filler_source_group': filler['source_group'],
                   'filler_video': filler['video'], 'filler_video_sha256': filler['video_sha256'],
                   'filler_source_indices': indices.tolist(), 'filler_source_times': times.tolist(),
                   'event_rgb_hashes': [hashlib.sha256(f.tobytes()).hexdigest() for f in frames],
                   'identical_suffix_rgb_sha256': suffix, 'views': []}
        for swapped, path, expected_hash in zip((False, True), paths, expected, strict=True):
            local, new_meta, checksum = cache.source(str(path), stage=False)
            m = root / 'metadata' / f'{checksum}.metadata.json'
            write_json(m, new_meta)
            new_indices, new_times, _ = longstream.schedule(new_meta, 307.5)
            if len(new_indices) != 616 or not np.allclose(new_times.ravel(), np.arange(616) / 2):
                raise ValueError('synthetic presentation clock drifted')
            actual = hashlib.sha256()
            for block in longstream.stream_pixels(str(local), new_indices, *hw):
                actual.update(block.tobytes())
            if actual.hexdigest() != expected_hash:
                raise ValueError('lossless source pixels changed during materialization')
            group = f'{a["source_group"]}-view{int(swapped)}'
            queries = pair_queries(a, swapped, plan['question'], plan['delays'])
            queries = [{**q, 'source_group': group, 'video': str(local)} for q in queries]
            view = {'source_group': group, 'pair_group': a['source_group'], 'swapped': swapped,
                    'memory_seed': int(digest(['pair-writer-v1', a['source_group']])[:8], 16),
                    'video': str(local), 'video_sha256': checksum, 'metadata': str(m),
                    'metadata_sha256': sha256(m), 'queries': queries}
            views[int(swapped)].append(view)
            receipt['views'].append({'group': group, 'video_sha256': checksum,
                                     'decoded_rgb_sha256': actual.hexdigest()})
        pairs.append(receipt)
        write_json(root / 'lineage' / f'{a["source_group"]}.json', receipt)
        print(f'PAIRED MEDIA VERIFIED {a["source_group"]}', flush=True)
    inputs = {**copy.deepcopy(base), 'protocol': 'bounded-scene-order-pairs-v1', 'scope': SCOPE,
              'base_input_id': base['id'], 'annotations_sha256': sha256(args.annotations),
              'review_manifest_sha256': sha256(args.review / 'manifest.json'),
              'source_code_sha256': sha256(longstream.__file__), 'pair_source_code_sha256': sha256(__file__),
              'sources': views[0] + views[1], 'questions': 48, 'pairs': pairs}
    inputs.pop('id')
    inputs['id'] = digest(inputs)
    ensure_run_manifest(root / 'inputs.json', inputs)
    write_json(root / 'data-audit.json', {'completed': True, 'original_event_sources': 8,
               'original_filler_sources': 8, 'materialized_views': 16, 'questions': 48,
               'training_video_hash_overlap': False, 'lossless_all_pixels_verified': True,
               'synthetic_presentation_clock': True, 'scope': SCOPE})


def report(args):
    inputs = json.loads((args.out / 'inputs.json').read_text())
    if inputs['pair_source_code_sha256'] != sha256(__file__):
        raise ValueError('paired protocol source changed')
    sources = inputs['sources']
    if args.smoke:
        sources = sources[:1] + sources[8:9]
    rows = {}
    for source in sources:
        done = json.loads((args.out / 'sources' / f'{digest(source["source_group"])}.json').read_text())
        if not done['completed'] or done['input_id'] != inputs['id'] or done['stream_encoder_calls'] != 308:
            raise ValueError('paired view did not complete one encoding per step')
        for q in source['queries']:
            row = json.loads((args.out / 'questions' / f'{digest(q["id"])}.json').read_text())
            if row['input_id'] != inputs['id'] or row['id'] != q['id'] or row['gold'] != q['answer']:
                raise ValueError('question identity drift')
            rows[(source['pair_group'], q['additional_wait_seconds'], source['swapped'])] = row
    scores = {}
    groups = sorted({s['pair_group'] for s in sources})
    for delay in (15, 60, 300):
        totals = {arm: {'pairs': len(groups), 'both_correct': 0, 'question_correct': 0,
                        'changed_choice': 0} for arm in longstream.BASELINES + longstream.ORACLES}
        for group in groups:
            a, b = [rows[(group, delay, swapped)] for swapped in (False, True)]
            if a['gold'] == b['gold']:
                raise ValueError('counterfactual label did not flip')
            for row in (a, b):
                if (row['stream_encoder_calls'] != delay + 8 or row['memory']['seen'] != delay + 8
                        or row['memory']['capacity'] != 32 or row['memory']['fifo_steps'] > 32
                        or row['memory']['reservoir_steps'] > 32 or row['memory']['recent_steps'] > 8):
                    raise ValueError('encoding count or retained-state capacity mismatch')
            for arm, total in totals.items():
                p, q = a['predictions'][arm], b['predictions'][arm]
                for row, prediction in ((a, p), (b, q)):
                    if (prediction['visual_tokens'] > 128 or prediction['query_time_encoder_calls'] != 0
                            or max((t for pair in prediction['read_times'] for t in pair), default=0) > row['query_time']):
                        raise ValueError('read budget, encoder replay or causality violation')
                if p['read_frame_indices'] != q['read_frame_indices'] or p['read_times'] != q['read_times']:
                    raise ValueError('pair has different temporal slot selection')
                if arm in ('recent4', 'recent8', 'text_only'):
                    if any(p[k] != q[k] for k in ('choice', 'option_logits', 'real_feature_rgb_hashes', 'read_times')):
                        raise ValueError('same-input control changed across paired histories')
                correct = (p['choice'] == a['gold'], q['choice'] == b['gold'])
                total['both_correct'] += int(all(correct))
                total['question_correct'] += sum(correct)
                total['changed_choice'] += int(p['choice'] != q['choice'])
        scores[str(delay)] = totals
    write_json(args.out / ('smoke-ok.json' if args.smoke else 'report.json'), {
        'completed': True, 'smoke': args.smoke, 'input_id': inputs['id'], 'scope': SCOPE,
        'pairs': len(groups), 'independent_event_sources': len(groups), 'questions': len(rows),
        'views': len(sources), 'results': scores, 'architecture_accepted': False, 'no_training': True,
        'limitations': ['Previously used development sources and assistant scene annotations.',
                        'Synthetic clip reordering and cross-video distractors, not a natural video benchmark.',
                        'Small diagnostic; no statistical generalization or new architecture claim.']})


def run(args):
    import sys

    prepare(args)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'TRACE_WANDB': '0', 'TRACE_SWANLAB': '0', 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    run_jobs([('gpu-check', [sys.executable, '-c',
              'import torch; assert torch.cuda.device_count()==4; '
              'assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24'], list(range(4)))],
             env, args.out / 'logs')
    command = [sys.executable, '-m', 'trace_vl.core_vlm.longstream', 'worker', '--out', str(args.out)]
    run_jobs([(f'smoke-{i}', [*command, '--shard', str(i), '--smoke'], [i]) for i in range(4)], env, args.out / 'logs')
    args.smoke = True
    report(args)
    run_jobs([(f'full-{i}', [*command, '--shard', str(i)], [i]) for i in range(4)], env, args.out / 'logs')
    args.smoke = False
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'run', 'report'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base', type=Path, default=Path('/data/yyy/tracevl-runs/core-longstream-29e5163-001'))
    p.add_argument('--review', type=Path, default=Path('/data/yyy/tracevl-runs/longstream-evidence-review-29e5163-001'))
    p.add_argument('--annotations', type=Path, default=Path('docs/core_vlm/longstream_pairs_v1.json'))
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    {'prepare': prepare, 'run': run, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
