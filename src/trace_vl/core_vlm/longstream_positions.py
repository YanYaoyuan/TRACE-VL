"""Cross evidence age with event position; preserve all eight scene-order sources."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from trace_vl.streaming_vision.media import VideoProbeCache
from trace_vl.v3.backbone import frame_size
from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream, longstream_pairs
from .control import checkpoint_identity
from .data import decode_frame_indices, sha256
from .experiment import run_jobs
from .prepare import digest, write_json
from .train import checked_checkpoint

OFFSETS = (0, 120, 300)
DELAYS = (15, 60, 300)
ARMS = longstream.BASELINES + ('prefix4',) + longstream.ORACLES + ('fifo_stored_oracle',)
SCOPE = ('Synthetic development scene-order diagnostic: all8 previously inspected event sources, '
         'crossed with0/120/300 seconds of common pre-event video and15/60/300 seconds of post-event '
         'distractor plus4 recent steps. Fixed Plain LoRA;32 history slots,128 read tokens; no training. '
         'Explicit prefix4 baseline and privileged FIFO/reservoir stored-routing controls. '
         'Causal last-frame hold exposes lowfps repetition. Up to608 seconds of synthetic presentation, '
         'not natural10-minute reasoning, unseen evaluation, human annotation or a novel method.')


def shifted_queries(annotation, swapped, offset):
    if offset not in OFFSETS:
        raise ValueError('predeclared event offsets required')
    rows = longstream_pairs.pair_queries(annotation, swapped,
        'Which of these two scenes was shown first in the video?', list(DELAYS))
    group = f'{annotation["source_group"]}-offset{offset}'
    return [{**q, 'id': f'{group}-{int(swapped)}-delay{q["additional_wait_seconds"]}',
             'query_time': q['query_time'] + offset, 'evidence': [float(offset), offset + 3.5]}
            for q in rows]


def prepare(args):
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.original / 'inputs.json').read_text())
    previous = json.loads((args.previous / 'inputs.json').read_text())
    previous_report = json.loads((args.previous / 'report.json').read_text())
    if (not previous_report['completed'] or previous_report['input_id'] != previous['id']
            or previous['base_input_id'] != original['id']):
        raise ValueError('completed original paired experiment required')
    if checked_checkpoint(Path(original['plain']), original['checkpoint']['run']) != original['checkpoint']:
        raise ValueError('Plain checkpoint changed')
    if checkpoint_identity(Path(original['model'])) != original['model_identity']:
        raise ValueError('base model changed')
    fingerprints = {'source_code_sha256': sha256(longstream.__file__),
                    'pair_source_code_sha256': sha256(longstream_pairs.__file__),
                    'position_source_code_sha256': sha256(__file__),
                    'previous_input_sha256': sha256(args.previous / 'inputs.json')}
    if (root / 'inputs.json').exists():
        prior = json.loads((root / 'inputs.json').read_text())
        if any(prior[k] != v for k, v in fingerprints.items()):
            raise ValueError('prepared run identity changed; use a new directory')
        for source in prior['sources']:
            if sha256(source['video']) != source['video_sha256'] or sha256(source['metadata']) != source['metadata_sha256']:
                raise ValueError('prepared source changed')
        return
    events = [s for s in original['sources'] if s['queries'][0]['cohort'] == 'activitynet_delayed']
    fillers = [s for s in original['sources'] if s['queries'][0]['cohort'] == 'ovo_long']
    if len(events) != 8 or len(fillers) != 8 or len(previous['pairs']) != 8:
        raise ValueError('preserve all8 event and8 filler sources')
    train_hashes = set(json.loads((Path(original['plain']).parent / 'run.json').read_text())['videos'].values())
    for source in events + fillers:
        if (sha256(source['video']) != source['video_sha256'] or source['video_sha256'] in train_hashes
                or sha256(source['metadata']) != source['metadata_sha256']):
            raise ValueError('original source changed or exact-byte training overlap')
    old_views = {(s['pair_group'], s['swapped']): s for s in previous['sources']}
    cache = VideoProbeCache(root / 'metadata', stage_media=False)
    media = root / 'media'
    media.mkdir(exist_ok=True)
    views, pairs = [[], []], []
    for number, (source, filler, receipt) in enumerate(zip(events, fillers, previous['pairs'], strict=True)):
        g = source['source_group']
        if receipt['source_group'] != g or receipt['source_video_sha256'] != source['video_sha256']:
            raise ValueError('reviewed event source changed')
        meta = json.loads(Path(source['metadata']).read_text())
        hw = frame_size(meta['streams'][0]['height'], meta['streams'][0]['width'], original['pixels'], 32)
        event_ids = [[i for pair in e['indices'] for i in pair] for e in receipt['events']]
        flat = sorted(i for ids in event_ids for i in ids)
        decoded = decode_frame_indices(source['video'], flat, *hw, 180, stop_after_selection=True)
        lookup = dict(zip(flat, decoded, strict=True))
        frames = [np.stack([lookup[i] for i in ids]) for ids in event_ids]
        if [hashlib.sha256(f.tobytes()).hexdigest() for f in frames] != receipt['event_rgb_hashes']:
            raise ValueError('reviewed event pixels changed')
        suffix_indices, suffix_times = longstream_pairs.filler_schedule(json.loads(Path(filler['metadata']).read_text()))
        prefix_source = fillers[(number + 1) % len(fillers)]
        prefix_meta = json.loads(Path(prefix_source['metadata']).read_text())
        for offset in OFFSETS:
            group = f'{g}-offset{offset}'
            prefix_indices, prefix_times = (longstream_pairs.filler_schedule(prefix_meta, offset) if offset else
                                            (np.asarray([], dtype=int), np.empty((0, 2))))
            paths = [media / f'{group}-view{int(s)}.mkv' for s in (False, True)]
            reuse_paths = None
            if offset == 0:
                reuse_paths = [Path(old_views[g, s]['video']) for s in (False, True)]
                if any(sha256(p) != old_views[g, s]['video_sha256'] for s, p in zip((False, True), reuse_paths, strict=True)):
                    raise ValueError('completed paired media changed')
                if receipt['filler_source_indices'] != suffix_indices.tolist() or receipt['filler_video_sha256'] != filler['video_sha256']:
                    raise ValueError('reused suffix lineage changed')
            expected, suffix_hash = longstream_pairs.write_views(paths, frames, filler['video'], suffix_indices, hw,
                reuse_paths=reuse_paths, shared_prefix=(prefix_source['video'], prefix_indices) if offset else None)
            lineage = {'pair_group': group, 'event_source_group': g, 'prefix_seconds': offset,
                       'reviewed_event_receipt': receipt, 'suffix_video_sha256': filler['video_sha256'],
                       'suffix_source_group': filler['source_group'], 'suffix_indices': suffix_indices.tolist(),
                       'suffix_times': suffix_times.tolist(), 'suffix_rgb_sha256': suffix_hash,
                       'prefix_video_sha256': prefix_source['video_sha256'], 'prefix_source_group': prefix_source['source_group'],
                       'prefix_indices': prefix_indices.tolist(), 'prefix_times': prefix_times.tolist(),
                       'reused_media': offset == 0, 'views': []}
            for swapped, path, rgb_hash in zip((False, True), paths, expected, strict=True):
                local, new_meta, checksum = cache.source(str(path), stage=False)
                metadata = root / 'metadata' / f'{checksum}.metadata.json'
                write_json(metadata, new_meta)
                ids, times, _ = longstream.schedule(new_meta, offset + 307.5)
                if len(ids) != 2 * (offset + 308) or not np.allclose(times.ravel(), np.arange(len(ids)) / 2):
                    raise ValueError('presentation clock drifted')
                actual = hashlib.sha256()
                for block in longstream.stream_pixels(local, ids, *hw):
                    actual.update(block.tobytes())
                if actual.hexdigest() != rgb_hash:
                    raise ValueError('lossless source-pixel verification failed')
                view_group = f'{group}-view{int(swapped)}'
                qs = [{**q, 'source_group': view_group, 'video': str(local)}
                      for q in shifted_queries(receipt['annotation'], swapped, offset)]
                views[int(swapped)].append({'source_group': view_group, 'pair_group': group,
                    'event_source_group': g, 'prefix_seconds': offset, 'swapped': swapped,
                    'memory_seed': int(digest(['pair-writer-v1', g])[:8], 16), 'queries': qs,
                    'video': str(local), 'video_sha256': checksum, 'metadata': str(metadata),
                    'metadata_sha256': sha256(metadata)})
                lineage['views'].append({'group': view_group, 'video_sha256': checksum, 'decoded_rgb_sha256': actual.hexdigest()})
            pairs.append(lineage)
            write_json(root / 'lineage' / f'{group}.json', lineage)
            print(f'POSITION MEDIA VERIFIED {group}', flush=True)
    sources = views[0] + views[1]
    first = events[0]['source_group']
    smoke_groups = [s['source_group'] for s in sources if s['event_source_group'] == first and s['prefix_seconds'] in (0, 300)]
    inputs = {**copy.deepcopy(original), **fingerprints, 'protocol': 'bounded-scene-order-position-cross-v1',
              'scope': SCOPE, 'sources': sources, 'pairs': pairs, 'questions': 144, 'position_controls': True,
              'previous_input_id': previous['id'], 'original_input_id': original['id'],
              'smoke_source_groups': smoke_groups, 'event_offsets_seconds': list(OFFSETS), 'delays_seconds': list(DELAYS)}
    inputs.pop('id')
    inputs['id'] = digest(inputs)
    ensure_run_manifest(root / 'inputs.json', inputs)
    write_json(root / 'data-audit.json', {'completed': True, 'original_event_sources': 8, 'original_filler_sources': 8,
        'pair_conditions': 24, 'views': 48, 'questions': 144, 'reused_prefix0_pairs': 8,
        'exact_byte_training_overlap': False, 'all_output_pixels_verified': True, 'scope': SCOPE})


def report(args):
    inputs = json.loads((args.out / 'inputs.json').read_text())
    if inputs['position_source_code_sha256'] != sha256(__file__):
        raise ValueError('position protocol source changed')
    sources = [s for s in inputs['sources'] if not args.smoke or s['source_group'] in inputs['smoke_source_groups']]
    rows = {}
    for source in sources:
        done = json.loads((args.out / 'sources' / f'{digest(source["source_group"])}.json').read_text())
        offset = source['prefix_seconds']
        if not done['completed'] or done['input_id'] != inputs['id'] or done['stream_encoder_calls'] != offset + 308:
            raise ValueError('source single-pass encoding incomplete')
        for q in source['queries']:
            r = json.loads((args.out / 'questions' / f'{digest(q["id"])}.json').read_text())
            d = q['additional_wait_seconds']
            if (r['id'] != q['id'] or r['input_id'] != inputs['id'] or r['gold'] != q['answer']
                    or r['stream_encoder_calls'] != offset + d + 8 or set(r['predictions']) != set(ARMS)
                    or r['memory']['prefix_control_steps'] != 4 or r['memory']['capacity'] != 32
                    or r['memory']['joint_experiment_unique_retained_steps'] > 76):
                raise ValueError('identity, encoding or memory budget failed')
            for p in r['predictions'].values():
                if (p['visual_tokens'] > 128 or p['query_time_encoder_calls'] != 0
                        or max((t for pair in p['read_times'] for t in pair), default=0) > q['query_time']):
                    raise ValueError('query-time replay, read budget or causality violation')
            rows[source['event_source_group'], offset, d, source['swapped']] = r
    results = {}
    for offset in sorted({s['prefix_seconds'] for s in sources}):
        groups = sorted({s['event_source_group'] for s in sources if s['prefix_seconds'] == offset})
        for delay in DELAYS:
            scores = {a: {'pairs': len(groups), 'both_correct': 0, 'question_correct': 0, 'changed_choice': 0} for a in ARMS}
            coverage = {a: {'none': 0, 'one_scene': 0, 'both_scenes': 0} for a in ('stored_oracle', 'fifo_stored_oracle')}
            for g in groups:
                a, b = [rows[g, offset, delay, swapped] for swapped in (False, True)]
                if a['gold'] == b['gold']:
                    raise ValueError('paired labels did not flip')
                for arm in ARMS:
                    p, q = a['predictions'][arm], b['predictions'][arm]
                    if p['read_frame_indices'] != q['read_frame_indices'] or p['read_times'] != q['read_times']:
                        raise ValueError('paired temporal selection differs')
                    if arm in ('text_only', 'recent4', 'recent8') or (arm == 'prefix4' and offset > 0):
                        if any(p[k] != q[k] for k in ('choice', 'option_logits', 'real_feature_rgb_hashes')):
                            raise ValueError('identical-input control differs')
                    if arm == 'prefix4' and offset == 0:
                        for r, pred in ((a, p), (b, q)):
                            oracle = r['predictions']['evidence_oracle']
                            if any(pred[k] != oracle[k] for k in ('choice', 'option_logits', 'read_times', 'real_feature_rgb_hashes')):
                                raise ValueError('prefix0 equivalence to evidence oracle failed')
                    correct = p['choice'] == a['gold'], q['choice'] == b['gold']
                    scores[arm]['both_correct'] += int(all(correct))
                    scores[arm]['question_correct'] += sum(correct)
                    scores[arm]['changed_choice'] += int(p['choice'] != q['choice'])
                for arm, count in coverage.items():
                    scenes = {int(t[0] - offset) // 2 for t in a['predictions'][arm]['read_times'] if offset <= t[0] < offset + 4}
                    count[('none', 'one_scene', 'both_scenes')[len(scenes)]] += 1
            results[f'offset{offset}/delay{delay}'] = {'pairs': len(groups), 'scores': scores, 'retained_event_coverage': coverage}
    write_json(args.out / ('smoke-ok.json' if args.smoke else 'report.json'), {'completed': True, 'smoke': args.smoke,
        'input_id': inputs['id'], 'questions': len(rows), 'views': len(sources), 'scope': SCOPE, 'results': results,
        'independent_event_sources': len({s['event_source_group'] for s in sources}),
        'pair_conditions': len(sources) // 2, 'architecture_accepted': False, 'no_training': True})


def run(args):
    import sys

    prepare(args)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TRACE_WANDB': '0',
           'TRACE_SWANLAB': '0', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    run_jobs([('gpu-check', [sys.executable, '-c', 'import torch; assert torch.cuda.device_count()==4; '
        'assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24'], list(range(4)))], env, args.out / 'logs')
    command = [sys.executable, '-m', 'trace_vl.core_vlm.longstream', 'worker', '--out', str(args.out)]
    run_jobs([(f'smoke-{i}', [*command, '--shard', str(i), '--smoke'], [i]) for i in range(4)], env, args.out / 'logs')
    args.smoke = True
    report(args)
    smoke = json.loads((args.out / 'smoke-ok.json').read_text())
    if smoke['questions'] != 12 or smoke['views'] != 4 or smoke['independent_event_sources'] != 1:
        raise ValueError('smoke must cover prefix0 and prefix300 in both orders')
    run_jobs([(f'full-{i}', [*command, '--shard', str(i)], [i]) for i in range(4)], env, args.out / 'logs')
    args.smoke = False
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'run', 'report'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--original', type=Path, default=Path('/data/yyy/tracevl-runs/core-longstream-29e5163-001'))
    p.add_argument('--previous', type=Path, default=Path('/data/yyy/tracevl-runs/core-longstream-pairs-3a30174-001'))
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    {'prepare': prepare, 'run': run, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
