"""Matched evaluation of the starting and two trained readers on reused development data."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.b0_window40 import run as b0
from trace_vl.complementary_memory import native_state as native
from trace_vl.complementary_memory.analysis import question_metrics
from trace_vl.complementary_memory.state_stream import Block, NativeStream, replay, score
from trace_vl.core_vlm import longstream
from trace_vl.core_vlm.longstream_delayed import CausalEvents, events_for
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VisualStep, grid_pool
from trace_vl.reservoir_capacity.supervisor import atomic_json, budget, run_jobs
from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from .control import checkpoint_identity
from .data import sha256
from .train import checked_checkpoint

VARIANTS = ('plain', 'trained_reservoir', 'trained_window40')
POLICIES = ('reservoir17', 'window40', 'recent_replay8', 'text_only')
TRAIN_ROOT = Path('/data/yyy/tracevl-runs/endpoint-reader-pilot-798eee2-001')
B0_ROOT = Path('/data/yyy/tracevl-runs/b0-window40-5e62524-001')
PRIMARY = ('trained_system', 'reservoir_training')
CONTRASTS = {
    'trained_system': {('trained_reservoir', 'reservoir17'): 1, ('trained_window40', 'window40'): -1},
    'reservoir_training': {('trained_reservoir', 'reservoir17'): 1, ('plain', 'reservoir17'): -1},
    'window_training': {('trained_window40', 'window40'): 1, ('plain', 'window40'): -1},
    'same_reservoir_policy': {('trained_reservoir', 'reservoir17'): 1,
                              ('trained_window40', 'reservoir17'): -1},
    'same_window_policy': {('trained_reservoir', 'window40'): 1, ('trained_window40', 'window40'): -1},
}


def read(path):
    return json.loads(Path(path).read_text())


def cluster_paired(values, clusters):
    """Preserve the source-weighted point estimate while resampling byte clusters."""
    if len(values) != len(clusters) or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('paired finite source values and cluster IDs required')
    groups = {}
    for value, cluster in zip(values, clusters, strict=True):
        groups.setdefault(cluster, []).append(value)
    if len(groups) < 2:
        raise ValueError('at least two independent byte clusters required')
    arrays = list(groups.values())
    sums, sizes = np.array([sum(v) for v in arrays]), np.array([len(v) for v in arrays])
    index = np.random.default_rng(7).integers(0, len(groups), (10000, len(groups)))
    means = sums[index].sum(1) / sizes[index].sum(1)
    return {'mean': float(np.mean(values)), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'ci97_5': np.quantile(means, [.0125, .9875]).tolist(),
            'sources': len(values), 'byte_clusters': len(groups)}


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = read(b0.REFERENCE / 'inputs.json')
    if (base['id'] != b0.REFERENCE_ID or sha256(b0.REFERENCE / 'report.json') != b0.REFERENCE_REPORT_SHA
            or base['id'] != digest({k: v for k, v in base.items() if k != 'id'})
            or base['code'] != native.code_identity() or base['versions'] != native.versions()):
        raise ValueError('frozen reference identity/helper/runtime changed')
    if checkpoint_identity(base['model']) != base['model_identity']:
        raise ValueError('base model changed')
    b0_input = read(B0_ROOT / 'inputs.json')
    if b0_input['reference'] != base or not read(B0_ROOT / 'report.json')['completed']:
        raise ValueError('window40 reference differs')
    if sha256(B0_ROOT / 'report.json') != 'c3c0c324f058f8ae12d5fb95f12dff2944cb477a5e547496badc9ccbc3ed8654':
        raise ValueError('window40 reference report changed')
    if len(base['sources']) != 77 or sum(len(s['queries']) for s in base['sources']) != 385:
        raise ValueError('all frozen sources/questions required')
    checkpoint = checked_checkpoint(Path(base['plain']), base['checkpoint']['run'])
    if checkpoint != base['checkpoint']:
        raise ValueError('starting adapter changed')
    checkpoints = {'plain': {'path': base['plain'], 'receipt': checkpoint}}
    for variant, arm in [('trained_reservoir', 'reservoir'), ('trained_window40', 'window40')]:
        root = args.pilot / arm
        report, run = read(root / 'report.json'), read(root / 'run.json')
        if (not report['completed'] or report['completed_sources'] != 64
                or run['initialization']['model_identity'] != base['model_identity']
                or run['initialization']['checkpoint'] != base['checkpoint']):
            raise ValueError('complete matched training initialization required')
        path = root / 'checkpoint-000064'
        checkpoints[variant] = {'path': str(path), 'receipt': checked_checkpoint(path, run['id'])}
    train_manifest = Path('/data/yyy/tracevl-runs/finevideo-reader-pilot-data-20261127-001/native-endpoint-pilot.private.jsonl')
    if sha256(train_manifest) != '32ca147b87be5441d3d79f7a453b361c44a04a3ae54fa49f561f1d8d1f1b91ef':
        raise ValueError('training manifest changed')
    train_rows = [json.loads(line) for line in train_manifest.read_text().splitlines()]
    if {r['video_sha256'] for r in train_rows} & {s['video_sha256'] for s in base['sources']}:
        raise ValueError('training/development exact-byte overlap')
    reference_hashes = {}
    for source in base['sources']:
        for q in source['queries']:
            name = f"{digest(q['id'])}.json"
            old, window = read(b0.REFERENCE / 'questions' / name), read(B0_ROOT / 'questions' / name)
            if (old['id'] != q['id'] or old['gold'] != q['answer'] or not old['completed']
                    or old['input_id'] != base['id'] or window['id'] != q['id']
                    or not window['completed'] or window['input_id'] != b0_input['id']):
                raise ValueError('reference question changed')
            reference_hashes[q['id']] = [sha256(b0.REFERENCE / 'questions' / name),
                                         sha256(B0_ROOT / 'questions' / name)]
    protocol = Path(__file__).parents[3] / 'docs/complementary_memory/endpoint_reader_evaluation.md'
    inputs = {'protocol': 'matched-endpoint-reader-evaluation-v1', 'reference': base,
              'checkpoints': checkpoints, 'reference_hashes': reference_hashes,
              'training_manifest_sha256': sha256(train_manifest),
              'code': {'evaluator': sha256(__file__), **native.code_identity()}, 'versions': native.versions(),
              'policies': list(POLICIES), 'memory_seed': 17, 'protocol_sha256': sha256(protocol),
              'scope': 'reused development diagnostic; single training/retention seed; not official benchmark',
              'shards': 2, 'primary': list(PRIMARY), 'primary_delay': 300,
              'comparison': 'negative NLL differences favor the first named system',
              'bootstrap': {'draws': 10000, 'seed': 7, 'cluster': 'exact video bytes',
                            'co_primary_ci': .975}, 'budgets': {'smoke_seconds': 2400, 'full_seconds': 21600}}
    inputs['id'] = digest(inputs)
    ensure_run_manifest(args.out / 'inputs.json', inputs)


def load_inputs(out):
    x = read(out / 'inputs.json')
    if (x['id'] != digest({k: v for k, v in x.items() if k != 'id'})
            or x['code'] != {'evaluator': sha256(__file__), **native.code_identity()}
            or x['versions'] != native.versions()):
        raise ValueError('evaluation identity changed')
    return x


def frozen_rows(inputs, q):
    name = f"{digest(q['id'])}.json"
    paths = [b0.REFERENCE / 'questions' / name, B0_ROOT / 'questions' / name]
    if [sha256(p) for p in paths] != inputs['reference_hashes'][q['id']]:
        raise ValueError('reference bytes changed during evaluation')
    return [read(p) for p in paths]


@torch.no_grad()
def evaluate_source(bb, inputs, source, out, variant):
    started = time.monotonic()
    if sha256(source['video']) != source['video_sha256'] or sha256(source['metadata']) != source['metadata_sha256']:
        raise ValueError('source media changed')
    metadata = read(source['metadata'])
    indices, times, _ = longstream.schedule(metadata, max(q['query_time'] for q in source['queries']))
    if indices.tolist() != source['sampled_frame_indices'] or times.tolist() != source['sampled_frame_times']:
        raise ValueError('frozen causal observation changed')
    fps = float(Fraction(metadata['streams'][0]['avg_frame_rate']))
    meta = metadata['streams'][0]
    hw = frame_size(meta['height'], meta['width'], inputs['reference']['pixels'], 32)
    queries = {q['id']: q for q in source['queries']}
    prompts = {k: native.native_prompt(bb, q, indices, times, metadata, hw) for k, q in queries.items()}
    head = next(iter(prompts.values()))[0]
    if any(h != head for h, _ in prompts.values()):
        raise ValueError('question changed live prompt head')
    live = {'reservoir17': NativeStream(bb, head, 'reservoir', recent=8, capacity=32,
                                       seed=native.memory_seed(source['source_group'], 17)),
            'window40': NativeStream(bb, head, 'window', recent=40)}
    recent, snapshots, completed = deque(maxlen=8), {}, []
    parity_parts, parity = ([], [], []), None
    torch.cuda.reset_peak_memory_stats()
    gaps = []

    def ask(q):
        if live['window40'].seen != q['planned_encoder_calls']:
            raise ValueError('query has wrong causal prefix')
        old, window = frozen_rows(inputs, q)
        if (snapshots[q['target_id']] != old['snapshot_steps']
                or live['window40'].counter.pos != old['streams']['window_stream']['next_position']):
            raise ValueError('snapshot/position differs from frozen reference')
        tail, options = prompts[q['id']][1], len(q['options'])
        before = {k: (s.snapshot(), s.rng.getstate()) for k, s in live.items()}
        predictions = {k: score(bb, s.fork(), s.counter.pos, tail, options) for k, s in live.items()}
        fresh = replay(bb, head, list(recent))
        predictions['recent_replay8'] = score(bb, fresh.fork(), fresh.counter.pos, tail, options)
        text = longstream.text_answer(bb, q, source['video_sha256'])
        predictions['text_only'] = {'choice': text['choice_fp32'], 'option_logits_fp32': text['option_logits_fp32']}
        for key, s in live.items():
            s.verify(before[key][0])
            if s.rng.getstate() != before[key][1]:
                raise ValueError('question changed retention RNG')
        if len(completed) == 0:
            for key, s in live.items():
                if score(bb, s.fork(), s.counter.pos, tail, options) != predictions[key]:
                    raise ValueError('repeat question is not deterministic')
                s.verify(before[key][0])
        if variant == 'plain':
            anchors = {'reservoir17': old['predictions']['reservoir17:S1K1'],
                       'window40': window['predictions']['window40'],
                       'recent_replay8': old['predictions']['recent_replay'],
                       'text_only': old['predictions']['text_only']}
            for key in POLICIES:
                gaps.append(b0.reproduction(predictions[key], anchors[key], q['answer'])['max_logit_gap'])
        for prediction in predictions.values():
            values = prediction['option_logits_fp32']
            if len(values) != options or not all(math.isfinite(v) for v in values):
                raise ValueError('nonfinite/wrong answer options')
        row = {'completed': True, 'inputs': inputs['id'], 'variant': variant, 'id': q['id'],
               'source_group': source['source_group'], 'video_sha256': source['video_sha256'],
               'gold': q['answer'], 'delay_seconds': q['delay_seconds'], 'query_time': q['query_time'],
               'target_time': q['target_time'], 'steps_seen': live['window40'].seen,
               'predictions': predictions, 'streams': {k: s.audit() for k, s in live.items()},
               'reference_max_gap': max(gaps) if gaps else None}
        write_json(out / 'questions' / f"{digest(q['id'])}.json", row)
        completed.append(q['id'])

    def dispatch(event):
        when, kind, identity = event
        if kind == 1:
            return ask(queries[identity])
        if not recent or any(b.end > when for b in recent):
            raise ValueError('noncausal anchor snapshot')
        snapshots[identity] = [b.index for b in recent]

    events = CausalEvents(events_for(source['queries']), dispatch)
    pixels = longstream.stream_pixels(source['video'], indices, *hw)
    try:
        for i, frames in enumerate(pixels):
            end = float(times[i][-1])
            events.due(end, inclusive=False)
            values, grid = bb.preprocess(frames)
            dense = bb.encode(values, grid)
            spatial = bb.merged_grid(grid)
            tokens, rows, cols = grid_pool(dense, spatial, (4, 4))
            block = Block(i, end, tuple(native.stamp_ids(bb, indices[2*i:2*i+2], fps)),
                          tokens.detach(), rows, cols, spatial)
            recent.append(block)
            for s in live.values():
                s.append(block)
                if s.audit()['kv_steps'] > 40:
                    raise ValueError('retained step budget exceeded')
            if parity is None:
                for store, value in zip(parity_parts, (block, VisualStep(dense, spatial, end,
                                               tuple(map(float, times[i]))), frames), strict=True):
                    store.append(value)
                if len(parity_parts[0]) == native.PARITY_STEPS:
                    q = min(queries.values(), key=lambda r: r['query_time'])
                    parity = native.check_parity(bb, head, prompts[q['id']][1], *parity_parts,
                                                 q, indices, times, metadata)
                    parity_parts = ([], [], [])
            events.due(end, inclusive=True)
            if i % 60 == 0:
                print(json.dumps({'variant': variant, 'source': source['source_group'],
                                  'step': i + 1, 'total_steps': len(times)}), flush=True)
        events.due(float('inf'), inclusive=True)
    finally:
        pixels.close()
    if len(completed) != len(queries) or events.cursor != len(events.events) or parity is None:
        raise ValueError('incomplete source/questions/native parity')
    torch.cuda.synchronize()
    write_json(out / 'sources' / f"{digest(source['source_group'])}.json", {
        'completed': True, 'inputs': inputs['id'], 'source_group': source['source_group'],
        'steps': len(times), 'questions': len(completed), 'seconds': time.monotonic() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'parity': parity, 'reference_max_gap': max(gaps) if gaps else None})


def selected(inputs, smoke):
    base = inputs['reference']
    return [s for s in base['sources'] if not smoke or s['source_group'] in base['smoke_source_groups']]


def worker(args):
    inputs = load_inputs(args.out)
    config = inputs['checkpoints'][args.variant]
    if checked_checkpoint(Path(config['path']), config['receipt']['run']) != config['receipt']:
        raise ValueError('evaluation adapter changed')
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    bb = Backbone(BackboneConfig(name=inputs['reference']['model'], dtype='bfloat16', local_files_only=True),
                  StreamConfig(pixels_per_frame=inputs['reference']['pixels']), device='cuda:0')
    bb.add_lora(8, 16, dropout=0., adapter=str(Path(config['path']) / 'adapter'))
    bb.model.eval().requires_grad_(False)
    out = args.out / args.variant
    for source in selected(inputs, args.smoke)[args.shard::2]:
        done = out / 'sources' / f"{digest(source['source_group'])}.json"
        if done.exists():
            old = read(done)
            if not old['completed'] or old['inputs'] != inputs['id']:
                raise ValueError('source completion belongs to another experiment')
            continue
        evaluate_source(bb, inputs, source, out, args.variant)


def report(args):
    inputs = load_inputs(args.out)
    sources = selected(inputs, args.smoke)
    rows, audits = {}, []
    for variant in VARIANTS:
        rows[variant] = {}
        for s in sources:
            audit = read(args.out / variant / 'sources' / f"{digest(s['source_group'])}.json")
            if not audit['completed'] or audit['inputs'] != inputs['id']:
                raise ValueError('source incomplete')
            audits.append(audit)
            for q in s['queries']:
                row = read(args.out / variant / 'questions' / f"{digest(q['id'])}.json")
                if (not row['completed'] or row['inputs'] != inputs['id'] or row['id'] != q['id']
                        or row['variant'] != variant or row['gold'] != q['answer']
                        or row['steps_seen'] != q['planned_encoder_calls']
                        or row['query_time'] != q['query_time'] or set(row['predictions']) != set(POLICIES)):
                    raise ValueError('wrong/incomplete query coverage')
                rows[variant][q['id']] = row
    result = {'completed': True, 'smoke': args.smoke, 'inputs': inputs['id'],
              'sources_per_variant': len(sources), 'questions_per_variant': len(rows['plain']),
              'peak_allocated_gib': max(a['peak_allocated_gib'] for a in audits),
              'max_native_parity_gap': max(a['parity']['max_letter_logit_gap'] for a in audits),
              'max_reference_gap': max(a['reference_max_gap'] or 0 for a in audits),
              'max_seconds_per_source_step': max(a['seconds'] / a['steps'] for a in audits)}
    if args.smoke:
        write_json(args.out / 'smoke-ok.json', result)
        return result
    if len(sources) != 77 or any(len(v) != 385 for v in rows.values()):
        raise ValueError('all 77/385 per variant required before quality analysis')
    analysis = {}
    for delay in sorted({q['delay_seconds'] for s in sources for q in s['queries']}):
        ids = [q['id'] for s in sources for q in s['queries'] if q['delay_seconds'] == delay]
        clusters = [rows['plain'][i]['video_sha256'] for i in ids]
        values = [{(variant, policy): question_metrics(rows[variant][i]['predictions'][policy]['option_logits_fp32'],
                                                       rows[variant][i]['gold'])
                   for variant in VARIANTS for policy in POLICIES} for i in ids]
        arms = {f'{v}/{p}': {m: float(np.mean([x[v, p][m] for x in values])) for m in ('nll', 'acc')}
                for v in VARIANTS for p in POLICIES}
        contrasts = {name: {m: cluster_paired([sum(w * x[key][m] for key, w in weights.items())
                                              for x in values], clusters) for m in ('nll', 'acc')}
                     for name, weights in CONTRASTS.items()}
        analysis[str(delay)] = {'arms': arms, 'contrasts': contrasts, 'sources': len(ids),
                                'exact_byte_clusters': len(set(clusters))}
    result.update(analysis=analysis, scope=inputs['scope'], official_benchmark=False,
                  perception_scope='0s anchored-state recall is a perception proxy; later questions target old state')
    write_json(args.out / 'report.json', result)
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args)
        inputs = load_inputs(args.out)
        gpus = [int(x) for x in args.gpus.split(',')]
        if len(gpus) != 3 or len(set(gpus)) != 3:
            raise ValueError('three distinct GPUs, two source workers per device required')
        state = {'stage': 'smoke', 'started_unix': time.time(), 'inputs': inputs['id'], 'gpus': gpus}
        def resources():
            snap = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                            '--format=csv,noheader,nounits'], text=True)
            free = {int(l.split(',')[0]): int(l.split(',')[1]) for l in snap.splitlines()}
            if any(free[g] < 28 * 1024 for g in gpus):
                raise ValueError('two workers need at least28GiB free per selected GPU')
            return snap
        def jobs(smoke):
            result = []
            for variant, gpu in zip(VARIANTS, gpus, strict=True):
                for shard in range(2):
                    command = [sys.executable, '-m', 'trace_vl.core_vlm.endpoint_reader_eval', 'worker',
                               '--out', str(args.out), '--variant', variant, '--shard', str(shard)]
                    result.append((f'{variant}-{shard}', command + (['--smoke'] if smoke else []), [gpu]))
            return result
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        try:
            state['gpu_before_smoke'] = resources()
            atomic_json(args.out / 'run-status.json', state)
            with budget(2400) as deadline:
                run_jobs(jobs(True), env, args.out / 'logs/smoke', deadline, args.out / 'smoke-jobs.json')
                args.smoke = True
                smoke = report(args)
                if smoke['peak_allocated_gib'] > 13:
                    raise ValueError('smoke exceeds13GiB owned allocation per worker')
                omitted = set(inputs['reference']['smoke_source_groups'])
                shard_steps = [sum(max(q['planned_encoder_calls'] for q in s['queries'])
                                   for s in inputs['reference']['sources'][i::2]
                                   if s['source_group'] not in omitted) for i in range(2)]
                estimate = max(shard_steps) * smoke['max_seconds_per_source_step'] * 1.4 + 300
                gate = {'passed': estimate < 21600, 'estimated_full_seconds': estimate,
                        'maximum_remaining_steps_per_worker': max(shard_steps),
                        'source_time_safety_factor': 1.4, 'startup_allowance_seconds': 300}
                atomic_json(args.out / 'cost-gate.json', gate)
                if not gate['passed']:
                    raise ValueError('full cost projection exceeds6h budget')
            if args.smoke_only:
                state['stage'] = 'smoke_completed'
                return
            state.update(stage='full', full_started_unix=time.time(), gpu_before_full=resources())
            atomic_json(args.out / 'run-status.json', state)
            with budget(21600) as deadline:
                run_jobs(jobs(False), env, args.out / 'logs/full', deadline, args.out / 'full-jobs.json')
                args.smoke = False
                report(args)
                state['stage'] = 'completed'
        except BaseException as exc:
            state.update(stage='held_or_failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            state['updated_unix'] = time.time()
            atomic_json(args.out / 'run-status.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('run', 'worker', 'report'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pilot', type=Path, default=TRAIN_ROOT)
    p.add_argument('--gpus', default='0,1,2')
    p.add_argument('--variant', choices=VARIANTS, default='plain')
    p.add_argument('--shard', type=int, choices=(0, 1), default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-only', action='store_true')
    args = p.parse_args()
    {'run': run, 'worker': worker, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
