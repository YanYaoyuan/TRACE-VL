"""Frozen CausalPatchPool: exact replay, history-off, and module-off controls."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import score_answer, select_records
from .data import read_split, sha256
from .experiment import run_jobs
from .mixed_diagnostics import check_replay
from .native import NativeVideoQALoss, prepare_native
from .pool_probe_cache import (
    FrozenVisionCache,
    observation_key,
    read_frames,
    stage_data,
    validate_observation,
)
from .prepare import digest, write_json
from .temporal import CausalPatchPool, attach_temporal, load_temporal
from .train import checked_checkpoint
from .transfer import ENDPOINTS, audit_observations, collect, comparison, stats

VARIANTS = ('normal', 'history_off', 'module_off')
SCOPE = ('Frozen checkpoint diagnostic, not training or an architecture acceptance gate. '
         'History-off removes only the compressor previous-region condition; the LM still sees all selected frames. '
         'Module-off retains the co-trained LoRA and is not a separately trained static-pooling baseline. '
         'One seed, external development data; unadjusted exploratory video-cluster intervals. '
         'Shared staging/decoding/vision costs are not deployment throughput.')


def versions():
    return {'torch': str(torch.__version__),
            **{name: importlib.metadata.version(name) for name in ('transformers', 'peft')}}


@contextmanager
def intervention(net, variant):
    module = getattr(net, 'temporal', None)
    if variant not in VARIANTS or not isinstance(module, CausalPatchPool):
        raise ValueError('pool intervention requires a restored CausalPatchPool')
    enabled = module.history_enabled
    try:
        if variant == 'history_off':
            module.history_enabled = False
        elif variant == 'module_off':
            del net.temporal
        yield
    finally:
        if not hasattr(net, 'temporal'):
            net.temporal = module
        module.history_enabled = enabled


def load_source(base):
    base = Path(base)
    report = json.loads((base / 'report.json').read_text())
    experiment = json.loads((base / 'experiment.json').read_text())
    if not report.get('completed') or not report.get('training', {}).get('passed'):
        raise ValueError('source comparison must be complete and training-audited')
    cfg = experiment['config']
    checkpoint = base / 'train/causal_pool' / f"checkpoint-{cfg['updates']:06d}"
    train = json.loads((checkpoint.parent / 'run.json').read_text())
    if train['config']['temporal'] != 'causal-pool':
        raise ValueError('source is not the trained causal-pool arm')
    checked = checked_checkpoint(checkpoint, train['id'])
    if checked != report['training']['checkpoints']['causal_pool']:
        raise ValueError('source checkpoint changed')
    cases, sources, references = [], {}, {}
    for dataset in ('ovo', 'nextqa'):
        reference_root = base / 'eval' / dataset / 'causal_pool'
        run = json.loads((reference_root / 'run.json').read_text())
        manifest = Path(run['config']['manifest'])
        if sha256(manifest) != run['manifest_sha256']:
            raise ValueError('frozen evaluation manifest changed')
        for key in ('steps', 'pixels', 'pool_grid', 'timestamp_policy', 'max_new_tokens'):
            if cfg[key] != run['config'][key]:
                raise ValueError(f'source observation protocol differs: {key}')
        if (run['model'] != train['model'] or run['adapter'] != checked
                or run['config']['limit'] or run['config']['split'] != 'val'
                or not run['config'].get('mc_choice_score') or run['config'].get('temporal_off')
                or run['config'].get('block_shuffle_seed') is not None):
            raise ValueError('source model/checkpoint/scoring differs')
        if versions() != {k: run[k] for k in versions()}:
            raise ValueError('installed evaluation libraries differ from source run')
        rows = select_records(read_split(manifest, 'val'), 0)
        maps = {arm: collect(rows, base / 'eval' / dataset / arm) for arm in ('plain', 'causal_pool')}
        audit_observations(rows, maps)
        references[dataset] = maps
        for row in rows:
            prediction = maps['causal_pool'][row['id']]
            validate_observation(row, prediction['observation'], cfg['steps'])
            if any(m[row['id']]['visual_tokens'] != 16 * cfg['steps'] for m in maps.values()):
                raise ValueError('source visual token budget differs')
            cases.append({'dataset': dataset, 'row': row, 'reference': prediction})
        for arm in ('plain', 'causal_pool'):
            for name in ('run.json', 'predictions.jsonl'):
                p = base / 'eval' / dataset / arm / name
                sources[f'{dataset}/{arm}/{name}'] = sha256(p)
        sources[f'{dataset}/manifest'] = sha256(manifest)
    identity = {'protocol': 'causal-pool-fixed-checkpoint-probe-v1', 'source': sources,
                'source_report': sha256(base / 'report.json'), 'checkpoint': checked,
                'model': train['model'], 'config': cfg, 'libraries': versions()}
    identity['id'] = digest(identity)
    return identity, cases, references, checkpoint


def prepare(base, out, data, workers):
    identity, cases, _, checkpoint = load_source(base)
    out, data = Path(out), Path(data)
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    with (data / '.prepare.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_run_manifest(data / 'source.json', identity)
        media = stage_data(cases, identity['config']['model'], identity['model'], data, workers)
        case_text = ''.join(json.dumps(x, sort_keys=True) + '\n' for x in cases)
        cases_hash = hashlib.sha256(case_text.encode()).hexdigest()
        inputs = {**identity, 'cases_sha256': cases_hash, 'checkpoint_path': str(checkpoint), 'media': media}
        ensure_run_manifest(out / 'inputs.json', inputs)
        if (out / 'cases.jsonl').exists() and sha256(out / 'cases.jsonl') != cases_hash:
            raise ValueError('prepared case manifest changed')
        atomic_text(out / 'cases.jsonl', case_text)
        write_json(out / 'preparation.json', {'completed': True, 'source_id': identity['id'], **media})
    return inputs


def prediction_path(out, dataset, variant, qid):
    return Path(out) / 'eval' / dataset / variant / 'questions' / f'{digest(qid)}.json'


def assert_replay(reference, prediction):
    for key in ('id', 'choice', 'token_ids', 'answer', 'visual_tokens', 'observation'):
        if reference[key] != prediction[key]:
            raise ValueError(f'normal replay differs from source: {reference["id"]} {key}')


def generate(net, steps, row, variant, run_id, tokenizer):
    begin = time.monotonic()
    with intervention(net, variant):
        ids, choice = net.generate(steps, row, max_new_tokens=row['_max_new_tokens'], return_choice=True)
        visual = net.prefix(steps, row)[4]['visual_tokens_total']
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return {'id': row['id'], 'run': run_id, 'video': row['video'], 'choice': choice,
            'token_ids': ids, 'answer': text, 'visual_tokens': visual,
            'observation': row['native_layout'].audit, 'intervention': variant,
            'answer_seconds': time.monotonic() - begin, **score_answer(text, row)}


def worker(out, shard, shards, smoke=False):
    out = Path(out)
    inputs = json.loads((out / 'inputs.json').read_text())
    cfg = inputs['config']
    run = json.loads((out / 'run.json').read_text())
    training_id = json.loads((Path(inputs['checkpoint_path']).parent / 'run.json').read_text())['id']
    if checked_checkpoint(Path(inputs['checkpoint_path']), training_id) != inputs['checkpoint']:
        raise ValueError('checkpoint changed since preparation')
    if sha256(out / 'cases.jsonl') != inputs['cases_sha256']:
        raise ValueError('prepared case manifest changed')
    cases = [json.loads(line) for line in (out / 'cases.jsonl').read_text().splitlines()]
    if smoke:
        cases = [item for ds in ('ovo', 'nextqa')
                 for item in [x for x in cases if x['dataset'] == ds][:4]]
    selected = cases[shard::shards]
    if not 0 <= shard < shards:
        raise ValueError('invalid shard')
    # Preserve the source generation/determinism settings.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    with (out / f'.worker-{shard}.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bb = Backbone(BackboneConfig(name=inputs['media']['model'], dtype='bfloat16', local_files_only=True),
                      StreamConfig(pixels_per_frame=cfg['pixels']), device='cuda:0')
        bb.add_lora(cfg['rank'], 2 * cfg['rank'], dropout=0.,
                    adapter=str(Path(inputs['checkpoint_path']) / 'adapter'))
        bb.model.eval().requires_grad_(False)
        net = NativeVideoQALoss(bb, mode='grid', pool_grid=tuple(cfg['pool_grid']),
                               answer_end_token_id=bb.tokenizer.eos_token_id)
        attach_temporal(net, 'causal-pool')
        load_temporal(net, Path(inputs['checkpoint_path']))
        net.eval().requires_grad_(False)
        cache = FrozenVisionCache(out / 'vision-cache', {
            'source_id': inputs['id'], 'model': inputs['model'], 'libraries': versions(),
            'capability': torch.cuda.get_device_capability(0), 'dtype': str(bb.dtype),
            'native': sha256(Path(__file__).with_name('native.py')),
            'backbone': sha256(Path(__file__).parent / '../v3/backbone.py')})
        verified = False
        def progress(number):
            write_json(out / f'progress-{shard}.json', {'run': run['id'], 'shard': shard,
                'phase': 'smoke' if smoke else 'full', 'done': number, 'target': len(selected),
                'updated_unix': time.time()})

        with torch.no_grad():
            for number, item in enumerate(selected, 1):
                ds, row, ref = item['dataset'], dict(item['row']), item['reference']
                paths = {v: prediction_path(out, ds, v, row['id']) for v in VARIANTS}
                if all(p.is_file() for p in paths.values()):
                    saved = {v: json.loads(p.read_text()) for v, p in paths.items()}
                    assert_replay(ref, saved['normal'])
                    if any(vv['run'] != run['id'] or vv['intervention'] != v for v, vv in saved.items()):
                        raise ValueError('saved intervention/run differs')
                    progress(number)
                    continue
                begin = time.monotonic()
                audit = ref['observation']
                images = read_frames(Path(inputs['media']['frames']) / f'{observation_key(audit)}.npz', audit)
                times = np.asarray(audit['frame_times'])
                row['query_time'] = audit['query_time']
                layout = prepare_native(bb.processor, images, times, row, audit,
                    allow_native_timestamp_approximation=cfg['timestamp_policy'] == 'native-index-fps')
                if layout.audit != audit:
                    raise ValueError('replayed native prompt/observation changed')
                steps = cache.get(bb, layout, verify_fresh=not verified)
                row.update(native_layout=layout, _max_new_tokens=cfg['max_new_tokens'])
                shared_seconds = time.monotonic() - begin
                predictions = {v: generate(net, steps, row, v, run['id'], bb.tokenizer) for v in VARIANTS}
                assert_replay(ref, predictions['normal'])
                if not verified:
                    again = generate(net, steps, row, 'normal', run['id'], bb.tokenizer)
                    assert_replay(predictions['normal'], again)
                    verified = True
                    print('VERIFIED exact replay, fresh/cache tensor equality and intervention reset', flush=True)
                for v, p in paths.items():
                    write_json(p, {**predictions[v], 'shared_preparation_seconds': shared_seconds})
                progress(number)
                print(json.dumps({'dataset': ds, 'question': row['id'], 'step': number, 'done': number,
                                  'total': len(selected), 'vision_hits': cache.hits,
                                  'vision_writes': cache.writes}), flush=True)
        write_json(out / f'{"smoke" if smoke else "full"}-shard-{shard}.json',
                   {'completed': True, 'run': run['id'],
                    'cases': [[x['dataset'], x['row']['id']] for x in selected]})


def answer_changes(rows, left, right):
    """Distinguish unchanged answers from cancelling gains and losses."""
    changed = [r for r in rows if left[r['id']]['choice'] != right[r['id']]['choice']]
    return {'n': len(rows), 'choice_changed': len(changed),
            'choice_changed_fraction': len(changed) / len(rows) if rows else None,
            'generated_tokens_changed': sum(left[r['id']]['token_ids'] != right[r['id']]['token_ids']
                                            for r in rows),
            'left_only_correct': sum(left[r['id']]['choice'] == r['answer'] for r in changed),
            'right_only_correct': sum(right[r['id']]['choice'] == r['answer'] for r in changed),
            'both_wrong_changed': sum(left[r['id']]['choice'] != r['answer']
                                      and right[r['id']]['choice'] != r['answer'] for r in changed),
            'changed_ids': [r['id'] for r in changed]}


def make_report(cases, maps, references):
    results = {}
    for ds in ('ovo', 'nextqa'):
        rows = [x['row'] for x in cases if x['dataset'] == ds]
        expected = {x['id'] for x in rows}
        current = maps[ds]
        if set(current) != set(VARIANTS) or any(set(m) != expected for m in current.values()):
            raise ValueError('incomplete or extra intervention coverage')
        for variant, values in current.items():
            if any(p['intervention'] != variant for p in values.values()):
                raise ValueError('incorrect intervention')
            for row in rows:
                choice = values[row['id']]['choice']
                if type(choice) is not int or not 0 <= choice < len(row['options']):
                    raise ValueError('invalid intervention choice')
        check_replay(rows, references[ds]['causal_pool'], current['normal'])
        for row in rows:
            assert_replay(references[ds]['causal_pool'][row['id']], current['normal'][row['id']])
            if len({m[row['id']]['visual_tokens'] for m in current.values()}) != 1:
                raise ValueError('intervention token budgets differ')
        audit_observations(rows, current)
        current = {**current, 'plain': {qid: p for qid, p in references[ds]['plain'].items() if qid in expected}}
        subsets = {'all': rows, **{key: [r for r in rows if r['task'] in tasks]
                                   for key, tasks in ENDPOINTS[ds].items()}}
        contrasts = (('normal', 'history_off'), ('normal', 'module_off'),
                     ('history_off', 'module_off'), ('module_off', 'plain'))
        results[ds] = {'scores': {v: stats(rows, m) for v, m in current.items()},
            'paired': {f'{a}_minus_{b}': comparison(rows, current[a], current[b], ds)
                       for a, b in contrasts},
            'answer_changes': {f'{a}_minus_{b}': {name: answer_changes(subset, current[a], current[b])
                                                for name, subset in subsets.items()}
                               for a, b in contrasts}}
    return {'completed': True, 'training_launched': False, 'scope': SCOPE, 'results': results,
            'next_step': 'Inspect history-off vs normal and module-off before a new architecture/training recipe. '
                         'These diagnostics do not change the original +2 pp gate.'}


def report(base, out, smoke=False):
    out = Path(out)
    inputs = json.loads((out / 'inputs.json').read_text())
    identity, cases, references, _ = load_source(base)
    if identity['id'] != inputs['id']:
        raise ValueError('source experiment changed')
    if smoke:
        cases = [item for ds in ('ovo', 'nextqa')
                 for item in [x for x in cases if x['dataset'] == ds][:4]]
    maps = {ds: {v: {} for v in VARIANTS} for ds in ('ovo', 'nextqa')}
    run_id = json.loads((out / 'run.json').read_text())['id']
    for item in cases:
        for v in VARIANTS:
            p = json.loads(prediction_path(out, item['dataset'], v, item['row']['id']).read_text())
            if p['run'] != run_id:
                raise ValueError('prediction run differs')
            maps[item['dataset']][v][item['row']['id']] = p
    result = make_report(cases, maps, references)
    result['smoke'] = smoke
    if not smoke:
        for ds, variants in maps.items():
            for name, predictions in variants.items():
                folder = out / 'eval' / ds / name
                atomic_text(folder / 'predictions.jsonl', ''.join(json.dumps(p) + '\n'
                            for p in predictions.values()))
                write_json(folder / 'report.json', stats([x['row'] for x in cases if x['dataset'] == ds],
                                                        predictions))
    write_json(out / ('smoke-ok.json' if smoke else 'report.json'), result)
    if not smoke:
        lines = ['TRACE-VL 固定检查点机制验证', '', SCOPE, '']
        for ds, values in result['results'].items():
            for name, score in values['scores'].items():
                lines.append(f'{ds}/{name}: {100 * score["choice_accuracy"]:.3f}% (n={score["n"]})')
            for name, paired_values in values['paired'].items():
                value = paired_values['primary']['choice']
                lines.append(f'{ds}/{name}: primary {value["delta_pp"]:+.3f} pp; CI {value["ci95_pp"]}')
                changes = values['answer_changes'][name]['primary']
                lines.append(f'  primary answer changes: {changes["choice_changed"]}/{changes["n"]}; '
                             f'left-only correct={changes["left_only_correct"]}, '
                             f'right-only correct={changes["right_only_correct"]}')
        atomic_text(out / 'digest.txt', '\n'.join(lines) + '\n')
    return result


def run(args):
    locations = [Path(p).resolve() for p in (args.base, args.out, args.data)]
    if any(a == b or a in b.parents or b in a.parents
           for i, a in enumerate(locations) for b in locations[i + 1:]):
        raise ValueError('source, output and staged data must be separate directories')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.launcher-python.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = prepare(args.base, out, args.data, args.workers)
        code = {name: sha256(Path(__file__).parent / name) for name in (
            'pool_probe.py', 'pool_probe_cache.py', 'temporal.py', 'native.py', 'qa.py',
            'data.py', 'control.py', 'train.py', 'transfer.py', 'experiment.py',
            'prepare.py', 'mixed_diagnostics.py', '../v3/backbone.py',
            '../v3/positions.py', '../v3/run_io.py')}
        identity = {'source_id': inputs['id'], 'variants': VARIANTS, 'code': code,
                    'devices': args.gpus, 'input_file': sha256(out / 'inputs.json')}
        identity['id'] = digest(identity)
        ensure_run_manifest(out / 'run.json', identity)
        if args.prepare_only:
            print('PREPARED exact local observations; no model inference', flush=True)
            return
        if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
            raise ValueError('distinct nonnegative GPU indices required')
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        if not torch.cuda.is_available() or max(args.gpus) >= torch.cuda.device_count():
            raise ValueError('requested CUDA devices are unavailable')
        free = {i: torch.cuda.mem_get_info(i)[0] / 2**30 for i in args.gpus}
        if min(free.values()) < args.minimum_free_gib:
            raise ValueError(f'GPU free GiB {free}; need {args.minimum_free_gib} per selected GPU')
        for smoke in (True, False):
            print(f'{"SMOKE" if smoke else "FULL"}: same checkpoint, three settings per question', flush=True)
            cmd = [sys.executable, '-m', 'trace_vl.core_vlm.pool_probe', '--out', str(out),
                   '--worker', '--shards', str(len(args.gpus))]
            if smoke:
                cmd += ['--smoke']
            run_jobs([(f'{"smoke" if smoke else "full"}-{i}', [*cmd, '--shard', str(i)], [gpu])
                      for i, gpu in enumerate(args.gpus)], env, out / 'logs')
            report(args.base, out, smoke)
            if smoke and args.smoke_only:
                print(f'VERIFIED POOL SMOKE: {out / "smoke-ok.json"}', flush=True)
                return
        print(f'VERIFIED POOL MECHANISM DIAGNOSTIC: {out / "digest.txt"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--minimum-free-gib', type=float, default=12)
    args = parser.parse_args()
    if args.worker:
        worker(args.out, args.shard, args.shards, args.smoke)
    else:
        if (args.base is None or args.data is None or not 1 <= args.workers <= 64
                or not np.isfinite(args.minimum_free_gib) or args.minimum_free_gib <= 0):
            parser.error('--base, --data and 1..64 workers required')
        run(args)


if __name__ == '__main__':
    main()
