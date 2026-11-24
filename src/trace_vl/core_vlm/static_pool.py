"""Independently train the missing static-pooling control against frozen references."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import checkpoint_identity, score_answer
from .data import sha256
from .experiment import run_jobs, training_command
from .native import NativeVideoQALoss, prepare_native
from .pool_probe import answer_changes, assert_replay, load_source, versions
from .pool_probe_cache import FrozenVisionCache, observation_key, read_frames
from .prepare import digest, write_json
from .temporal import StaticPatchPool, attach_temporal, load_temporal
from .train import checked_checkpoint, tensor_digest
from .transfer import ENDPOINTS, audit_observations, comparison, stats

SCOPE = ('One additional independently trained static region-pooling arm; the source '
         'Plain and CausalPatchPool references are not retrained. Same seed, samples, '
         'LoRA initialization, learning-rate schedule, frames and 128 visual tokens. '
         'Static pooling retains an inactive predictor matrix for identical initialization '
         'and checkpoint layout; it is audited unchanged, not claimed as active capacity. '
         'Single-seed development comparison, unadjusted video-cluster intervals; '
         'no automatic architecture acceptance or training scale-up. Shared cache '
         'and staging times are not deployment throughput.')


def matched_training(reference, current, reference_initial, initial, reference_logs, logs):
    """Audit realized training, allowing only architecture and byte-identical model relocation."""
    for key in ('model', 'manifest_sha256', 'ids', 'videos', 'world_size', 'objective'):
        if reference[key] != current[key]:
            raise ValueError(f'unmatched training {key}')
    def cfg(r):
        return {k: v for k, v in r['config'].items() if k not in ('model', 'temporal')}
    if (cfg(reference) != cfg(current) or current['config']['temporal'] != 'static-pool'
            or current['initial_checkpoint'] is not None
            or reference_initial['lora_sha256'] != initial['lora_sha256']
            or initial.get('temporal_history_enabled') is not False):
        raise ValueError('static training recipe, fresh initialization or history control differs')
    if not logs or [x['update'] for x in logs] != list(range(1, len(logs) + 1)):
        raise ValueError('static update coverage differs')
    if len(logs) > len(reference_logs):
        raise ValueError('static run exceeded the frozen reference budget')
    fields = ('question_ids_by_rank', 'lr', 'answer_tokens', 'teacher_questions')
    if any(any(a[k] != b[k] for k in fields)
           for a, b in zip(logs, reference_logs, strict=False)):
        raise ValueError('realized sample/LR/token/teacher schedule differs')
    for row in logs:
        if (row.get('temporal_history_enabled') is not False
                or any(not math.isfinite(row[k]) for k in ('loss', 'gradient_norm',
                                                          'temporal_gradient_norm_after_clip'))
                or row['gradient_norm'] <= 0 or row['temporal_gradient_norm_after_clip'] <= 0):
            raise ValueError('static training has invalid gradients or enabled history')
        if row.get('vision_cache_writes_rank0', 0) or not row.get('vision_cache_hits_rank0', 0):
            raise ValueError('static training did not reuse the frozen source cache')
    return {'passed': True, 'updates': len(logs), 'same_initial_lora': True,
            'same_samples_lr_answer_tokens_and_teacher': True, 'history_enabled': False}


def audit_training(base, out, target):
    folder = out / 'train/static_pool'
    run = json.loads((folder / 'run.json').read_text())
    initial = json.loads((folder / 'initialization.json').read_text())
    logs = [json.loads(x) for x in (folder / 'metrics.jsonl').read_text().splitlines()]
    if len(logs) != target:
        raise ValueError('static training did not reach the declared checkpoint')
    checked = checked_checkpoint(folder / f'checkpoint-{target:06d}', run['id'])
    for name in ('plain', 'causal_pool'):
        source = base / 'train' / name
        matched_training(json.loads((source / 'run.json').read_text()), run,
                         json.loads((source / 'initialization.json').read_text()), initial,
                         [json.loads(x) for x in (source / 'metrics.jsonl').read_text().splitlines()], logs)
    state = torch.load(folder / f'checkpoint-{target:06d}/training.pt',
                       map_location='cpu', weights_only=True)['parameters']
    predictor = tensor_digest({'predict.weight': state['temporal.predict.weight']})
    if predictor != initial['predictor_sha256'] or not state['temporal.score.weight'].count_nonzero():
        raise ValueError('inactive predictor changed or static scorer did not learn')
    if tensor_digest({k: v for k, v in state.items() if 'lora_' in k}) == initial['lora_sha256']:
        raise ValueError('static arm LoRA did not learn')
    return {'passed': True, 'updates': target, 'checkpoint': checked,
            'same_initial_lora_and_realized_schedule': True,
            'history_disabled_every_update': True, 'inactive_predictor_unchanged': True,
            'trainable_parameters_including_inactive_predictor': initial['trainable_parameters'],
            'inactive_predictor_parameters': state['temporal.predict.weight'].numel()}


def prepare(args):
    base, observations, out = args.base.resolve(), args.observations.resolve(), args.out.resolve()
    if any(a == out or a in out.parents or out in a.parents for a in (base, observations)):
        raise ValueError('new control requires a separate output directory')
    source, cases, _, _ = load_source(base)
    obs = json.loads((observations / 'inputs.json').read_text())
    completed = json.loads((observations / 'report.json').read_text())
    if (obs['id'] != source['id'] or not completed.get('completed')
            or sha256(observations / 'cases.jsonl') != obs['cases_sha256']):
        raise ValueError('completed observation probe differs from the frozen source')
    if cases != [json.loads(x) for x in (observations / 'cases.jsonl').read_text().splitlines()]:
        raise ValueError('source question/observation coverage changed')
    source_train = json.loads((base / 'train/plain/run.json').read_text())
    for name in ('plain', 'causal_pool'):
        run = json.loads((base / f'train/{name}/run.json').read_text())
        record = checked_checkpoint(base / f'train/{name}/checkpoint-{run["config"]["updates"]:06d}', run['id'])
        if record != json.loads((base / 'report.json').read_text())['training']['checkpoints'][name]:
            raise ValueError('source training checkpoint changed')
    manifest = Path(source_train['config']['manifest'])
    if sha256(manifest) != source_train['manifest_sha256']:
        raise ValueError('source training manifest changed')
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    paths = {Path(row['video']).resolve() for row in rows}
    model = Path(obs['media']['model']).resolve()
    if (not str(model).startswith('/data/') or any(not str(p).startswith('/data/') or not p.is_file()
                                                 for p in paths)):
        raise ValueError('model and training videos must resolve to existing /data paths')
    if checkpoint_identity(model) != source_train['model']:
        raise ValueError('staged model bytes differ from source training')
    dependencies = ('native.py', 'data.py', 'feature_cache.py', '../v3/backbone.py')
    for name in dependencies:
        if sha256(Path(__file__).parent / name) != source_train['code'][name]:
            raise ValueError(f'frozen vision/cache implementation changed: {name}')
    cfg = dict(source['config'])
    cfg['model'] = str(model)
    cfg['gpus'] = list(range(source_train['world_size']))
    if cfg['updates'] != source_train['config']['updates']:
        raise ValueError('source update budgets differ')
    cache = Path(source_train['config']['feature_cache']).resolve()
    if not str(cache).startswith('/data/') or not cache.is_dir():
        raise ValueError('verified training cache missing from local disk')
    if shutil.disk_usage(out).free < 30 * 2**30:
        raise ValueError('insufficient disk reserve for new control checkpoints')
    code = {str(p.relative_to(Path(__file__).parent.parent)): sha256(p)
            for p in Path(__file__).parent.glob('*.py')}
    for name in ('backbone.py', 'positions.py', 'run_io.py'):
        code['v3/' + name] = sha256(Path(__file__).parent / '../v3' / name)
    identity = {'protocol': 'independent-static-pool-control-v1', 'source_id': source['id'],
                'source': source, 'base': str(base), 'observations': str(observations),
                'observation_inputs_sha256': sha256(observations / 'inputs.json'),
                'cases_sha256': obs['cases_sha256'], 'config': cfg, 'manifest': str(manifest),
                'feature_cache': str(cache), 'media': obs['media'], 'code': code,
                'scope': SCOPE, 'revision': os.environ.get('TRACE_REV', 'local')}
    identity['id'] = digest(identity)
    ensure_run_manifest(out / 'inputs.json', identity)
    atomic_text(out / 'cases.jsonl', (observations / 'cases.jsonl').read_text())
    return identity


def selected_cases(out, smoke):
    inputs = json.loads((out / 'inputs.json').read_text())
    if sha256(out / 'cases.jsonl') != inputs['cases_sha256']:
        raise ValueError('prepared case manifest changed')
    cases = [json.loads(x) for x in (out / 'cases.jsonl').read_text().splitlines()]
    if smoke:
        cases = [item for ds in ('ovo', 'nextqa')
                 for item in [x for x in cases if x['dataset'] == ds][:4]]
    return cases


def question_path(out, ds, qid, smoke):
    return out / ('flow-eval' if smoke else 'eval') / ds / 'static_pool/questions' / f'{digest(qid)}.json'


def evaluate_worker(out, shard, shards, smoke):
    inputs = json.loads((out / 'inputs.json').read_text())
    cfg = inputs['config']
    evaluation = json.loads((out / ('flow-eval-run.json' if smoke else 'eval-run.json')).read_text())
    checkpoint = Path(evaluation['checkpoint_path'])
    train_run = json.loads((checkpoint.parent / 'run.json').read_text())
    if checked_checkpoint(checkpoint, train_run['id']) != evaluation['checkpoint']:
        raise ValueError('static checkpoint changed')
    cases = selected_cases(out, smoke)[shard::shards]
    if not 0 <= shard < shards:
        raise ValueError('invalid shard')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    with (out / f'.eval-worker-{shard}.lock').open('a') as guard, torch.no_grad():
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bb = Backbone(BackboneConfig(name=cfg['model'], dtype='bfloat16', local_files_only=True),
                      StreamConfig(pixels_per_frame=cfg['pixels']), device='cuda:0')
        bb.add_lora(cfg['rank'], 2 * cfg['rank'], dropout=0., adapter=str(checkpoint / 'adapter'))
        bb.model.eval().requires_grad_(False)
        net = NativeVideoQALoss(bb, mode='grid', pool_grid=tuple(cfg['pool_grid']),
                               answer_end_token_id=bb.tokenizer.eos_token_id)
        attach_temporal(net, 'static-pool')
        load_temporal(net, checkpoint)
        net.eval().requires_grad_(False)
        source_inputs = inputs['source']
        cache = FrozenVisionCache(Path(inputs['observations']) / 'vision-cache', {
            'source_id': source_inputs['id'], 'model': source_inputs['model'], 'libraries': versions(),
            'capability': torch.cuda.get_device_capability(0), 'dtype': str(bb.dtype),
            'native': sha256(Path(__file__).with_name('native.py')),
            'backbone': sha256(Path(__file__).parent / '../v3/backbone.py')})
        verified, first = False, None

        def generate(steps, row):
            ids, choice = net.generate(steps, row, max_new_tokens=cfg['max_new_tokens'], return_choice=True)
            text = bb.tokenizer.decode(ids, skip_special_tokens=True)
            if not isinstance(net.temporal, StaticPatchPool) or net.temporal.history_enabled:
                raise ValueError('static evaluation enabled historical conditioning')
            return {'id': row['id'], 'run': evaluation['id'], 'video': row['video'],
                    'choice': choice, 'token_ids': ids, 'answer': text,
                    'visual_tokens': net.prefix(steps, row)[4]['visual_tokens_total'],
                    'observation': row['native_layout'].audit, 'architecture': 'static-pool',
                    'history_enabled': False, **score_answer(text, row)}

        for number, item in enumerate(cases, 1):
            ds, row, reference = item['dataset'], dict(item['row']), item['reference']
            path = question_path(out, ds, row['id'], smoke)
            if path.is_file() and not smoke:
                saved = json.loads(path.read_text())
                if saved['id'] != row['id'] or saved['run'] != evaluation['id']:
                    raise ValueError('saved static prediction belongs to another run/question')
            else:
                audit = reference['observation']
                images = read_frames(Path(inputs['media']['frames']) / f'{observation_key(audit)}.npz', audit)
                row['query_time'] = audit['query_time']
                layout = prepare_native(bb.processor, images, np.asarray(audit['frame_times']), row, audit,
                    allow_native_timestamp_approximation=cfg['timestamp_policy'] == 'native-index-fps')
                if layout.audit != audit or not (cache.root / f'{observation_key(audit)}.pt').is_file():
                    raise ValueError('frozen layout changed or existing vision cache missing')
                steps = cache.get(bb, layout, verify_fresh=not verified)
                row['native_layout'] = layout
                prediction = generate(steps, row)
                if first is None:
                    assert_replay(prediction, generate(steps, row))
                    first = (steps, row, prediction)
                    verified = True
                    print('VERIFIED STATIC: checkpoint restore, exact fresh/cache features and repeated answer', flush=True)
                write_json(path, prediction)
            write_json(out / f'progress-{shard}.json', {'phase': 'smoke' if smoke else 'full-evaluation',
                'done': number, 'target': len(cases), 'updated_unix': time.time()})
            print(json.dumps({'dataset': ds, 'done': number, 'total': len(cases),
                              'vision_hits': cache.hits, 'vision_writes': cache.writes}), flush=True)
        if first is not None:
            assert_replay(first[2], generate(first[0], first[1]))
        if cache.writes:
            raise ValueError('evaluation must only read the frozen source vision cache')
        write_json(out / f'{"flow" if smoke else "full"}-shard-{shard}.json', {
            'completed': True, 'run': evaluation['id'], 'cases': [[x['dataset'], x['row']['id']] for x in cases],
            'actual_replay_and_fresh_cache_verified': verified})


def make_report(cases, static, references):
    result = {}
    for ds in ('ovo', 'nextqa'):
        rows = [x['row'] for x in cases if x['dataset'] == ds]
        expected = {r['id'] for r in rows}
        if len(expected) != len(rows) or set(static[ds]) != expected:
            raise ValueError('static evaluation coverage differs')
        for item in (x for x in cases if x['dataset'] == ds):
            p = static[ds][item['row']['id']]
            if (p['observation'] != item['reference']['observation'] or p['visual_tokens'] != 128
                    or p.get('history_enabled') is not False or p.get('architecture') != 'static-pool'
                    or type(p.get('choice')) is not int
                    or not 0 <= p['choice'] < len(item['row']['options'])):
                raise ValueError('static observation, budget, architecture or choice differs')
        maps = {**references[ds], 'static_pool': static[ds]}
        audit_observations(rows, maps)
        subsets = {'all': rows, **{k: [r for r in rows if r['task'] in tasks]
                                   for k, tasks in ENDPOINTS[ds].items()}}
        contrasts = (('causal_pool', 'static_pool'), ('static_pool', 'plain'), ('causal_pool', 'plain'))
        result[ds] = {'scores': {k: stats(rows, v) for k, v in maps.items()},
            'paired': {f'{a}_minus_{b}': comparison(rows, maps[a], maps[b], ds) for a, b in contrasts},
            'answer_changes': {f'{a}_minus_{b}': {k: answer_changes(subset, maps[a], maps[b])
                                                for k, subset in subsets.items()} for a, b in contrasts}}
    return {'completed': True, 'scope': SCOPE, 'newly_trained_arms': ['static_pool'],
            'reused_trained_arms': ['plain', 'causal_pool'], 'architecture_accepted': False,
            'results': result, 'next_step': 'Review static attribution and decide whether a second seed is justified. '
                                               'No automatic scaling; original +2 pp gate is unchanged.'}


def evaluate(inputs, out, smoke, env):
    cfg = inputs['config']
    target = 2 if smoke else cfg['updates']
    checkpoint = out / 'train/static_pool' / f'checkpoint-{target:06d}'
    training = json.loads((checkpoint.parent / 'run.json').read_text())
    identity = {'experiment_id': inputs['id'], 'smoke': smoke,
                'checkpoint_path': str(checkpoint), 'checkpoint': checked_checkpoint(checkpoint, training['id'])}
    identity['id'] = digest(identity)
    ensure_run_manifest(out / ('flow-eval-run.json' if smoke else 'eval-run.json'), identity)
    cmd = [sys.executable, '-m', 'trace_vl.core_vlm.static_pool', '--out', str(out),
           '--worker', '--shards', str(len(cfg['gpus']))]
    if smoke:
        cmd += ['--smoke']
    run_jobs([(f'{"flow" if smoke else "eval"}-{i}', [*cmd, '--shard', str(i)], [gpu])
              for i, gpu in enumerate(cfg['gpus'])], env, out / 'logs')
    cases = selected_cases(out, smoke)
    maps = {ds: {} for ds in ('ovo', 'nextqa')}
    for item in cases:
        p = json.loads(question_path(out, item['dataset'], item['row']['id'], smoke).read_text())
        if p['run'] != identity['id']:
            raise ValueError('static prediction run differs')
        maps[item['dataset']][item['row']['id']] = p
    _, _, refs, _ = load_source(Path(inputs['base']))
    result = make_report(cases, maps, refs)
    if smoke:
        # Scores against fully trained references are not a meaningful smoke comparison.
        result = {'completed': True, 'smoke': True, 'questions': len(cases),
                  'static_checkpoint_update': target, 'matched_observations_and_128_tokens': True}
        write_json(out / 'smoke-ok.json', result)
    else:
        result['training_audit'] = json.loads((out / 'train-audit.json').read_text())
        result['original_decision'] = json.loads((Path(inputs['base']) / 'report.json').read_text())['decision']
        for ds, mapped in maps.items():
            atomic_text(out / f'eval/{ds}/static_pool/predictions.jsonl',
                        ''.join(json.dumps(x) + '\n' for x in mapped.values()))
        write_json(out / 'report.json', result)
        lines = ['TRACE-VL independently trained static-pooling control', '', SCOPE, '']
        for ds, values in result['results'].items():
            for name, score in values['scores'].items():
                lines.append(f'{ds}/{name}: {100 * score["choice_accuracy"]:.3f}% (n={score["n"]})')
            for name, paired in values['paired'].items():
                primary = paired['primary']['choice']
                lines.append(f'{ds}/{name}: primary {primary["delta_pp"]:+.3f} pp; CI {primary["ci95_pp"]}')
        atomic_text(out / 'digest.txt', '\n'.join(lines) + '\n')


def run(args):
    args.out = args.out.resolve()
    for path in (args.base.resolve(), args.observations.resolve()):
        if path == args.out or path in args.out.parents or args.out in path.parents:
            raise ValueError('new control requires a separate output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.experiment.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = prepare(args)
        cfg, out = inputs['config'], args.out
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'TRACE_WANDB': '0', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        gpu_check = ('import torch; '
                     f'assert torch.cuda.device_count()=={len(cfg["gpus"])}; '
                     'free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(torch.cuda.device_count())]; '
                     f'print(free); assert min(free)>={cfg["minimum_free_gib"]}')
        run_jobs([('gpu-check', [sys.executable, '-c', gpu_check], cfg['gpus'])], env, out / 'logs')
        cmd = training_command(cfg, Path(inputs['manifest']), out / 'train/static_pool', 'grid')
        cmd += ['--temporal', 'static-pool', '--feature-cache', inputs['feature_cache'],
                '--preservation-teacher', '--preservation-weight', str(cfg['preservation_weight'])]
        print('STATIC TRAIN SMOKE: first two updates of the full frozen recipe', flush=True)
        smoke_path = out / 'smoke-ok.json'
        if not smoke_path.is_file():
            run_jobs([('train-smoke', [*cmd, '--stop-after', '2'], cfg['gpus'])], env, out / 'logs')
            write_json(out / 'train-smoke-audit.json', audit_training(Path(inputs['base']), out, 2))
            evaluate(inputs, out, True, env)
        smoke_result = json.loads(smoke_path.read_text())
        if (not smoke_result.get('completed') or smoke_result.get('smoke') is not True
                or smoke_result.get('questions') != 8):
            raise ValueError('static training and QA smoke gate did not pass')
        if args.smoke_only:
            print(f'VERIFIED STATIC TRAINING AND QA SMOKE: {smoke_path}', flush=True)
            return
        print('STATIC FULL TRAIN: resume the two audited updates to the fixed 2500-update budget', flush=True)
        run_jobs([('train-full', cmd, cfg['gpus'])], env, out / 'logs')
        write_json(out / 'train-audit.json', audit_training(Path(inputs['base']), out, cfg['updates']))
        evaluate(inputs, out, False, env)
        print(f'VERIFIED INDEPENDENT STATIC CONTROL: {out / "digest.txt"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        evaluate_worker(args.out, args.shard, args.shards, args.smoke)
    else:
        if args.base is None or args.observations is None:
            parser.error('--base and --observations are required')
        run(args)


if __name__ == '__main__':
    main()
