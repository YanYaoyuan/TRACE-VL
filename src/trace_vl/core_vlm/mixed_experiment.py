"""Fresh-base matched mixture pilot: ordinary LoRA vs causal temporal + LoRA."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
from pathlib import Path

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import select_records
from .data import read_split, sha256
from .experiment import run_jobs, shared_args, training_command
from .mixed_data import prepare
from .prepare import digest, write_json
from .train import checked_checkpoint, tensor_digest
from .transfer import collect, external_report

ARMS = {'plain': 'none', 'temporal': 'causal'}


def audit_training(root, target, *, arms=None, config_differences=()):
    """Check real saved checkpoints and realized sample order, not log length alone."""
    import torch
    root = Path(root)
    arms = ARMS if arms is None else arms
    runs, logs, initial, complete = {}, {}, {}, {}
    for name in arms:
        folder = root / name
        runs[name] = json.loads((folder / 'run.json').read_text())
        initial[name] = json.loads((folder / 'initialization.json').read_text())
        checkpoint = folder / f'checkpoint-{target:06d}'
        complete[name] = checked_checkpoint(checkpoint, runs[name]['id'])
        if complete[name]['update'] != target or initial[name]['run'] != runs[name]['id']:
            raise ValueError('not the declared final checkpoint/initialization')
        logs[name] = [json.loads(x) for x in (folder / 'metrics.jsonl').read_text().splitlines()]
        if [x['update'] for x in logs[name]] != list(range(1, target + 1)):
            raise ValueError('training coverage differs')
        if any(not math.isfinite(x['loss']) or not math.isfinite(x['gradient_norm'])
               for x in logs[name]):
            raise ValueError('nonfinite training')
        state = torch.load(checkpoint / 'training.pt', map_location='cpu', weights_only=True)
        if tensor_digest({k: v for k, v in state['parameters'].items() if 'lora_' in k}) == initial[name]['lora_sha256']:
            raise ValueError('LoRA did not change')
        if arms[name] == 'causal':
            up = state['parameters'].get('temporal.up.weight')
            if up is None or not torch.isfinite(up).all() or not up.count_nonzero():
                raise ValueError('temporal residual did not learn')
            if any(x['temporal_gradient_norm_after_clip'] <= 0 for x in logs[name]):
                raise ValueError('temporal QA gradients missing')
    for key in ('model', 'manifest_sha256', 'ids', 'videos', 'world_size', 'objective'):
        if any(runs[name][key] != runs['plain'][key] for name in arms):
            raise ValueError(f'unmatched training {key}')
    for name in arms:
        if runs[name]['initial_checkpoint'] is not None or runs[name]['config']['temporal'] != arms[name]:
            raise ValueError('must start from original base in the declared arm')
    def config(name):
        return {k: v for k, v in runs[name]['config'].items() if k not in {'temporal', *config_differences}}
    if any(config(name) != config('plain') or initial[name]['lora_sha256'] != initial['plain']['lora_sha256'] for name in arms):
        raise ValueError('unmatched schedule or LoRA initialization')
    def schedule(name):
        return [(x['question_ids_by_rank'], x['lr'], x['answer_tokens'], x.get('teacher_questions', 0)) for x in logs[name]]
    if any(schedule(name) != schedule('plain') for name in arms):
        raise ValueError('realized sample/LR/token budgets differ')
    return {'passed': True, 'updates': target, 'same_initial_lora': True,
            'same_samples_lr_answer_tokens': True, 'checkpoints': complete,
            'trainable_parameters': {k: v['trainable_parameters'] for k, v in initial.items()},
            'note': 'matched examples/updates/LoRA settings; extra temporal parameters/FLOPs are not compute-matched'}


def matched_report(manifests, directory, names, contrasts):
    result = external_report(manifests, directory, names, contrasts)
    for dataset, manifest in manifests.items():
        rows = read_split(manifest, 'val')
        maps = {name: collect(rows, directory / dataset / name) for name in names}
        for row in rows:
            values = [maps[name][row['id']]['visual_tokens'] for name in names]
            if any(type(v) is not int or v <= 0 for v in values) or len(set(values)) != 1:
                raise ValueError('visual token budgets differ')
        result[dataset]['visual_token_budget_matched'] = True
    return result


def decision(results):
    """New, predeclared feasibility gate. Does not modify the old transfer gate."""
    checks = {}
    for reference in ('plain', 'original_grid'):
        contrast = 'temporal_minus_' + reference
        temporal = results['nextqa']['paired'][contrast]['primary']['choice']
        checks['temporal_gain_vs_' + reference] = bool(
            temporal and temporal['ci95_pp'] and temporal['delta_pp'] >= 2 and temporal['ci95_pp'][0] > 0)
        for dataset, endpoint in (('ovo', 'primary'), ('ovo', 'no_harm'), ('nextqa', 'no_harm')):
            value = results[dataset]['paired'][contrast][endpoint]['choice']
            checks[f'{dataset}_{endpoint}_retained_vs_{reference}'] = bool(
                value and value['ci95_pp'] and value['ci95_pp'][0] > -2)
    return {'passed': all(checks.values()), 'checks': checks,
            'rule': 'NExT temporal +2 pp and CI lower >0 vs plain AND original-grid; '
                    'OVO backward/realtime and NExT descriptive CI lower >-2 pp vs both',
            'meaning': 'single-seed development feasibility; no automatic scaling; failure may be inconclusive'}


def run(cfg, out, prepare_only=False, smoke_only=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.experiment.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cfg['updates'] < 2 or cfg['flow_updates'] != 2 or not cfg['gpus']:
            raise ValueError('two-update smoke and positive full schedule required')
        external_cfg = json.loads((Path(cfg['external_data']) / 'inputs.json').read_text())['config']
        for key in ('model', 'steps', 'pixels', 'pool_grid', 'timestamp_policy'):
            if cfg[key] != external_cfg[key]:
                raise ValueError(f'external observation/model protocol changed: {key}')
        code = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
        identity = {'config': cfg, 'code': code, 'protocol': 'mixed-core-temporal-v1'}
        identity['id'] = digest(identity)
        ensure_run_manifest(out / 'run.json', identity)
        print('[1/5] Fixed training-only mixture; reuse frozen external development data', flush=True)
        manifest, external = prepare(cfg, cfg['data_out'])
        if prepare_only:
            print('PREPARED: no GPU work', flush=True)
            return
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
        gpu_check = ('import torch; '
                     f'assert torch.cuda.device_count()=={len(cfg["gpus"])}; '
                     'free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(torch.cuda.device_count())]; '
                     f'print(free); assert min(free)>={cfg["minimum_free_gib"]}')
        run_jobs([('gpu-check', [sys.executable, '-c', gpu_check], cfg['gpus'])], env, out / 'logs')

        def train(name, flow):
            folder = out / ('smoke-train' if flow else 'train') / name
            cmd = training_command(cfg, manifest, folder, 'grid', flow=flow)
            cmd += ['--temporal', ARMS[name]]  # deliberately NO --init-checkpoint
            run_jobs([(f'{folder.parent.name}-{name}', cmd, cfg['gpus'])], env, out / 'logs')
            updates = cfg['flow_updates'] if flow else cfg['updates']
            path = folder / f'checkpoint-{updates:06d}'
            checked_checkpoint(path, json.loads((folder / 'run.json').read_text())['id'])
            return path

        def evaluate(dataset, path, name, checkpoint, flow=False):
            directory = out / ('smoke-eval' if flow else 'eval') / dataset / name
            cmd = [sys.executable, '-m', 'trace_vl.core_vlm.evaluate', *shared_args(cfg, path),
                   '--out', str(directory), '--mode', 'grid', '--shards', str(len(cfg['gpus'])),
                   '--max-new-tokens', str(cfg['max_new_tokens']), '--mc-choice-score']
            if checkpoint is not None:
                cmd += ['--checkpoint', str(checkpoint)]
            tag = f'{"smoke-" if flow else ""}{dataset}-{name}'
            run_jobs([(f'{tag}-{i}', [*cmd, '--shard', str(i)], [gpu])
                      for i, gpu in enumerate(cfg['gpus'])], env, out / 'logs')
            run_jobs([(tag + '-report', [*cmd, '--report-only'], [])], env, out / 'logs')

        print('[2/5] Two-step DDP QA training AND checkpoint evaluation for both arms', flush=True)
        smoke_manifests = {}
        for dataset, path in external.items():
            rows = select_records(read_split(path, 'val'), 4)
            tiny = out / f'smoke-{dataset}.jsonl'
            atomic_text(tiny, ''.join(json.dumps(r) + '\n' for r in rows))
            smoke_manifests[dataset] = tiny
        for name in ARMS:
            checkpoint = train(name, True)
            for dataset, path in smoke_manifests.items():
                evaluate(dataset, path, name, checkpoint, flow=True)
        smoke = audit_training(out / 'smoke-train', cfg['flow_updates'])
        smoke['evaluation'] = matched_report(smoke_manifests, out / 'smoke-eval', tuple(ARMS), [('temporal', 'plain')])
        write_json(out / 'smoke-ok.json', smoke)
        if smoke_only:
            print('VERIFIED SMOKE: no full training requested', flush=True)
            return
        print('[3/5] Matched fresh-base training, fixed final checkpoints', flush=True)
        checkpoints = {name: train(name, False) for name in ARMS}
        audit = audit_training(out / 'train', cfg['updates'])
        write_json(out / 'training-audit.json', audit)
        print('[4/5] Original grid + plain LoRA + temporal LoRA on identical external observations', flush=True)
        for dataset, path in external.items():
            for name, checkpoint in {'original_grid': None, **checkpoints}.items():
                evaluate(dataset, path, name, checkpoint)
        print('[5/5] Paired choice/format scores and predeclared feasibility report', flush=True)
        results = matched_report(external, out / 'eval', ('original_grid', *ARMS),
            [('temporal', 'plain'), ('temporal', 'original_grid'), ('plain', 'original_grid')])
        report = {'completed': True, 'results': results, 'decision': decision(results), 'training_audit': audit,
                  'scope': 'official NExT-QA train used; NExT validation is in-dataset generalization, '
                           'OVO is cross-dataset development. One seed; fixed final checkpoint; '
                           'ordinary temporal adapter baseline, not an established novelty or official streaming score.'}
        write_json(out / 'report.json', report)
        lines = ['# Core video-model mixed-training pilot', '', json.dumps(report['decision']), '',
                 '| Dataset | Arm | Choice accuracy | Strict accuracy | Format valid |', '|---|---|---:|---:|---:|']
        for dataset, value in results.items():
            for name, score in value['scores'].items():
                lines.append(f"| {dataset} | {name} | {100*score['choice_accuracy']:.2f}% | "
                             f"{100*score['strict_accuracy']:.2f}% | {100*score['format_valid_fraction']:.2f}% |")
        lines.extend(['', report['scope'], '', 'No further training is launched automatically.'])
        atomic_text(out / 'REPORT.md', '\n'.join(lines) + '\n')
        print('VERIFIED MIXED CORE PILOT:', out / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare-only', action='store_true')
    group.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.out, args.prepare_only, args.smoke_only)


if __name__ == '__main__':
    main()
