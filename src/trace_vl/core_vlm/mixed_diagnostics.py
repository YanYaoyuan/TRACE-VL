"""Frozen mixed-core checkpoints: module ablation and matched block-order controls.

Evaluation only. These exploratory contrasts do not change the pilot gate.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import select_records
from .data import read_split, sha256
from .experiment import run_jobs, shared_args
from .interventions import block_order
from .prepare import digest, write_json
from .train import checked_checkpoint
from .transfer import ENDPOINTS, audit_observations, collect, comparison, stats

VARIANTS = {
    'temporal_off': ('temporal', ['--temporal-off']),
    'plain_shuffle': ('plain', ['--block-shuffle-seed', '17']),
    'temporal_shuffle': ('temporal', ['--block-shuffle-seed', '17']),
}
SCOPE = ('Frozen final checkpoints and external development populations; one training seed, '
         'exploratory unadjusted source-video bootstrap intervals. No training or scaling. '
         'Shuffle permutes encoded two-frame content blocks at fixed chronological timestamp slots; '
         'it is not raw-frame shuffling and encoder temporal information may remain. '
         'Off removes the temporal residual but keeps its co-trained LoRA; it does not isolate '
         'the effect of training with the module. Neither contrast alone proves temporal reasoning.')


def interaction(rows, maps, samples=4000, seed=7):
    """Direct paired difference in order sensitivity; same sampled videos in all arms."""
    groups = {}
    weights = {'temporal': 1, 'temporal_shuffle': -1, 'plain': -1, 'plain_shuffle': 1}
    for row in rows:
        value = sum(w * int(maps[k][row['id']]['choice'] == row['answer']) for k, w in weights.items())
        group = groups.setdefault(row.get('source_group', row['video']), [0, 0])
        group[0] += value
        group[1] += 1
    if not groups:
        return None
    values = np.array(list(groups.values()))
    ci = None
    if len(values) > 1:
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(len(values), size=(samples, len(values)))].sum(axis=1)
        ci = np.quantile(100 * draws[:, 0] / draws[:, 1], [.025, .975]).tolist()
    return {'n': len(rows), 'source_videos': len(values),
            'delta_pp': float(100 * values[:, 0].sum() / values[:, 1].sum()), 'ci95_pp': ci,
            'bootstrap_samples': samples, 'seed': seed, 'unit': 'source-video cluster',
            'contrast': '(temporal ordered - temporal shuffled) - (plain ordered - plain shuffled)'}


def audit_variants(rows, maps):
    audit_observations(rows, maps)
    for row in rows:
        values = [m[row['id']]['visual_tokens'] for m in maps.values()]
        if any(type(v) is not int or v <= 0 for v in values) or len(set(values)) != 1:
            raise ValueError('unmatched visual token budgets')
        for name in VARIANTS:
            pred = maps[name][row['id']]
            obs = pred['observation']
            count = len(obs['frame_times'])
            shuffled = name.endswith('_shuffle')
            expected = {'temporal_off': name == 'temporal_off',
                        'block_shuffle_seed': 17 if shuffled else None,
                        'block_order': block_order(count, obs['video_sha256'], 17) if shuffled else list(range(count))}
            if pred.get('intervention') != expected:
                raise ValueError(f'incorrect intervention: {name} {row["id"]}')


def check_replay(rows, reference, replay):
    audit_observations(rows, {'reference': reference, 'replay': replay})
    for row in rows:
        for key in ('choice', 'token_ids', 'visual_tokens'):
            if reference[row['id']][key] != replay[row['id']][key]:
                raise ValueError(f'ordered replay differs: {row["id"]} {key}')


def hld_analysis(rows, maps):
    selected = [r for r in rows if r['task'] == 'HLD']
    return {'scope': 'Exploratory all-HLD error inventory. Option letters are not semantic hallucination labels.',
            'n': len(selected), 'videos': len({r['video'] for r in selected}),
            'gold_option_counts': dict(Counter(str(r['answer']) for r in selected)),
            'arms': {name: {**stats(selected, m), 'predicted_option_counts': dict(Counter(
                str(m[r['id']]['choice']) for r in selected))} for name, m in maps.items()},
            'cases': [{**r, 'predictions': {name: {'choice': m[r['id']]['choice'],
                       'generated_answer': m[r['id']]['answer'],
                       'correct': m[r['id']]['choice'] == r['answer']} for name, m in maps.items()}}
                      for r in selected]}


def make_report(populations, maps):
    result = {}
    for ds, rows in populations.items():
        audit_variants(rows, maps[ds])
        contrasts = [('temporal', 'temporal_off'), ('temporal_off', 'plain'),
                     ('plain', 'plain_shuffle'), ('temporal', 'temporal_shuffle')]
        subsets = {'all': rows, **{k: [r for r in rows if r['task'] in tasks]
                                  for k, tasks in ENDPOINTS[ds].items()}}
        result[ds] = {'scores': {k: stats(rows, m) for k, m in maps[ds].items()},
                      'paired': {f'{a}_minus_{b}': comparison(rows, maps[ds][a], maps[ds][b], ds)
                                 for a, b in contrasts},
                      'order_interaction': {k: interaction(v, maps[ds]) for k, v in subsets.items()},
                      'observations_and_visual_tokens_matched': True}
    return {'completed': True, 'training_launched': False, 'scope': SCOPE, 'results': result}


def load_base(base, manifest_root, cfg):
    """Read only the fixed populations; audit coverage and protocol before GPU work."""
    report = json.loads((base / 'report.json').read_text())
    training = json.loads((base / 'training-audit.json').read_text())
    if not report.get('completed') or not training.get('passed'):
        raise ValueError('baseline pilot is incomplete')
    populations, maps, hashes = {}, {}, {}
    for ds in ENDPOINTS:
        manifest = manifest_root / f'{ds}.jsonl'
        rows = read_split(manifest, 'val')
        populations[ds], maps[ds] = rows, {}
        hashes[f'{ds}/manifest'] = sha256(manifest)
        for arm in ('original_grid', 'plain', 'temporal'):
            folder = base / 'eval' / ds / arm
            run = json.loads((folder / 'run.json').read_text())
            if run['manifest_sha256'] != sha256(manifest):
                raise ValueError('baseline population changed')
            for key in ('model', 'steps', 'pixels', 'pool_grid', 'timestamp_policy', 'max_new_tokens'):
                if run['config'][key] != cfg[key]:
                    raise ValueError(f'baseline protocol changed: {key}')
            if (run['config']['mode'] != 'grid' or run['config']['limit'] != 0
                    or run['config']['split'] != 'val' or not run['config'].get('mc_choice_score')):
                raise ValueError('baseline observation/scoring protocol differs')
            maps[ds][arm] = collect(rows, folder)
            for file in ('run.json', 'predictions.jsonl'):
                hashes[f'{ds}/{arm}/{file}'] = sha256(folder / file)
        audit_observations(rows, maps[ds])
    return populations, maps, hashes


def run(base, out, devices, *, report_only=False):
    base, out = Path(base).resolve(), Path(out).resolve()
    if out == base or base in out.parents or out in base.parents:
        raise ValueError('diagnostics require a separate output directory')
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError('devices must be distinct nonnegative GPU indices')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.diagnostics.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = json.loads((base / 'experiment-config.json').read_text())
        manifest_root = Path(cfg['external_data'])
        rows, maps, hashes = load_base(base, manifest_root, cfg)
        checkpoints = {}
        for arm in ('plain', 'temporal'):
            folder = base / 'train' / arm
            training = json.loads((folder / 'run.json').read_text())
            checkpoint = folder / f'checkpoint-{cfg["updates"]:06d}'
            record = checked_checkpoint(checkpoint, training['id'])
            if record['update'] != cfg['updates']:
                raise ValueError('not the final checkpoint')
            for ds in ENDPOINTS:
                old = json.loads((base / 'eval' / ds / arm / 'run.json').read_text())
                if old['adapter'] != record:
                    raise ValueError('baseline checkpoint changed')
            checkpoints[arm] = checkpoint
        code = {str(p.relative_to(Path(__file__).parent.parent)): sha256(p)
                for p in Path(__file__).parent.glob('*.py')}
        for name in ('backbone.py', 'positions.py', 'run_io.py'):
            code['v3/' + name] = sha256(Path(__file__).parent.parent / 'v3' / name)
        identity = {'base': str(base), 'config': cfg, 'baseline_hashes': hashes, 'code': code,
                    'devices': devices, 'scope': SCOPE,
                    'checkpoint_hashes': {k: sha256(v / 'complete.json') for k, v in checkpoints.items()}}
        identity['id'] = digest(identity)
        ensure_run_manifest(out / 'inputs.json', identity)
        write_json(out / 'hld-before.json', hld_analysis(rows['ovo'], maps['ovo']))
        # Small, self-contained report bundle; never copy videos or checkpoints.
        for ds in ENDPOINTS:
            atomic_text(out / 'manifests' / f'{ds}.jsonl', (manifest_root / f'{ds}.jsonl').read_text())
            for arm in ('original_grid', 'plain', 'temporal'):
                for file in ('run.json', 'predictions.jsonl'):
                    atomic_text(out / 'reused' / ds / arm / file, (base / 'eval' / ds / arm / file).read_text())
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'offline',
               'TRACE_WANDB': '0', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}

        def evaluate(ds, manifest, name, arm, flags, *, flow=False):
            kind = 'flow' if flow else 'eval'
            dest = out / kind / ds / name
            command = [sys.executable, '-m', 'trace_vl.core_vlm.evaluate', *shared_args(cfg, manifest),
                       '--out', str(dest), '--mode', 'grid', '--shards', str(len(devices)),
                       '--max-new-tokens', str(cfg['max_new_tokens']), '--mc-choice-score',
                       '--checkpoint', str(checkpoints[arm]), *flags]
            tag = f'{kind}-{ds}-{name}'
            run_jobs([(f'{tag}-{i}', [*command, '--shard', str(i)], [gpu])
                      for i, gpu in enumerate(devices)], env, out / 'logs')
            run_jobs([(tag + '-report', [*command, '--report-only'], [])], env, out / 'logs')
            return collect(read_split(manifest, 'val'), dest)

        if not report_only:
            print('[1/3] Frozen inputs verified; GPU availability and ordered replay', flush=True)
            gpu_check = ('import torch; '
                         f'assert torch.cuda.device_count()=={len(devices)}; '
                         'free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(torch.cuda.device_count())]; '
                         f'print(free); assert min(free)>={cfg.get("minimum_free_gib", 30)}')
            run_jobs([('gpu-check', [sys.executable, '-c', gpu_check], devices)], env, out / 'logs')
            for ds, population in rows.items():
                tiny = select_records(population, 4)
                manifest = out / f'flow-{ds}.jsonl'
                atomic_text(manifest, ''.join(json.dumps(r) + '\n' for r in tiny))
                small = {k: {r['id']: m[r['id']] for r in tiny} for k, m in maps[ds].items()}
                for arm in ('plain', 'temporal'):
                    replay = evaluate(ds, manifest, arm, arm, [], flow=True)
                    check_replay(tiny, small[arm], replay)
                for name, (arm, flags) in VARIANTS.items():
                    small[name] = evaluate(ds, manifest, name, arm, flags, flow=True)
                audit_variants(tiny, small)
            write_json(out / 'flow-ok.json', {'passed': True, 'inputs_id': identity['id']})
            print('[2/3] Six evaluation arms; no training', flush=True)
            for ds in ENDPOINTS:
                for name, (arm, flags) in VARIANTS.items():
                    evaluate(ds, manifest_root / f'{ds}.jsonl', name, arm, flags)
        flow = json.loads((out / 'flow-ok.json').read_text())
        if flow != {'passed': True, 'inputs_id': identity['id']}:
            raise ValueError('missing or incompatible GPU flow verification')
        _, _, final_hashes = load_base(base, manifest_root, cfg)
        if final_hashes != hashes:
            raise ValueError('baseline inputs changed during diagnostics')
        print('[3/3] Paired ablations, direct interaction and HLD inventory', flush=True)
        for ds in ENDPOINTS:
            for name in VARIANTS:
                maps[ds][name] = collect(rows[ds], out / 'eval' / ds / name)
        report = make_report(rows, maps)
        report['inputs_id'] = identity['id']
        report['prior_gate_unchanged'] = json.loads((base / 'report.json').read_text()).get('decision')
        write_json(out / 'report.json', report)
        write_json(out / 'hld-analysis.json', hld_analysis(rows['ovo'], maps['ovo']))
        lines = ['MIXED CORE DIAGNOSTICS (evaluation only)', SCOPE]
        for ds, values in report['results'].items():
            lines.append(ds + ' primary endpoint:')
            for contrast, endpoints in values['paired'].items():
                lines.append(contrast + ': ' + json.dumps(endpoints['primary']['choice']))
            lines.append('order_interaction: ' + json.dumps(values['order_interaction']['primary']))
        atomic_text(out / 'digest.txt', '\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)
        print('VERIFIED MIXED DIAGNOSTICS:', out / 'report.json', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--devices', default='0,1,2,3')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    run(args.base, args.out, [int(x) for x in args.devices.split(',')], report_only=args.report_only)


if __name__ == '__main__':
    main()
