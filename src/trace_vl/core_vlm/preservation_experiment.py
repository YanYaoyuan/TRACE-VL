"""Core-model recipe repair: CE vs preserved LoRA vs preserved temporal LoRA."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
from pathlib import Path

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .answerability import probe_rows
from .control import select_records
from .data import read_split, sha256
from .experiment import paired, run_jobs, shared_args, training_command
from .mixed_experiment import audit_training as base_audit
from .mixed_experiment import decision
from .prepare import digest, write_json
from .preservation_data import reuse_data as prepare
from .train import checked_checkpoint
from .transfer import choice_map, collect, external_report, stats

ARMS = {'ce': 'none', 'plain': 'none', 'temporal': 'causal'}


def audit_training(root, target):
    result = base_audit(root, target, arms=ARMS, config_differences=('preservation_weight',))
    weights = {}
    for name in ARMS:
        cfg = json.loads((Path(root) / name / 'run.json').read_text())['config']
        weights[name] = cfg['preservation_weight']
        if not cfg['preservation_teacher'] or cfg['preservation_temperature'] != 2.:
            raise ValueError('all arms must execute the same original teacher')
        logs = [json.loads(x) for x in (Path(root) / name / 'metrics.jsonl').read_text().splitlines()]
        if not sum(x['teacher_questions'] for x in logs):
            raise ValueError('no general-family teacher anchors seen')
        if any(not all(math.isfinite(x[k]) for k in ('answer_ce', 'preservation_kl_per_question')) for x in logs):
            raise ValueError('nonfinite preservation logs')
    if weights != {'ce': 0., 'plain': .5, 'temporal': .5}:
        raise ValueError('predeclared CE/anchor weights changed')
    result['preservation_weights'] = weights
    return result


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


def answerability_report(manifests, out, names):
    out = Path(out)
    result = {'scope': 'HLD has benchmark labels; added abstention-option probes are '
              'source-answerable diagnostics, not human-verified sampled-evidence false-refusal rates. '
              'No camera false-alarm or open-ended calibration claim.', 'datasets': {}}
    for dataset, manifest in manifests.items():
        rows = read_split(manifest, 'val')
        maps = {n: collect(rows, out / 'eval' / dataset / n) for n in names}
        groups = {'all': rows}
        if dataset == 'ovo':
            groups.update({'HLD': [r for r in rows if r['task'] == 'HLD'],
                           'non_HLD': [r for r in rows if r['task'] != 'HLD']})
        entry = {'subgroups': {g: {n: stats(rs, m) for n, m in maps.items()} for g, rs in groups.items()}}
        probes = read_split(out / f'probe-{dataset}.jsonl', 'val')
        pmaps = {n: collect(probes, out / 'eval' / (dataset + '-abstention') / n) for n in names}
        from .transfer import audit_observations
        audit_observations(probes, pmaps)
        entry['probe'] = {n: {**stats(probes, p), 'abstention_choice_rate': sum(
            p[r['id']]['choice'] == r['abstention_index'] for r in probes) / len(probes)} for n, p in pmaps.items()}
        contrasts = [('plain', 'ce'), ('temporal', 'plain'), ('plain', 'original_grid'), ('temporal', 'original_grid')]
        entry['paired_subgroups'] = {g: {f'{a}_minus_{b}': paired(rs, choice_map(maps[a]), choice_map(maps[b]))
                                        for a, b in contrasts} for g, rs in groups.items()}
        # Encode selection as a binary outcome for the same video-cluster bootstrap.
        binary_rows = [{**r, 'answer': 0} for r in probes]
        binary = {n: {r['id']: {'answer': 'A' if p[r['id']]['choice'] == r['abstention_index'] else 'B'}
                      for r in probes} for n, p in pmaps.items()}
        entry['paired_abstention_rate_pp'] = {f'{a}_minus_{b}': paired(binary_rows, binary[a], binary[b])
                                              for a, b in contrasts}
        result['datasets'][dataset] = entry
    return result


def capability_gate(diagnostics):
    """Do not hide HLD loss behind a pooled accuracy gain; same -2 pp margin."""
    groups = diagnostics['datasets']['ovo']['paired_subgroups']
    checks = {}
    for name in ('plain', 'temporal'):
        for group in ('HLD', 'non_HLD'):
            value = groups[group][name + '_minus_original_grid']
            checks[name + '_' + group] = bool(value and value['ci95_pp'] and value['ci95_pp'][0] > -2)
    return {'passed': all(checks.values()), 'checks': checks,
            'rule': 'Both preserved arms retain HLD and non-HLD vs original-grid: CI lower > -2 pp. '
                    'Failure can be imprecision; refusal probes require separate review.'}


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
        identity = {'config': cfg, 'code': code, 'protocol': 'core-preservation-v1'}
        identity['id'] = digest(identity)
        ensure_run_manifest(out / 'run.json', identity)
        print('[1/5] Fixed training-only mixture; reuse frozen external development data', flush=True)
        manifest, external = prepare(cfg, out / 'data')
        probes = {}
        for dataset, path in external.items():
            rows = probe_rows(read_split(path, 'val'))
            if not rows:
                raise ValueError('no answerable-source refusal probe rows')
            dest = out / f'probe-{dataset}.jsonl'
            text = ''.join(json.dumps(r) + '\n' for r in rows)
            if dest.exists() and dest.read_text() != text:
                raise ValueError('refusal probe population changed')
            atomic_text(dest, text)
            probes[dataset] = dest
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
            cmd += ['--preservation-teacher', '--preservation-weight', str(0. if name == 'ce' else .5),
                    '--preservation-temperature', '2', '--temporal', ARMS[name]]  # deliberately NO --init-checkpoint
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

        print('[2/5] Two-step DDP QA training AND checkpoint evaluation for all three arms', flush=True)
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
        print('[4/5] Original grid + CE + preserved plain/temporal LoRA on identical external observations', flush=True)
        for dataset, path in external.items():
            for name, checkpoint in {'original_grid': None, **checkpoints}.items():
                evaluate(dataset, path, name, checkpoint)
        # Freeze these prompt-modified populations before training; keep official
        # evaluation manifests unchanged. They cannot decide the primary gate.
        for dataset, path in probes.items():
            for name, checkpoint in {'original_grid': None, **checkpoints}.items():
                evaluate(dataset + '-abstention', path, name, checkpoint)
        print('[5/5] Paired choice/format scores and predeclared feasibility report', flush=True)
        results = matched_report(external, out / 'eval', ('original_grid', *ARMS),
            [('plain', 'ce'), ('temporal', 'plain'), ('temporal', 'original_grid'), ('plain', 'original_grid'), ('ce', 'original_grid')])
        report = {'completed': True, 'results': results, 'decision': decision(results), 'training_audit': audit,
                  'supervision': json.loads((out / 'data/supervision.json').read_text()),
                  'scope': 'official NExT-QA train used; NExT validation is in-dataset generalization, '
                           'OVO is cross-dataset development. One seed; fixed final checkpoint; '
                           'ordinary temporal adapter baseline, not an established novelty or official streaming score.'}
        report['answerability_diagnostics'] = answerability_report(external, out, ('original_grid', *ARMS))
        report['capability_gate'] = capability_gate(report['answerability_diagnostics'])
        report['decision']['architecture_gate_passed'] = report['decision']['passed']
        report['decision']['passed'] &= report['capability_gate']['passed']
        report['decision']['answerability_review_required'] = True
        write_json(out / 'report.json', report)
        lines = ['# Core video-model preservation pilot', '', json.dumps(report['decision']), '',
                 '| Dataset | Arm | Choice accuracy | Strict accuracy | Format valid |', '|---|---|---:|---:|---:|']
        for dataset, value in results.items():
            for name, score in value['scores'].items():
                lines.append(f"| {dataset} | {name} | {100*score['choice_accuracy']:.2f}% | "
                             f"{100*score['strict_accuracy']:.2f}% | {100*score['format_valid_fraction']:.2f}% |")
        lines.extend(['', report['scope'], '', 'No further training is launched automatically.'])
        atomic_text(out / 'REPORT.md', '\n'.join(lines) + '\n')
        print('VERIFIED CORE PRESERVATION PILOT:', out / 'report.json', flush=True)


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
