"""CPU-only 20k QA preparation and an explicitly released matched GPU queue."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import statistics
from pathlib import Path

from trace_vl.v3.run_io import ensure_run_manifest

from .data import read_split, sha256
from .mixed_data import prepare as prepare_mixture
from .mixed_data import resolve_general_data
from .prepare import digest, write_json
from .prepare import prepare as prepare_general


def retention_report(manifests, directory):
    from .experiment import paired
    from .transfer import choice_map, collect
    rows = read_split(manifests['ovo'], 'val')
    maps = {n: collect(rows, Path(directory) / 'ovo' / n) for n in ('plain', 'temporal', 'original_grid')}
    values = {}
    for group, selected in {'HLD': [r for r in rows if r['task'] == 'HLD'],
                            'non_HLD': [r for r in rows if r['task'] != 'HLD']}.items():
        values[group] = {n: paired(selected, choice_map(maps[n]), choice_map(maps['original_grid']))
                         for n in ('plain', 'temporal')}
    return {'paired': values, 'passed': all(v and v['ci95_pp'] and v['ci95_pp'][0] > -2
                                           for g in values.values() for v in g.values()),
            'rule': 'HLD and non-HLD CI lower > -2 pp vs original-grid, for both arms; '
                    'failure may be imprecision; not a camera false-alarm metric.'}


def throughput(pilot, updates):
    """Provisional wall-time estimate from positive adjacent step times; excludes eval/I/O."""
    result = {}
    for name in ('ce', 'plain', 'temporal'):
        path = Path(pilot) / 'train' / name / 'metrics.jsonl'
        if not path.is_file():
            continue
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # possibly a live log snapshot; never used as completion evidence
        spans = [b['wall_seconds_this_invocation'] - a['wall_seconds_this_invocation']
                 for a, b in zip(rows, rows[1:]) if b['update'] == a['update'] + 1
                 and b['wall_seconds_this_invocation'] > a['wall_seconds_this_invocation']]
        spans = [v for v in spans[-200:] if math.isfinite(v)]
        if len(spans) >= 10:
            result[name] = {'observed_intervals': len(spans), 'median_seconds_per_update': statistics.median(spans),
                            'projected_training_hours': statistics.median(spans) * updates / 3600}
    return result


def queue_plan(cfg, inventory):
    counts = inventory['families']
    expected = {'general': cfg['questions_per_family'], 'temporal': cfg['questions_per_family']}
    amendment = inventory.get('media_exclusion')
    if amendment:
        excluded = amendment['excluded_questions']
        if (not cfg.get('repair_from') or not excluded or amendment['replacement_questions'] != 0
                or len({r['id'] for r in excluded}) != len(excluded)
                or {r['video'] for r in excluded} != set(cfg.get('exclude_training_videos', []))
                or any(r['split'] != 'train' or r['mixture_family'] != 'temporal' for r in excluded)
                or amendment['selected_before'] != 2 * cfg['questions_per_family']
                or amendment['retained_after'] != sum(counts.values())
                or amendment['excluded_by_family'] != {'temporal': len(excluded)}):
            raise ValueError('invalid explicit training-media amendment')
        expected['temporal'] -= len(excluded)
    elif cfg.get('repair_from'):
        raise ValueError('repair exclusion audit missing')
    if counts != expected:
        raise ValueError('prepared family counts differ from the fixed expansion target')
    effective = len(cfg['gpus']) * cfg['accumulation'] * cfg.get('microbatch_size', 1)
    return {'prepared': True, 'gpu_started': False, 'questions': sum(counts.values()), 'families': counts,
            'target_before_media_exclusion': 2 * cfg['questions_per_family'], 'media_exclusion': amendment,
            'arms': ['plain', 'temporal'], 'updates_per_arm': cfg['updates'], 'effective_batch': effective,
            'example_presentations_per_arm': cfg['updates'] * effective,
            'dataset_pass_equivalents': cfg['updates'] * effective / sum(counts.values()),
            'recipes': {'ce': {'preservation_weight': 0}, 'preserved': {'preservation_weight': .5}},
            'throughput_from_running_pilot': throughput(cfg['pilot_run'], cfg['updates']),
            'note': 'Both arms use the same recipe, original base, data, seed, updates and token budget. '
                    'Temporal adds parameters/FLOPs. Estimates exclude data/evaluation and may change with larger data. '
                    'Explicit recipe + completed pilot report hash + review note required to release training. '
                    'No new download, no final-test evaluation, no automatic follow-on training.'}


def temporal_capacity(cfg, out):
    """A cheap upper-bound check before decoding the larger general-QA pool."""
    from .mixed_data import NEXTQA_TRAIN_SHA256, TEMPORAL_TASKS
    from .transfer_data import nextqa_rows
    path = Path(cfg['nextqa_train'])
    if sha256(path) != NEXTQA_TRAIN_SHA256:
        raise ValueError('official NExT-QA train checksum differs')
    with path.open() as handle:
        videos = len({r['video'] for r in csv.DictReader(handle)})
    rows = nextqa_rows(path, videos=videos, seed=cfg['seed'], training_id_collisions=True)
    selected = [r for r in rows if r['task'] in TEMPORAL_TASKS]
    result = {'requested': cfg['questions_per_family'], 'raw_temporal_questions': len(selected),
              'note': 'Upper bound before source exclusions and media checks; not a prepared dataset.'}
    write_json(Path(out) / 'capacity-preflight.json', result)
    if len(selected) < cfg['questions_per_family']:
        raise ValueError('insufficient raw temporal QA for the expansion target; see capacity-preflight.json')
    return result


def prepare(cfg, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'config': cfg, 'code': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
        identity['id'] = digest(identity)
        ensure_run_manifest(out / 'preparation.json', identity)
        temporal_capacity(cfg, out)
        original = resolve_general_data({**cfg, 'general_data': cfg.get('protected_general_data')})
        # Keep the old source split rule. Protect every previously held-out source
        # explicitly as well, including differently named exact-byte copies.
        old_cfg = json.loads((original / 'run.json').read_text())['config']
        general = dict(old_cfg)
        general.update(sources=cfg['general_sources'], qa_kinds=['mc'], max_videos=10000,
                       max_questions_per_video=6, minimum_selected_questions=cfg['questions_per_family'], workers=cfg['workers'], archive_workers=cfg['archive_workers'],
                       protected_manifests=[str(original / 'manifest.jsonl')],
                       media_roots=[*old_cfg.get('media_roots', []),
                                    *[str(original / 'sources' / name / 'media') for name in cfg['general_sources']]])
        write_json(out / 'general-config.json', general)
        print('[1/3] Expand general MC QA from downloaded YouTube subsets; protect prior validation', flush=True)
        repair = None
        if cfg.get('repair_from'):
            from .scale_repair import reuse_failed_preparation
            repair = reuse_failed_preparation(cfg, out)
            print('REUSED completed general QA metadata and existing media from ' + repair['prior_run'], flush=True)
        else:
            prepare_general(out / 'general-config.json', out / 'general')
        mixture = {**cfg, 'general_data': str(out / 'general'), 'data_out': str(out / 'mixture')}
        # These are verified prior training media, not alternate labels/populations.
        prior = Path('/nas-data/vlm/yyy/tracevl-data/core-mixed-temporal-97272fb/nextqa-media/media')
        if prior.is_dir():
            mixture['nextqa_roots'] = [*cfg['nextqa_roots'], str(prior)]
        if repair:
            mixture['media_repair'] = repair
            mixture['nextqa_roots'] = [*mixture['nextqa_roots'],
                str(Path(repair['prior_run']) / 'mixture/nextqa-media/media')]
        write_json(out / 'mixture-config.json', mixture)
        print('[2/3] Freeze 10k general + 10k official train temporal QA; decode and overlap audit', flush=True)
        prepare_mixture(mixture, out / 'mixture')
        inventory = json.loads((out / 'mixture/inventory.json').read_text())
        plan = queue_plan(cfg, inventory)
        write_json(out / 'queue-plan.json', plan)
        print('[3/3] PREPARED SCALE DATA: ' + str(out / 'queue-plan.json'), flush=True)
        print(json.dumps(plan, indent=2), flush=True)
        return plan


def release(cfg, out, recipe, report_path, expected_hash, note):
    """No result-driven implicit launch: operator records a reviewed completed pilot."""
    if recipe not in {'ce', 'preserved'} or not note.strip():
        raise ValueError('explicit ce/preserved recipe and nonempty review note required')
    report_path = Path(report_path).resolve()
    if report_path != (Path(cfg['pilot_run']) / 'report.json').resolve():
        raise ValueError('report must come from the declared preservation pilot')
    if sha256(report_path) != expected_hash:
        raise ValueError('pilot report hash differs from the reviewed report')
    report = json.loads(report_path.read_text())
    if report.get('completed') is not True or report.get('training_audit', {}).get('passed') is not True:
        raise ValueError('pilot has not completed with a valid training audit')
    # Run the same actual final-checkpoint audit, not just a report-file check.
    from .preservation_experiment import audit_training
    pilot_cfg = json.loads((report_path.parent / 'experiment-config.json').read_text())
    actual = audit_training(report_path.parent / 'train', pilot_cfg['updates'])
    if actual != report['training_audit']:
        raise ValueError('pilot checkpoint audit differs from report')
    for key in ('model', 'steps', 'pixels', 'pool_grid', 'timestamp_policy'):
        if cfg[key] != pilot_cfg[key]:
            raise ValueError('pilot observation/backbone changed: ' + key)
    out = Path(out)
    prep = json.loads((out / 'preparation.json').read_text())
    if prep['config'] != cfg or prep['code'] != {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}:
        raise ValueError('prepared configuration/code differs; preserve this directory and use its pinned launcher')
    inventory = json.loads((out / 'mixture/inventory.json').read_text())
    queue_plan(cfg, inventory)
    training = {**cfg, 'mixture_data': str(out / 'mixture'), 'reviewed_pairs': None,
                'preservation_weight': .5 if recipe == 'preserved' else 0.}
    run = out / ('training-' + recipe)
    run.mkdir(exist_ok=True)
    record = {'recipe': recipe, 'pilot_report': str(report_path),
        'pilot_report_sha256': expected_hash, 'pilot_decision': report['decision'], 'review_note': note,
        'mixture_complete_sha256': sha256(out / 'mixture/complete.json'), 'training_config': training}
    record['id'] = digest(record)
    ensure_run_manifest(out / 'release.json', record)
    write_json(run / 'experiment-config.json', training)
    return training, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=('prepare', 'train'), default='prepare')
    p.add_argument('--recipe', choices=('ce', 'preserved'))
    p.add_argument('--pilot-report')
    p.add_argument('--pilot-sha256')
    p.add_argument('--review-note', default='')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.mode == 'prepare':
        prepare(cfg, a.out)
    else:
        if not all((a.recipe, a.pilot_report, a.pilot_sha256, a.review_note.strip())):
            p.error('training requires recipe, pilot report, its reviewed SHA256 and review note')
        training, run = release(cfg, a.out, a.recipe, a.pilot_report, a.pilot_sha256, a.review_note)
        from .mixed_experiment import run as run_training
        run_training(training, run)


if __name__ == '__main__':
    main()
