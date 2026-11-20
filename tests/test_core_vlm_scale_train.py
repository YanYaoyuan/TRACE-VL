"""Release an audited batch configuration without rebuilding frozen scale data."""
import copy
import json
import sys
from pathlib import Path

import pytest

from trace_vl.core_vlm import scale_train
from trace_vl.core_vlm.batch_benchmark import recommend
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.prepare import digest, write_json


@pytest.fixture
def prepared_trial(tmp_path, monkeypatch):
    prepared, bench, pilot = (tmp_path / n for n in ('prepared', 'bench', 'pilot'))
    (prepared / 'mixture').mkdir(parents=True)
    (bench / 'logs').mkdir(parents=True)
    pilot.mkdir()
    cfg = dict(pilot_run=str(pilot), model='base', steps=8, pixels=65536,
               pool_grid=[4, 5], timestamp_policy='native-index-fps', gpus=[0, 1, 2, 3],
               accumulation=4, updates=2500, questions_per_family=2)
    write_json(prepared / 'config.json', cfg)
    prep = {'config': cfg, 'code': {'old': 'retained'}}
    write_json(prepared / 'preparation.json', {**prep, 'id': digest(prep)})
    write_json(prepared / 'mixture/inventory.json', {'families': {'general': 2, 'temporal': 2}})
    write_json(prepared / 'mixture/complete.json', {'previous': 'data completion'})
    (prepared / 'mixture/manifest.jsonl').write_text('frozen manifest\n')
    write_json(pilot / 'experiment-config.json', {**cfg, 'updates': 1000})
    audit = {'passed': True, 'updates': 1000}
    write_json(pilot / 'report.json', {'completed': True, 'training_audit': audit,
                                      'decision': {'passed': False}})
    monkeypatch.setattr(scale_train, 'audit_training', lambda root, target: audit)
    parent = Path(scale_train.__file__).parent
    identity = dict(config=cfg, steps=3, warmup=1, recipe='preserved',
                    prepared_complete_sha256=sha256(prepared / 'mixture/complete.json'),
                    manifest_sha256=sha256(prepared / 'mixture/manifest.jsonl'),
                    code={p.name: sha256(p) for p in parent.glob('*.py')},
                    backbone_sha256=sha256(parent.parent / 'v3/backbone.py'))
    write_json(bench / 'run.json', {**identity, 'id': digest(identity)})
    trials = {}
    for arm in ('plain', 'temporal'):
        for size in (1, 2, 4):
            name = f'{arm}-mb{size}'
            rows = [dict(update=i, seconds_per_update=16. / size, loss=1. / i,
                         gradient_norm=1., question_ids_by_rank=[[f'q-{i}']], lr=2e-5,
                         answer_tokens=20, teacher_questions=1, peak_reserved_gib_max=12.,
                         effective_batch_size=16) for i in range(1, 4)]
            (bench / 'logs' / (name + '.log')).write_text('loader message\n' +
                ''.join(json.dumps(r) + '\n' for r in rows))
            trials[name] = dict(passed=True, median_seconds_per_update=16. / size,
                                peak_reserved_gib_max=12., first_loss=1., initial_lora='same',
                                question_ids_by_update=[r['question_ids_by_rank'] for r in rows])
    write_json(bench / 'trials.json', trials)
    write_json(bench / 'batch-report.json', dict(completed=True, trials=trials, recommendation=recommend(trials)))
    return prepared, bench, tmp_path / 'out'


def test_release_reuses_frozen_inputs_and_preserves_recipe_and_global_batch(prepared_trial):
    prepared, bench, out = prepared_trial
    before = {p: p.read_bytes() for p in prepared.rglob('*') if p.is_file()}
    cfg = scale_train.configure(prepared, bench, out)
    assert cfg['mixture_data'] == str(prepared / 'mixture')
    assert cfg['microbatch_size'] == 4 and cfg['accumulation'] == 1
    assert cfg['preservation_weight'] == .5 and cfg['updates'] == 2500
    assert cfg['reviewed_pairs'] is None
    assert before == {p: p.read_bytes() for p in before}
    assert scale_train.configure(prepared, bench, out) == cfg
    release = json.loads((out / 'release.json').read_text())
    assert release['pilot_decision'] == {'passed': False}
    assert release['id'] == digest({k: v for k, v in release.items() if k != 'id'})


@pytest.mark.parametrize('change', ['schedule', 'time', 'loss', 'missing', 'gradient'])
def test_reject_changed_or_incomplete_raw_benchmark_logs(prepared_trial, change):
    _, bench, _ = prepared_trial
    path = bench / 'logs/temporal-mb4.log'
    rows = [json.loads(s) for s in path.read_text().splitlines()[1:]]
    if change == 'schedule':
        rows[1]['question_ids_by_rank'] = [['different']]
    elif change == 'time':
        rows[-1]['seconds_per_update'] += 10
    elif change == 'loss':
        rows[0]['loss'] += .001
    elif change == 'gradient':
        rows[0]['gradient_norm'] = 0
    else:
        rows.pop()
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError):
        scale_train.audit_benchmark(bench)


def test_reject_changed_prepared_data_before_any_training(prepared_trial):
    prepared, bench, out = prepared_trial
    (prepared / 'mixture/manifest.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='prepared training data differ'):
        scale_train.configure(prepared, bench, out)
    assert not (out / 'release.json').exists()


def test_finished_pilot_report_cannot_skip_actual_checkpoint_audit(prepared_trial, monkeypatch):
    prepared, bench, out = prepared_trial
    def fail(*args):
        raise ValueError('checkpoint missing')
    monkeypatch.setattr(scale_train, 'audit_training', fail)
    with pytest.raises(ValueError, match='checkpoint missing'):
        scale_train.configure(prepared, bench, out)
    assert not (out / 'release.json').exists()


def test_resuming_with_different_supervision_refuses_to_mix_runs(prepared_trial):
    prepared, bench, out = prepared_trial
    scale_train.configure(prepared, bench, out)
    review = out / 'review.jsonl'
    review.write_text('reviewed records')
    with pytest.raises(ValueError, match='different run'):
        scale_train.configure(prepared, bench, out, review)


def test_changed_training_source_invalidates_old_batch_measurement(prepared_trial):
    prepared, bench, out = prepared_trial
    identity = json.loads((bench / 'run.json').read_text())
    identity['code']['train.py'] = 'other trainer'
    identity.pop('id')
    write_json(bench / 'run.json', {**identity, 'id': digest(identity)})
    with pytest.raises(ValueError, match='implementation changed'):
        scale_train.configure(prepared, bench, out)


def test_entry_point_runs_existing_queue_and_forwards_prepare_only(prepared_trial, monkeypatch):
    prepared, bench, out = prepared_trial
    calls = []
    monkeypatch.setattr(scale_train, 'run_training', lambda cfg, path, **kw: calls.append((copy.deepcopy(cfg), path, kw)))
    monkeypatch.setattr(sys, 'argv', ['scale-train', '--prepared', str(prepared), '--benchmark', str(bench),
                                    '--out', str(out), '--prepare-only'])
    scale_train.main()
    assert len(calls) == 1 and calls[0][1:] == (out, {'prepare_only': True})
    assert calls[0][0]['microbatch_size'] == 4
