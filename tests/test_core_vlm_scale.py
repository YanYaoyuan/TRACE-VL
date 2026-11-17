import json
from pathlib import Path

import pytest

from trace_vl.core_vlm.prepare import protected_holdouts
from trace_vl.core_vlm.scale import queue_plan, release, throughput


def test_protect_validation_sources_and_bytes_but_not_training(tmp_path):
    val, train = tmp_path / 'v_abcdefghijk.mp4', tmp_path / 'train.mp4'
    val.write_bytes(b'held-out video')
    train.write_bytes(b'train')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in [
        {'id': 'v', 'question': 'q', 'answer_text': 'a', 'split': 'val', 'video': str(val), 'source_aliases': ['parent']},
        {'id': 't', 'question': 'q', 'answer_text': 'a', 'split': 'train', 'video': str(train), 'source_aliases': ['train']}]))
    blocked, hashes, records = protected_holdouts([manifest])
    from trace_vl.core_vlm.data import sha256
    assert {'parent', 'abcdefghijk', 'v_abcdefghijk'} <= blocked
    assert 'train' not in blocked
    assert hashes == {sha256(val)} and records[str(manifest)] == sha256(manifest)


def test_retention_checks_both_hld_and_answerable_groups(tmp_path, monkeypatch):
    from trace_vl.core_vlm import scale, transfer
    rows = [dict(id=str(i), task='HLD' if i < 2 else 'OCR', video=str(i),
                 question='q', options=['yes', 'no'], answer=0) for i in range(4)]
    monkeypatch.setattr(scale, 'read_split', lambda *a: rows)
    bad = set()
    def collect(rows, path):
        return {r['id']: {'choice': int(path.name == 'temporal' and r['id'] in bad)} for r in rows}
    monkeypatch.setattr(transfer, 'collect', collect)
    assert scale.retention_report({'ovo': 'manifest'}, tmp_path)['passed']
    bad.update(['2', '3'])
    report = scale.retention_report({'ovo': 'manifest'}, tmp_path)
    assert not report['passed']
    assert report['paired']['HLD']['temporal']['delta_pp'] == 0
    assert report['paired']['non_HLD']['temporal']['delta_pp'] == -100


def test_wall_time_estimate_excludes_resume_reset_and_incomplete_line(tmp_path):
    folder = tmp_path / 'train/plain'
    folder.mkdir(parents=True)
    values = [10*(i+1) for i in range(15)] + [1, 11, 21]
    rows = [{'update': i+1, 'wall_seconds_this_invocation': t} for i, t in enumerate(values)]
    (folder / 'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"update":')
    result = throughput(tmp_path, 2500)['plain']
    assert result['median_seconds_per_update'] == 10
    assert result['projected_training_hours'] == pytest.approx(25000/3600)


def test_plan_is_cpu_only_and_counts_exposures(tmp_path):
    cfg = {'gpus': [0, 1, 2, 3], 'accumulation': 4, 'updates': 2500,
           'questions_per_family': 10000, 'pilot_run': str(tmp_path)}
    plan = queue_plan(cfg, {'families': {'general': 10000, 'temporal': 10000}})
    assert not plan['gpu_started']
    assert plan['dataset_pass_equivalents'] == 2
    assert plan['example_presentations_per_arm'] == 40000
    with pytest.raises(ValueError, match='counts'):
        queue_plan(cfg, {'families': {'general': 9000, 'temporal': 10000}})


def test_unfinished_or_changed_pilot_never_releases_gpu(tmp_path):
    from trace_vl.core_vlm.data import sha256
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'completed': False, 'training_audit': {'passed': True}}))
    cfg = {'pilot_run': str(tmp_path)}
    with pytest.raises(ValueError, match='has not completed'):
        release(cfg, tmp_path / 'scale', 'preserved', report, sha256(report), 'reviewed')
    with pytest.raises(ValueError, match='hash differs'):
        release(cfg, tmp_path / 'scale', 'preserved', report, 'old hash', 'reviewed')
    with pytest.raises(ValueError, match='review note'):
        release(cfg, tmp_path / 'scale', 'preserved', report, sha256(report), '')
    assert not (tmp_path / 'scale').exists()


def test_fake_completed_report_cannot_skip_checkpoint_audit(tmp_path):
    from trace_vl.core_vlm.data import sha256
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'completed': True, 'training_audit': {'passed': True}}))
    (tmp_path / 'experiment-config.json').write_text(json.dumps({'updates': 1000}))
    with pytest.raises(FileNotFoundError):
        release({'pilot_run': str(tmp_path)}, tmp_path / 'scale', 'ce', report, sha256(report), 'reviewed')
    assert not (tmp_path / 'scale').exists()


def test_preparation_default_never_calls_training(tmp_path, monkeypatch):
    import sys

    from trace_vl.core_vlm import mixed_experiment, scale
    config = tmp_path / 'config.json'
    config.write_text('{}')
    calls = []
    monkeypatch.setattr(scale, 'prepare', lambda cfg, out: calls.append((cfg, out)))
    monkeypatch.setattr(mixed_experiment, 'run', lambda *a, **kw: pytest.fail('GPU training started'))
    monkeypatch.setattr(sys, 'argv', ['scale', '--config', str(config), '--out', str(tmp_path)])
    scale.main()
    assert calls == [({}, str(tmp_path))]


@pytest.mark.parametrize('recipe,weight', [('ce', 0.), ('preserved', .5)])
def test_reviewed_release_reuses_prepared_mixture_and_refuses_recipe_change(tmp_path, monkeypatch, recipe, weight):
    from trace_vl.core_vlm import preservation_experiment, scale
    from trace_vl.core_vlm.data import sha256
    pilot, out = tmp_path / 'pilot', tmp_path / 'scale'
    pilot.mkdir()
    (out / 'mixture').mkdir(parents=True)
    cfg = {'pilot_run': str(pilot), 'model': 'base', 'steps': 8, 'pixels': 65536,
           'pool_grid': [4, 5], 'timestamp_policy': 'native-index-fps', 'gpus': [0, 1, 2, 3],
           'accumulation': 4, 'updates': 2500, 'questions_per_family': 10000}
    (pilot / 'experiment-config.json').write_text(json.dumps({**cfg, 'updates': 1000}))
    audit = {'passed': True, 'updates': 1000}
    monkeypatch.setattr(preservation_experiment, 'audit_training', lambda root, target: audit)
    report = pilot / 'report.json'
    report.write_text(json.dumps({'completed': True, 'training_audit': audit, 'decision': {'passed': False}}))
    (out / 'preparation.json').write_text(json.dumps({'config': cfg, 'code': {
        p.name: sha256(p) for p in Path(scale.__file__).parent.glob('*.py')}}))
    (out / 'mixture/inventory.json').write_text(json.dumps({'families': {'general': 10000, 'temporal': 10000}}))
    (out / 'mixture/complete.json').write_text('{}')
    training, run = release(cfg, out, recipe, report, sha256(report), 'recipe chosen after comparison')
    assert training['mixture_data'] == str(out / 'mixture')
    assert training['preservation_weight'] == weight
    assert run == out / ('training-' + recipe)
    assert release(cfg, out, recipe, report, sha256(report), 'recipe chosen after comparison') == (training, run)
    with pytest.raises(ValueError, match='different run'):
        release(cfg, out, 'preserved' if recipe == 'ce' else 'ce', report, sha256(report), 'different choice')


def test_temporal_capacity_failure_precedes_any_video_work(tmp_path, monkeypatch):
    import csv

    from trace_vl.core_vlm import mixed_data, scale
    from trace_vl.core_vlm.data import sha256
    path = tmp_path / 'train.csv'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['video', 'qid', 'type', 'question', 'answer',
                                                    'a0', 'a1', 'a2', 'a3', 'a4'])
        writer.writeheader()
        writer.writerow(dict(video='1', qid='1', type='TN', question='when?', answer=0,
                             **{f'a{i}': str(i) for i in range(5)}))
    monkeypatch.setattr(mixed_data, 'NEXTQA_TRAIN_SHA256', sha256(path))
    with pytest.raises(ValueError, match='insufficient raw temporal'):
        scale.temporal_capacity({'nextqa_train': str(path), 'seed': 17, 'questions_per_family': 10000}, tmp_path)
    assert json.loads((tmp_path / 'capacity-preflight.json').read_text())['raw_temporal_questions'] == 1


def test_explicit_missing_media_plan_reports_actual_counts_and_refuses_unrecorded_shrink(tmp_path):
    import copy
    cfg = {'gpus': [0, 1, 2, 3], 'accumulation': 4, 'updates': 2500,
           'questions_per_family': 10000, 'pilot_run': str(tmp_path),
           'repair_from': str(tmp_path / 'old'), 'exclude_training_videos': ['8850274965.mp4']}
    amendment = {'selected_before': 20000, 'retained_after': 19998, 'replacement_questions': 0,
                 'excluded_by_family': {'temporal': 2},
                 'excluded_questions': [dict(id=str(i), video='8850274965.mp4', split='train',
                                             mixture_family='temporal') for i in range(2)]}
    inv = {'families': {'general': 10000, 'temporal': 9998}, 'media_exclusion': amendment}
    plan = queue_plan(cfg, inv)
    assert plan['questions'] == 19998 and plan['target_before_media_exclusion'] == 20000
    assert plan['example_presentations_per_arm'] == 40000
    for key, value in [('replacement_questions', 1), ('retained_after', 20000)]:
        bad = copy.deepcopy(inv)
        bad['media_exclusion'][key] = value
        with pytest.raises(ValueError, match='amendment'):
            queue_plan(cfg, bad)
    with pytest.raises(ValueError, match='audit missing'):
        queue_plan(cfg, {'families': inv['families']})
