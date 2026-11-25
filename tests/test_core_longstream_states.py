import copy
import json
from types import SimpleNamespace

import pytest

from trace_vl.core_vlm import longstream_delayed_states as module
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.prepare import digest, write_json


def cohort():
    rows = [{'id': f'q{i}', 'source_group': f'used{i}', 'query_time': 60.0+10*i,
             'task': 'Action Perception' if i else 'Object Perception',
             'temporal_clue_type': 'Concurrent', 'frames_required': 'single'} for i in range(3)]
    return rows, {'original_targets': 3, 'original_sources': 3, 'question_ids': ['q0', 'q1', 'q2'],
                  'source_groups': ['used0', 'used1', 'used2']}


def test_reject_missing_duplicate_nonstate_or_new_sources():
    rows, spec = cohort()
    allowed = set(spec['source_groups'])
    module.validate_rows(rows, allowed, spec)
    for bad, boundary in ((rows[:-1], allowed), (rows, allowed-{'used0'}), ([rows[0], rows[0], rows[2]], allowed)):
        with pytest.raises(ValueError, match='population'):
            module.validate_rows(bad, boundary, spec)
    for changes in ({'query_time': 59.9}, {'query_time': float('nan')}, {'task': 'Counting'},
                    {'temporal_clue_type': 'Prior'}, {'evidence': [1, 2]}, {'frames_required': 'unknown'}):
        bad = copy.deepcopy(rows)
        bad[0].update(changes)
        with pytest.raises(ValueError, match='task/metadata'):
            module.validate_rows(bad, allowed, spec)


def test_smoke_long_short_and_ties():
    sources = [{'source_group': g, 'planned_stream_encoder_calls': n} for g, n in [('a', 5), ('b', 5), ('c', 1000)]]
    assert module.choose_smoke(sources) == module.choose_smoke(sources[::-1]) == ['a', 'c']
    with pytest.raises(ValueError, match='two distinct'):
        module.choose_smoke(sources[:1])


def test_real_prepare_all_delays_stable_resume_and_media_corruption(tmp_path, monkeypatch):
    data, ref = tmp_path/'data', tmp_path/'ref'
    data.mkdir()
    ref.mkdir()
    checkpoint = tmp_path/'train'/'checkpoint-002500'
    checkpoint.mkdir(parents=True)
    write_json(checkpoint.parent/'run.json', {'videos': {'training': 'different-hash'}})
    base = {'model': '/data/fake', 'model_identity': {'frozen': True}, 'plain': str(checkpoint),
            'checkpoint': {'run': 'fixed'}, 'pixels': 65536, 'capacity': 32, 'seed': 17,
            'primary_score': 'fp32_letter_head_over_bf16_hidden', 'secondary_score': 'bf16_letter_head'}
    base['id'] = digest(base)
    write_json(ref/'inputs.json', base)
    monkeypatch.setattr(module.core, 'REFERENCE_ID', base['id'])
    monkeypatch.setattr(module, 'checked_checkpoint', lambda *a: base['checkpoint'])
    monkeypatch.setattr(module, 'checkpoint_identity', lambda *a: base['model_identity'])
    pts = tmp_path/'pts.json'
    write_json(pts, {'frames': [{'best_effort_timestamp_time': str(i/2)} for i in range(1001)]})
    rows, spec = cohort()
    sources, backed = [], {}
    for row in rows:
        video = '/data/fake-'+row['source_group']+'.mp4'
        backing = tmp_path/(row['source_group']+'.bytes')
        backing.write_bytes(row['source_group'].encode())
        backed[video] = backing
        row.update(video=video, question='What is held now?', options=['a', 'b'], answer=1)
        sources.append({'source_group': row['source_group'], 'video': video, 'video_sha256': sha256(backing),
                        'metadata': str(pts), 'metadata_sha256': sha256(pts), 'last_valid_source_time': 500.0})
    questions = data/'questions.jsonl'
    questions.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    old = tmp_path/'old.jsonl'
    old.write_text(''.join(json.dumps({'source_group': g})+'\n' for g in spec['source_groups']))
    monkeypatch.setattr(module, 'OLD_DEVELOPMENT_MANIFEST', old)
    monkeypatch.setattr(module, 'sha256', lambda p: sha256(backed.get(str(p), p)))
    rule = {'tasks': sorted(module.TASKS), 'selection_uses_model_outcomes': False, 'delays_seconds': [0, 15, 60, 180, 300]}
    rule['id'] = digest(rule)
    write_json(data/'selection-rule.json', rule)
    spec.update(selection_rule_id=rule['id'], old_development_manifest_sha256=sha256(old),
                completed_inventory_sources=250, model_outcomes_used=False)
    spec['id'] = digest(spec)
    write_json(data/'candidate-spec.json', spec)
    ready = {'completed': True, 'sources': sources, 'candidate_spec_id': spec['id'],
             'selection_rule_id': rule['id'], 'questions_sha256': sha256(questions)}
    ready['id'] = digest(ready)
    write_json(data/'ready.json', ready)
    args = SimpleNamespace(out=tmp_path/'sealed', reference=ref, data=data, data_id=ready['id'])
    module.prepare(args)
    sealed = json.loads((args.out/'inputs.json').read_text())
    assert sealed['id'] == digest({k: v for k, v in sealed.items() if k != 'id'})
    assert sealed['questions'] == 15 and len(sealed['sources']) == 3
    assert sealed['cohorts'] == {str(d): {'questions': 3, 'sources': 3} for d in (0, 15, 60, 180, 300)}
    assert all(len(s['queries']) == 5 and {q['answer'] for q in s['queries']} == {1} for s in sealed['sources'])
    assert all(q['query_time'] == q['target_time']+q['delay_seconds'] for s in sealed['sources'] for q in s['queries'])
    module.prepare(args)
    assert json.loads((args.out/'inputs.json').read_text()) == sealed
    backed[sources[0]['video']].write_bytes(b'changed-media')
    with pytest.raises(ValueError, match='source bytes'):
        module.prepare(args)
