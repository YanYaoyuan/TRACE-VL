import copy
import json
from collections import Counter

import numpy as np
import pytest

from trace_vl.core_vlm import answerability as a
from trace_vl.core_vlm import preservation_data as d
from trace_vl.core_vlm.data import read_split, sha256
from trace_vl.core_vlm.prepare import digest, write_json


def row(i, video='v.mp4', **kw):
    return dict(id=str(i), video=str(video), split='train', question='what changed?',
                options=['red', 'blue'], answer=1, **kw)


def test_probe_is_frozen_balanced_and_not_hld_training():
    rows = [row(i, task='ASI') for i in range(12)] + [row('h', task='HLD')]
    rows.append({**row('u'), 'options': ['yes', 'Unable to answer']})
    before = copy.deepcopy(rows)
    probes = a.probe_rows(rows)
    assert probes == a.probe_rows(list(reversed(rows)))
    assert len(probes) == 12 and rows == before
    assert Counter(r['abstention_index'] for r in probes) == {0: 4, 1: 4, 2: 4}
    assert all(r['options'][r['answer']] == 'blue' and r['split'] == 'val' for r in probes)
    assert all(r['source_question_id'] not in {'h', 'u'} for r in probes)


def test_review_rejects_heldout_unreviewed_mismatched_frames_and_wrong_evidence(tmp_path, monkeypatch):
    video = tmp_path / 'v.mp4'
    video.write_bytes(b'video')
    training = [row('train', video)]
    pair = dict(source_id='train', reviewer='reviewer-1', reason='blue car appears only at 2 s',
                question_and_sampled_frames_reviewed=True, video_sha256=sha256(video),
                evidence_intervals=[[2., 3.]], views=[
                    dict(answerable=True, observation={'start': 0, 'end': 3}, query_time=3., decoded_rgb_sha256='with'),
                    dict(answerable=False, observation={'start': 0, 'end': 1}, query_time=1., decoded_rgb_sha256='without')])
    def decode(r, **kw):
        visible = r['query_time'] == 3
        return None, np.array([[0., 2. if visible else .5]]), r, {'decoded_rgb_sha256': 'with' if visible else 'without'}
    monkeypatch.setattr(a, 'decode_observation', decode)
    path = tmp_path / 'pairs.jsonl'
    def run(p):
        path.write_text(json.dumps(p) + '\n')
        return a.reviewed_training_pairs(path, training, steps=1, pixels=1024)
    result = run(pair)
    assert len(result) == 2
    assert result[0]['options'][result[0]['answer']] == 'blue'
    assert result[1]['options'][result[1]['answer']] == a.UNKNOWN
    for key, value in [('source_id', 'heldout'), ('question_and_sampled_frames_reviewed', False),
                       ('reviewer', ''), ('video_sha256', 'changed'), ('evidence_intervals', [[0, 3]])]:
        with pytest.raises(ValueError):
            run({**pair, key: value})
    bad = copy.deepcopy(pair)
    bad['views'][0]['decoded_rgb_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='frames changed'):
        run(bad)


def test_frozen_mixture_reuse_and_tamper_rejection(tmp_path, monkeypatch):
    root, out = tmp_path / 'mixture', tmp_path / 'out'
    root.mkdir()
    videos = [root / 'g.mp4', root / 't.mp4']
    for i, p in enumerate(videos):
        p.write_bytes(str(i).encode())
    rows = [row(i, p, mixture_family=f) for i, (p, f) in enumerate(zip(videos, ['general', 'temporal']))]
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_json(root / 'inventory.json', {'source_overlap': 0, 'exact_byte_overlap': 0})
    for name in ('media.json', 'annotation-audit.json'):
        write_json(root / name, {})
    external = {'manifests': {'ovo': 'one', 'nextqa': 'two'}}
    cfg = dict(mixture_data=str(root), external_data='external', steps=8, pixels=65536, pool_grid=[4,5], seed=17, workers=2)
    inputs = {'config': cfg, 'external': external}
    inputs['id'] = digest(inputs)
    write_json(root / 'inputs.json', inputs)
    write_json(root / 'complete.json', {'inputs': inputs['id'], 'videos': {str(p): sha256(p) for p in videos},
               'files': {n: sha256(root / n) for n in ('manifest.jsonl', 'inventory.json', 'media.json', 'annotation-audit.json')}})
    monkeypatch.setattr(d, 'frozen_external', lambda _: ({'ovo': 'frozen', 'nextqa': 'frozen2'}, external))
    path, _ = d.reuse_data(cfg, out)
    first = path.read_bytes()
    d.reuse_data(cfg, out)
    assert path.read_bytes() == first
    assert len(read_split(path, 'train')) == 2
    assert json.loads((out / 'supervision.json').read_text())['answerability_supervision_active'] is False
    videos[0].write_bytes(b'tampered')
    with pytest.raises(ValueError, match='media changed'):
        d.reuse_data(cfg, out)


def test_refusal_report_keeps_hld_and_answerable_proxy_separate(tmp_path, monkeypatch):
    from trace_vl.core_vlm import preservation_experiment as runner
    from trace_vl.core_vlm import transfer
    rows = [row('h', task='HLD', split_unused=None), row('a', task='ASI')]
    rows = [{**r, 'split': 'val'} for r in rows]
    manifest = tmp_path / 'ovo.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    probes = a.probe_rows(rows)
    (tmp_path / 'probe-ovo.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in probes))
    def collect(rs, directory):
        return {r['id']: {'choice': r.get('abstention_index', r['answer']) if directory.name == 'plain' else r['answer'],
                         'answer': chr(65 + r['answer'])} for r in rs}
    monkeypatch.setattr(runner, 'collect', collect)
    monkeypatch.setattr(transfer, 'audit_observations', lambda *a: None)
    report = runner.answerability_report({'ovo': manifest}, tmp_path, ('original_grid', 'ce', 'plain', 'temporal'))
    data = report['datasets']['ovo']
    assert data['subgroups']['HLD']['plain']['n'] == 1
    assert data['subgroups']['non_HLD']['plain']['n'] == 1
    assert data['probe']['plain']['abstention_choice_rate'] == 1
    assert data['probe']['plain']['choice_accuracy'] == 0
    assert data['paired_abstention_rate_pp']['plain_minus_ce']['delta_pp'] == 100


def test_capability_gate_rejects_hidden_hld_loss():
    from trace_vl.core_vlm.preservation_experiment import capability_gate
    groups = {g: {a + '_minus_original_grid': {'ci95_pp': [0, 3]} for a in ('plain', 'temporal')}
              for g in ('HLD', 'non_HLD')}
    data = {'datasets': {'ovo': {'paired_subgroups': groups}}}
    assert capability_gate(data)['passed']
    groups['HLD']['temporal_minus_original_grid']['ci95_pp'] = [-10, 0]
    assert not capability_gate(data)['passed']
