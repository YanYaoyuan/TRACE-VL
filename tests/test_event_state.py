import copy
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from trace_vl.event_state.common import bind, digest, read_json, write_json
from trace_vl.event_state.data import benchmark_inventory, joy_records, prepare, split_for
from trace_vl.event_state.features import causal_frames, save_tensor, valid_tensor
from trace_vl.event_state.model import ResponseHead
from trace_vl.event_state.report import calibrate, outcome, paired
from trace_vl.event_state.train import timing_loss, train_arm


@pytest.fixture
def cfg():
    c = read_json(Path(__file__).parents[1]/'configs/event_state/pilot.json')
    c.update(dim=16, heads=2, layers=1, state_tokens=2, grid=2, batch=2, epochs=1,
             max_updates=2, reserve_gb=0, max_videos=100, require_benchmark_inventory=False,
             benchmark_roots=[], min_videos={'train': 1, 'calibration': 1, 'dev': 1},
             min_conditions={'train': 1, 'calibration': 1, 'dev': 1})
    return c


def annotation(name='v_a.mp4', start=0, target=2):
    return {'task_type': 'event_grounding', 'video_name': name, 'source': 'test',
            'question': [{'content': 'Tell me when the object moves.', 'time': str(start)}],
            'response': [{'content': 'Now', 'time': str(target)}]}


def test_annotations_do_not_encode_response_text(cfg):
    a, _ = joy_records([annotation()], cfg)
    b = annotation()
    b['response'][0]['content'] = 'SECRET FUTURE ANSWER'
    bb, _ = joy_records([b], cfg)
    assert a == bb
    assert a[0]['ticks'][0] >= .5
    assert a[0]['label_type'] == 'annotated_response_readiness'


@pytest.mark.parametrize('bad', ['../../bad.mp4', '/tmp/bad.mp4', 'noextension'])
def test_unsafe_media_names_rejected(cfg, bad):
    rows, rejected = joy_records([annotation(bad)], cfg)
    assert not rows and rejected['malformed'] == 1


def test_source_split_before_conditions(cfg):
    rows, _ = joy_records([annotation(target=2), annotation(target=4)], cfg)
    assert len(rows) == 2 and rows[0]['split'] == rows[1]['split']


@pytest.mark.parametrize('arm', ['clock', 'frame', 'gru', 'temporal', 'event_state'])
def test_future_frames_cannot_change_prefix(cfg, arm):
    torch.manual_seed(7)
    m = ResponseHead(12, cfg, arm).eval()
    x, q = torch.randn(2, 5, 4, 12), torch.randn(2, 12)
    clock = torch.arange(5).expand(2, -1).float()
    with torch.no_grad():
        first = m(x, q, clock)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:])*100
        second = m(changed, q, clock)
        prefix = m(x[:, :3], q, clock[:, :3])
    torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(first[:, :3], prefix, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('arm', ['frame', 'gru', 'temporal', 'event_state'])
def test_independent_queries_and_real_gradients(cfg, arm):
    torch.manual_seed(9)
    m = ResponseHead(12, cfg, arm)
    x, q, clock = torch.randn(1, 4, 4, 12), torch.randn(1, 12), torch.arange(4)[None].float()
    a = m(x, q, clock)
    m(x*3, q*2, clock)
    b = m(x, q, clock)
    torch.testing.assert_close(a, b)
    loss = timing_loss(a, torch.tensor([[0., 0., 1., 1.]]), torch.ones(1, 4))
    loss.backward()
    assert m.visual[1].weight.grad.abs().sum() > 0
    assert m.query[1].weight.grad.abs().sum() > 0
    if arm == 'event_state':
        assert m.replace.weight.grad.abs().sum() > 0
        assert m.initial.grad.abs().sum() > 0


def test_padding_not_loss():
    x = torch.tensor([[1., 2., 1000.]])
    y = torch.tensor([[0., 1., 0.]])
    torch.testing.assert_close(timing_loss(x, y, torch.tensor([[1., 1., 0.]])),
                               timing_loss(x[:, :2], y[:, :2], torch.ones(1, 2)))


def test_receipts_detect_corruption_and_identity(tmp_path):
    bind(tmp_path/'id.json', {'x': 1})
    with pytest.raises(ValueError):
        bind(tmp_path/'id.json', {'x': 2})
    p = tmp_path/'v.pt'
    save_tensor(p, {'x': torch.ones(1)})
    assert valid_tensor(p)
    p.write_bytes(b'broken')
    assert not valid_tensor(p)


def test_benchmark_inventory_and_missing_annotation(tmp_path):
    write_json(tmp_path/'annotation.json', [{'video': 'path/v_abc.mp4'}, {'video_id': 'def'}])
    ids, _ = benchmark_inventory([str(tmp_path)], True)
    assert {'abc', 'def'} <= ids
    with pytest.raises(ValueError):
        benchmark_inventory([str(tmp_path/'missing')], True)


def make_video(p):
    import cv2
    p.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'MJPG'), 4, (32, 32))
    for i in range(32):
        w.write(np.full((32, 32, 3), i*5, np.uint8))
    w.release()


def test_causal_frame_timestamps(tmp_path):
    p = tmp_path/'v.avi'
    make_video(p)
    seen = list(causal_frames(p, [.6, 1., 1.4]))
    assert [round(s[1], 2) for s in seen] == [.5, 1., 1.25]
    assert all(s[1] <= s[0] for s in seen)


def test_prepare_extract_full_decode_and_splits(tmp_path, cfg):
    names = {}
    for i in range(100):
        name = f'v_{i}.avi'
        names.setdefault(split_for(str(i), cfg['seed']), name)
    root = tmp_path/'raw'
    root.mkdir()
    raw = []
    archive = root/'videos.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for s, name in names.items():
            p = tmp_path/'staging'/name
            make_video(p)
            # Different content hashes avoid exact duplicate leakage across splits.
            with open(p, 'ab') as f:
                f.write(s.encode())
            tar.add(p, arcname='original/'+name)
            raw.append(annotation(name))
    write_json(tmp_path/'ann.json', raw)
    rows = prepare(tmp_path/'ann.json', [str(root)], tmp_path/'data', cfg, 'test')
    assert len(rows) == 3
    report = read_json(tmp_path/'data'/'data-audit.json')
    assert report['ready'] and len(report['verified']) == 3
    assert archive.exists()
    assert all(v['verification'] == 'full_decode_and_content_sha256' for v in report['verified'].values())


def test_calibration_and_coverage():
    row = {'id': 'a', 'source_video_id': 'v', 'ticks': [1, 2, 3, 4], 'response_s': 3,
           'scores': [.1, .2, .8, .9], 'split': 'dev'}
    th = calibrate([row], .05)
    assert outcome(row, th)['on_time']
    assert not outcome(row, th)['early']
    assert paired([row], [row], th, th, samples=10)['delta_pp'] == 0
    other = copy.deepcopy(row)
    other['response_s'] = 4
    with pytest.raises(ValueError):
        paired([row], [other], th, th, samples=10)


def test_training_completion_resume_and_config_mismatch(tmp_path, cfg):
    cache = tmp_path/'features'
    write_json(cache/'identity.json', {'model': 'synthetic-test'})
    rows = []
    for i in range(4):
        key = f'v{i}'
        row = {'id': str(i), 'source_video_id': key, 'ticks': [1, 2, 3], 'query_s': 0,
               'response_s': 2, 'split': 'train'}
        rows.append(row)
        save_tensor(cache/f'video-{digest(key)[:24]}.pt', {'ticks': row['ticks'],
                    'visual': torch.randn(3, 4, 12)})
        save_tensor(cache/f'query-{i}.pt', {'query': torch.randn(12)})
    out = tmp_path/'train'
    train_arm(rows, cache, out, cfg, 'event_state', 'cpu', 'test')
    ck = torch.load(out/'checkpoint.pt', weights_only=True)
    assert ck['step'] == 2
    before = (out/'checkpoint.pt').read_bytes()
    train_arm(rows, cache, out, cfg, 'event_state', 'cpu', 'test')
    assert (out/'checkpoint.pt').read_bytes() == before
    with pytest.raises(ValueError):
        train_arm(rows, cache, out, {**cfg, 'lr': .01}, 'event_state', 'cpu', 'test')
    (out/'checkpoint.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        train_arm(rows, cache, out, cfg, 'event_state', 'cpu', 'test')


def test_parameter_match():
    cfg = read_json(Path(__file__).parents[1]/'configs/event_state/pilot.json')
    n = [sum(p.numel() for p in ResponseHead(2560, cfg, a).parameters())
         for a in ('temporal', 'event_state')]
    assert abs(n[0]-n[1])/n[1] < .1


def test_interrupted_resume_matches_uninterrupted(tmp_path, cfg, monkeypatch):
    import trace_vl.event_state.train as module
    cfg = {**cfg, 'epochs': 40, 'max_updates': 30}
    cache = tmp_path/'features'
    write_json(cache/'identity.json', {'model': 'synthetic-test'})
    rows = []
    torch.manual_seed(3)
    for i in range(2):
        key = f'v{i}'
        row = {'id': str(i), 'source_video_id': key, 'ticks': [1, 2, 3], 'query_s': 0,
               'response_s': 2, 'split': 'train'}
        rows.append(row)
        save_tensor(cache/f'video-{digest(key)[:24]}.pt', {'ticks': row['ticks'], 'visual': torch.randn(3, 4, 12)})
        save_tensor(cache/f'query-{i}.pt', {'query': torch.randn(12)})
    full, interrupted = tmp_path/'full', tmp_path/'interrupted'
    train_arm(rows, cache, full, cfg, 'gru', 'cpu', 'test')
    original = module.save_tensor

    def crash_after_save(path, value):
        original(path, value)
        if value['step'] == 25:
            raise RuntimeError('simulated power loss')

    monkeypatch.setattr(module, 'save_tensor', crash_after_save)
    with pytest.raises(RuntimeError, match='power loss'):
        train_arm(rows, cache, interrupted, cfg, 'gru', 'cpu', 'test')
    assert not (interrupted/'complete.json').exists()
    monkeypatch.setattr(module, 'save_tensor', original)
    train_arm(rows, cache, interrupted, cfg, 'gru', 'cpu', 'test')
    a = torch.load(full/'checkpoint.pt', weights_only=True)
    b = torch.load(interrupted/'checkpoint.pt', weights_only=True)
    assert a['step'] == b['step'] == 30
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)


def test_incomplete_sharded_model_not_accepted(tmp_path):
    from trace_vl.event_state.features import model_complete
    write_json(tmp_path/'config.json', {})
    write_json(tmp_path/'model.safetensors.index.json', {'weight_map': {'a': 'a.safetensors', 'b': 'b.safetensors'}})
    (tmp_path/'a.safetensors').write_bytes(b'a')
    assert not model_complete(tmp_path)
    (tmp_path/'b.safetensors').write_bytes(b'b')
    assert model_complete(tmp_path)


def test_prediction_delay_and_late_are_distinguished():
    r = {'ticks': [1, 2, 3, 4, 5], 'scores': [0, 0, 0, 0, 1], 'response_s': 1}
    result = outcome(r, .5)
    assert result['late'] and not result['miss'] and not result['on_time']


def test_completed_features_do_not_reload_model_or_replace_profile(tmp_path, monkeypatch):
    from trace_vl.event_state.features import encode_shard
    from trace_vl.v3 import backbone
    row = {'id': 'q', 'source_video_id': 'v'}
    save_tensor(tmp_path/f'video-{digest("v")[:24]}.pt', {'visual': torch.zeros(1)})
    save_tensor(tmp_path/'query-q.pt', {'query': torch.zeros(1)})
    write_json(tmp_path/'profile-0.json', {'seconds': 123})

    def forbidden(*args, **kwargs):
        raise AssertionError('Completed features must not load a model')

    monkeypatch.setattr(backbone, 'Backbone', forbidden)
    encode_shard([row], 'unused', tmp_path, {})
    assert read_json(tmp_path/'profile-0.json') == {'seconds': 123}


def test_streamingbench_csv_and_packed_videos_without_extraction(tmp_path, monkeypatch):
    import zipfile
    # Official CSV schema: the question ID is NOT a source-video ID.
    (tmp_path/'Proactive_Output_50.csv').write_text(
        '\ufeffquestion_id,task_type,question,time_stamp\n'
        'Proactive Output_50_1,Proactive Output,when ready,00:00:03\n')
    with zipfile.ZipFile(tmp_path/'Proactive Output_1-25.zip', 'w') as z:
        z.writestr('Proactive Output/sample_21/v_abcdefghijk.mp4', b'not decoded by inventory')
        z.writestr('Proactive Output/sample_22/another.mp4', b'not decoded by inventory')
        z.writestr('__MACOSX/._ignored.mp4', b'ignored')
        z.writestr('notes.txt', b'not media')

    def no_member_reads(*args, **kwargs):
        raise AssertionError('Inventory must not decompress video members')

    monkeypatch.setattr(zipfile.ZipFile, 'open', no_member_reads)
    ids, reports = benchmark_inventory([str(tmp_path)], True)
    assert ids == {'abcdefghijk', 'another'}
    assert reports[0]['archives'][0]['video_members'] == 2
    assert reports[0]['archives'][0]['mode'] == 'central_directory_only'
    assert not list(tmp_path.rglob('*.mp4'))
    assert reports == benchmark_inventory([str(tmp_path)], True)[1]


def test_benchmark_corrupt_zip_still_blocks_even_with_annotation_ids(tmp_path):
    write_json(tmp_path/'annotation.json', [{'video_id': 'known'}])
    (tmp_path/'incomplete.zip').write_bytes(b'partial download')
    with pytest.raises(ValueError, match='ZIP directory unreadable'):
        benchmark_inventory([str(tmp_path)], True)


def test_question_only_csv_and_empty_zip_not_mistaken_for_complete_inventory(tmp_path):
    import zipfile
    (tmp_path/'questions.csv').write_text('question_id\nProactive Output_50_1\n')
    with zipfile.ZipFile(tmp_path/'empty.zip', 'w') as z:
        z.writestr('README.txt', b'no media')
    with pytest.raises(ValueError, match='0 IDs'):
        benchmark_inventory([str(tmp_path)], True)
