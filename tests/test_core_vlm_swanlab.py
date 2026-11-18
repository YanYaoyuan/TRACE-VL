import hashlib
import json
import sys
import types

import pytest


@pytest.fixture
def fake_swan(monkeypatch):
    calls = {'logs': [], 'init': [], 'finished': 0}
    run = types.SimpleNamespace(mode='online',
        log=lambda values, step=None: calls['logs'].append((step, values)),
        finish=lambda: calls.update(finished=calls['finished'] + 1))
    def init(**kwargs):
        calls['init'].append(kwargs)
        run.mode = kwargs['mode']
        return run
    sdk = types.SimpleNamespace(init=init, get_url=lambda: 'https://swanlab.cn/test',
        Settings=lambda **kw: kw, Text=lambda content, caption=None: {'content': content, 'caption': caption})
    monkeypatch.setitem(sys.modules, 'swanlab', sdk)
    monkeypatch.setenv('TRACE_TRACKER', 'swanlab')
    monkeypatch.setenv('TRACE_SWANLAB', '1')
    monkeypatch.setenv('SWANLAB_MODE', 'online')
    monkeypatch.setenv('TRACE_SWANLAB_PROJECT', 'trace-test')
    monkeypatch.setenv('TRACE_SWANLAB_WORKSPACE', 'team')
    return calls, run


def test_live_entry_routes_to_online_swanlab_without_wandb(fake_swan, tmp_path, monkeypatch):
    from trace_vl.v3.tracking import Tracker, enabled_from_env
    calls, _ = fake_swan
    monkeypatch.setenv('TRACE_WANDB', '0')
    assert enabled_from_env()
    tracker = Tracker.start(True, 'plain', 'experiment', 'core-train', {'microbatch_size': 4},
                            directory=tmp_path / 'wandb')
    assert tracker.active and tracker.mode == 'online'
    args = calls['init'][0]
    assert args['project'] == 'trace-test' and args['workspace'] == 'team'
    assert args['mode'] == 'online' and args['on_online_error'] == 'offline'
    assert args['logdir'] == str(tmp_path / 'swanlog')
    assert args['settings']['interactive'] is False
    tracker.log({'loss': .4, 'nan': float('nan'), 'bool': True, 'nested': {1: 2}}, step=6)
    assert calls['logs'][-1] == (6, {'loss': .4, 'nested/1': 2})
    tracker.finish()
    assert calls['finished'] == 1
    monkeypatch.setenv('TRACE_SWANLAB', '0')
    assert not enabled_from_env()


def test_log_failure_preserves_local_training_and_records_error(fake_swan, tmp_path):
    from trace_vl.v3.swanlab_tracking import SwanTracker
    _, run = fake_swan
    tracker = SwanTracker.start(True, 't', None, 'test', directory=tmp_path)
    def fail(*args, **kwargs):
        raise ConnectionError('network down')
    run.log = fail
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{"loss": 0.2}')
    with pytest.warns(UserWarning, match='local logs retained'):
        tracker.log({'loss': .2}, step=1)
    assert tracker.error and path.read_text() == '{"loss": 0.2}'
    tracker.finish()


def test_relative_paths_chunks_and_raw_bytes_are_reconstructable(fake_swan, tmp_path):
    from trace_vl.v3.swanlab_tracking import SwanTracker
    calls, _ = fake_swan
    paths = []
    for arm in ('plain', 'temporal'):
        path = tmp_path / 'eval' / arm / 'report.json'
        path.parent.mkdir(parents=True)
        path.write_bytes(('中文\r\n' * 5000 + arm).encode())
        paths.append(path)
    tracker = SwanTracker.start(True, 't', None, 'test', directory=tmp_path / 'swanlog')
    result = tracker.upload('all', 'reports', paths, base=tmp_path)
    assert set(result) == {'eval/plain/report.json', 'eval/temporal/report.json'}
    for path in paths:
        entry = result[str(path.relative_to(tmp_path))]
        chunks = sorted((step, values[entry['key']]['content']) for step, values in calls['logs']
                        if entry['key'] in values)
        raw = ''.join(content for _, content in chunks).encode()
        assert raw == path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        assert len(chunks) > 1
    tracker.finish()


def test_upload_existing_logs_reports_and_external_operator_log(fake_swan, tmp_path):
    from trace_vl.core_vlm.swanlab_upload import upload
    calls, _ = fake_swan
    root = tmp_path / 'experiment'
    logs = root / 'logs'
    logs.mkdir(parents=True)
    (logs / 'plain-mb4.log').write_text('loading\n' + json.dumps({'update': 1, 'loss': .5}) + '\n')
    (root / 'report.json').write_text('{"completed":true,"score":0.7}')
    operator = tmp_path / 'operator.log'
    operator.write_text('operator output')
    result = upload(root, [operator])
    assert result['mode'] == 'online'
    assert set(result['files']) == {'logs/plain-mb4.log', 'report.json', 'operator/0-operator.log'}
    assert any('curves/logs/plain-mb4_log/loss' in values for _, values in calls['logs'])
    assert any('reports/report_json/score' in values for _, values in calls['logs'])
    assert (root / '.swanlab-upload/receipt.json').is_file()
    assert calls['finished'] == 1


def test_live_sidecar_follows_updates_without_changing_training_files(fake_swan, tmp_path, monkeypatch):
    from trace_vl.core_vlm import swanlab_upload
    calls, _ = fake_swan
    root = tmp_path / 'experiment'
    path = root / 'train/plain/metrics.jsonl'
    path.parent.mkdir(parents=True)
    first = '{"update":1,"loss":0.9}\n'
    second = '{"update":2,"loss":0.5}\n'
    path.write_text(first)
    slept = []
    def wait(seconds):
        if not slept:
            path.write_text(first + second)  # stand-in for the existing trainer
            slept.append(seconds)
        else:
            raise KeyboardInterrupt
    monkeypatch.setattr(swanlab_upload.time, 'sleep', wait)
    swanlab_upload.upload(root, watch=1)
    points = [(step, values['curves/train/plain/metrics_jsonl/loss'])
              for step, values in calls['logs'] if 'curves/train/plain/metrics_jsonl/loss' in values]
    assert points == [(1, .9), (2, .5)]
    assert path.read_text() == first + second


def test_collector_skips_checkpoints_source_sdk_cache_and_training_qa(tmp_path):
    from trace_vl.core_vlm.swanlab_upload import files_to_upload
    paths = ['train/plain/metrics.jsonl', 'eval/ovo/plain/predictions.jsonl', 'data/manifest.jsonl',
             'source/config.json', 'train/plain/checkpoint-002500/state.json',
             'eval/ovo/plain/questions/one.json', 'swanlog/run/config.json', '.swanlab-upload/receipt.json']
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    assert {str(p.relative_to(tmp_path)) for p in files_to_upload(tmp_path)} == set(paths[:2])


def test_real_sdk_saves_scalars_and_exact_text_chunks_offline(tmp_path, monkeypatch):
    swanlab = pytest.importorskip('swanlab')
    from trace_vl.v3.swanlab_tracking import SwanTracker
    monkeypatch.setenv('SWANLAB_MODE', 'offline')
    monkeypatch.setenv('SWANLAB_ROOT', str(tmp_path / 'global'))
    monkeypatch.setenv('TRACE_SWANLAB_PROJECT', 'trace-core-sdk-test')
    monkeypatch.delenv('TRACE_SWANLAB_WORKSPACE', raising=False)
    swanlab.merge_settings(swanlab.Settings(probe=swanlab.Settings.Probe(
        hardware=False, monitor=False, git=False, requirements=False)))
    tracker = SwanTracker.start(True, 'regression', 'test', 'test', directory=tmp_path / 'swanlog')
    assert tracker.active and tracker.mode == 'offline'
    tracker.log({'loss': .5, 'peak_gib': 12.}, step=100)
    file = tmp_path / 'report.json'
    raw = ('中文\r\n' * 5000).encode()
    file.write_bytes(raw)
    tracker.upload('report', 'reports', [file], base=tmp_path)
    tracker.summary({'candidate_seconds': {1: 24., 4: 12.}, 'complete': True})
    tracker.finish()
    assert tracker.error is None
    media = list((tmp_path / 'swanlog').rglob('media/text/*.txt'))
    contents = [p.read_bytes() for p in media]
    assert len(contents) >= 3
    # Every report chunk must exist byte-for-byte, despite the earlier scalar step 100.
    decoded = raw.decode()
    for i in range(0, len(decoded), 12000):
        assert decoded[i:i + 12000].encode() in contents
