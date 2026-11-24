import copy
import hashlib
import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trace_vl.core_vlm.data import decode_observation, sha256
from trace_vl.core_vlm.native import NativeVideoQALoss, encode_native
from trace_vl.core_vlm.pool_probe import VARIANTS, assert_replay, intervention, make_report, run
from trace_vl.core_vlm.pool_probe_cache import (
    FrozenVisionCache,
    copy_verified,
    observation_key,
    prepare_frames,
    read_frames,
    validate_observation,
)
from trace_vl.core_vlm.temporal import attach_temporal


def test_history_control_removes_only_previous_condition_and_restores_after_errors():
    from test_core_vlm_native import tiny_example

    bb, layout, row = tiny_example()
    steps = encode_native(bb, layout)
    row = {**row, 'native_layout': layout}
    net = NativeVideoQALoss(bb, mode='grid', pool_grid=(1, 1))
    plain = net.prefix(steps, row)[0]
    attach_temporal(net, 'causal-pool')
    with torch.no_grad():
        net.temporal.score.weight.normal_(std=.8)
        net.temporal.predict.weight.normal_(std=.4)
    module = net.temporal
    weights = {k: p.clone() for k, p in module.state_dict().items()}
    baseline = net.prefix(steps, row)[0]
    with intervention(net, 'history_off'):
        reset = net.prefix(steps, row)[0]
    hook = module.predict.register_forward_hook(lambda _, args, output: torch.zeros_like(output))
    expected = net.prefix(steps, row)[0]
    hook.remove()
    assert torch.equal(reset, expected)
    assert not torch.equal(baseline, reset)
    with intervention(net, 'module_off'):
        assert torch.equal(net.prefix(steps, row)[0], plain)
    assert torch.equal(net.prefix(steps, row)[0], baseline)
    for variant in VARIANTS:
        with pytest.raises(RuntimeError), intervention(net, variant):
            raise RuntimeError('failed generation')
        assert net.temporal is module and module.history_enabled
    assert all(torch.equal(v, module.state_dict()[k]) for k, v in weights.items())
    with pytest.raises(ValueError):
        with intervention(SimpleNamespace(), 'history_off'):
            pass


@pytest.fixture
def source_video(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg and ffprobe required')
    video = tmp_path / 'example.mkv'
    frames = np.stack([np.full((32, 32, 3), i * 20, np.uint8) for i in range(8)])
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', '32x32', '-r', '2', '-i', 'pipe:0', '-c:v', 'ffv1', str(video)],
                   input=frames.tobytes(), check=True)
    return video


def test_local_replay_needs_no_probe_and_matches_recorded_rgb(source_video, tmp_path, monkeypatch):
    row = {'video': str(source_video), 'protocol': 'online', 'query_time': 1.6}
    images, _, _, audit = decode_observation(row, steps=2, pixels=1024)
    validate_observation(row, audit, 2)
    local = tmp_path / 'local' / 'copy'
    copy_verified(source_video, local, audit['video_sha256'])
    import trace_vl.core_vlm.data as data
    monkeypatch.setattr(data, 'probe_video', lambda *a, **kw: pytest.fail('must not reprobe'))
    saved = tmp_path / 'frames.npz'
    prepare_frames(local, saved, audit)
    assert np.array_equal(read_frames(saved, audit), images)
    # A resumed run can read the verified local frames even if NAS is unavailable.
    source_video.unlink()
    prepare_frames(local, saved, audit)
    np.savez(saved, images=np.zeros_like(images))
    with pytest.raises(ValueError, match='decoded RGB'):
        read_frames(saved, audit)
    local.write_bytes(b'changed')
    with pytest.raises(ValueError, match='local file changed'):
        copy_verified(source_video, local, audit['video_sha256'])


def test_reference_selection_refuses_future_duplicate_and_changed_rgb(source_video, tmp_path):
    row = {'video': str(source_video), 'protocol': 'online', 'query_time': 1.6}
    _, _, _, audit = decode_observation(row, steps=2, pixels=1024)
    future = {**audit, 'frame_times': [[0., .5], [1., 2.]]}
    with pytest.raises(ValueError, match='future'):
        validate_observation(row, future, 2)
    bad = {**audit, 'frame_indices': [0, 1, 1, 3]}
    with pytest.raises(ValueError, match='selection'):
        validate_observation(row, bad, 2)
    with pytest.raises(ValueError, match='decoded RGB'):
        prepare_frames(source_video, tmp_path / 'wrong.npz', {**audit, 'decoded_rgb_sha256': 'bad'})


def test_frozen_vision_cache_roundtrip_and_corruption_with_real_tiny_backbone(tmp_path):
    from test_core_vlm_native import tiny_example

    bb, layout, _ = tiny_example()
    layout.audit = dict(video_sha256='v', frame_indices=[0, 1, 2, 3], frame_times=layout.frame_times,
                        frame_hw=[64, 64], decoded_rgb_sha256='rgb')
    cache = FrozenVisionCache(tmp_path, {'model': 'one'})
    with torch.no_grad():
        expected = encode_native(bb, layout)
        first = cache.get(bb, layout, verify_fresh=True)
        loaded = cache.get(bb, layout, verify_fresh=True)
    assert cache.writes == 1 and cache.hits == 1
    assert all(torch.equal(x.tokens, y.tokens) and torch.equal(x.tokens, z.tokens)
               for x, y, z in zip(expected, first, loaded))
    assert FrozenVisionCache(tmp_path, {'model': 'other'}).root != cache.root
    path = cache.root / f'{observation_key(layout.audit)}.pt'
    value = torch.load(path, weights_only=True)
    value['features'][0] += 1
    torch.save(value, path)
    with pytest.raises(ValueError, match='cache changed'):
        cache.get(bb, layout)


def sample_report():
    cases, refs, maps = [], {}, {}
    for ds, task in (('ovo', 'HLD'), ('nextqa', 'TN')):
        rows = [dict(id=str(i), video=f'v{i}', source_group=f'v{i}', question='which?',
                     options=['x', 'y'], answer=i % 2, task=task, split='val') for i in range(4)]
        obs = dict(video_sha256='v', frame_indices=[0, 1], frame_times=[[0., 1.]],
                   decoded_rgb_sha256='rgb', input_ids_sha256='prompt',
                   timestamp_policy='native-index-fps', timestamp_text_seconds=[.5])
        source = {r['id']: dict(id=r['id'], run='old', video=r['video'], choice=r['answer'],
                  answer=chr(65+r['answer']), token_ids=[r['answer']], visual_tokens=16, observation=obs)
                  for r in rows}
        refs[ds] = {'plain': copy.deepcopy(source), 'causal_pool': source}
        maps[ds] = {v: {qid: {**copy.deepcopy(p), 'run': 'new', 'intervention': v}
                       for qid, p in source.items()} for v in VARIANTS}
        cases += [{'dataset': ds, 'row': r, 'reference': source[r['id']]} for r in rows]
    return cases, maps, refs


def test_report_uses_same_questions_and_observations_and_never_promotes_gate():
    cases, maps, refs = sample_report()
    report = make_report(cases, maps, refs)
    assert report['completed'] and not report['training_launched']
    assert 'not a separately trained' in report['scope']
    assert report['results']['nextqa']['paired']['normal_minus_history_off']['primary']['choice']['delta_pp'] == 0
    for field, value in (('choice', 1), ('token_ids', [99]), ('visual_tokens', 32)):
        bad = copy.deepcopy(maps)
        bad['ovo']['normal']['0'][field] = value
        with pytest.raises(ValueError, match='replay differs'):
            make_report(cases, bad, refs)
    bad = copy.deepcopy(maps)
    del bad['ovo']['history_off']['0']
    with pytest.raises(ValueError, match='coverage'):
        make_report(cases, bad, refs)
    bad = copy.deepcopy(maps)
    bad['nextqa']['module_off']['1']['observation']['frame_indices'] = [1, 2]
    with pytest.raises(ValueError, match='unmatched observation'):
        make_report(cases, bad, refs)


def test_replay_includes_generated_text_not_only_choice():
    _, maps, refs = sample_report()
    good = maps['ovo']['normal']['0']
    assert_replay(refs['ovo']['causal_pool']['0'], good)
    with pytest.raises(ValueError, match='answer'):
        assert_replay(refs['ovo']['causal_pool']['0'], {**good, 'answer': 'A. x'})


def test_answer_changes_expose_cancelling_gains_and_losses():
    cases, maps, refs = sample_report()
    # One control answer improves and one worsens: zero net accuracy difference.
    for values in (maps['ovo']['normal'], refs['ovo']['causal_pool']):
        values['0'].update(choice=1, token_ids=[1], answer='B')
    maps['ovo']['history_off']['1'].update(choice=0, token_ids=[0], answer='A')
    result = make_report(cases, maps, refs)['results']['ovo']
    assert result['paired']['normal_minus_history_off']['primary']['choice']['delta_pp'] == 0
    change = result['answer_changes']['normal_minus_history_off']['primary']
    assert change == dict(n=4, choice_changed=2, choice_changed_fraction=.5,
                          generated_tokens_changed=2, left_only_correct=1, right_only_correct=1,
                          both_wrong_changed=0, changed_ids=['0', '1'])
    maps['ovo']['module_off']['2']['choice'] = -1
    with pytest.raises(ValueError, match='invalid intervention choice'):
        make_report(cases, maps, refs)


@pytest.mark.parametrize('relative', ['base', 'base/nested', '.'])
def test_probe_refuses_source_output_overlap_before_writing(tmp_path, relative):
    args = SimpleNamespace(base=tmp_path / 'base', out=tmp_path / relative, data=tmp_path / 'data')
    with pytest.raises(ValueError, match='separate directories'):
        run(args)
    assert not args.base.exists() and not args.data.exists()


def test_prepare_only_builds_real_cross_package_code_identity(tmp_path, monkeypatch):
    import trace_vl.core_vlm.pool_probe as probe

    def prepared(base, out, data, workers):
        (out / 'inputs.json').write_text('{"id":"fixture-source"}')
        return {'id': 'fixture-source'}

    monkeypatch.setattr(probe, 'prepare', prepared)
    args = SimpleNamespace(base=tmp_path / 'base', out=tmp_path / 'out',
                           data=tmp_path / 'data', workers=2, gpus=[0], prepare_only=True)
    run(args)
    identity = json.loads((args.out / 'run.json').read_text())
    assert identity['source_id'] == 'fixture-source'
    assert identity['code']['../v3/backbone.py'] == sha256(
        __import__('pathlib').Path(probe.__file__).parent / '../v3/backbone.py')


def test_stage_space_failure_precedes_copy_or_decode(tmp_path, monkeypatch):
    import trace_vl.core_vlm.pool_probe_cache as cache
    root, video = tmp_path / 'out', tmp_path / 'source'
    video.write_bytes(b'not a real video')
    audit = dict(video_sha256=sha256(video), frame_indices=[0, 1], frame_times=[[0., 1.]],
                 frame_hw=[32, 32], decoded_rgb_sha256=hashlib.sha256(b'').hexdigest())
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='space'):
        cache.stage_data([{'row': {'video': str(video)}, 'reference': {'observation': audit}}],
                         tmp_path, {}, root)
    assert not (root / 'videos').exists()


def test_stage_deduplicates_and_resumes_without_nas(source_video, tmp_path):
    from trace_vl.core_vlm.pool_probe_cache import stage_data

    row = dict(video=str(source_video), protocol='online', query_time=1.6)
    _, _, _, audit = decode_observation(row, steps=2, pixels=1024)
    cases = [dict(row={**row, 'id': str(i)}, reference={'observation': audit}) for i in range(2)]
    model = tmp_path / 'original-model'
    model.mkdir()
    (model / 'config.json').write_text('{"model":"fixture"}')
    identity = {'config.json': sha256(model / 'config.json')}
    target = tmp_path / 'prepared'
    result = stage_data(cases, model, identity, target, workers=2, reserve_gib=0)
    assert result['videos'] == 1 and result['observations'] == 1 and result['questions'] == 2
    assert json.loads((target / 'model/config.json').read_text()) == {'model': 'fixture'}
    source_video.unlink()
    (model / 'config.json').unlink()
    assert stage_data(cases, model, identity, target, workers=2, reserve_gib=0) == result
    (target / 'model/config.json').write_text('{}')
    with pytest.raises(ValueError, match='local file changed'):
        stage_data(cases, model, identity, target, workers=2, reserve_gib=0)


def test_shell_launcher_is_valid_and_defaults_to_current_a100_paths():
    script = __import__('pathlib').Path(__file__).parents[1] / 'scripts/launch_core_pool_probe_a100.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    assert '/data/yyy/tracevl-runs/core-pool-2efc7cc-001' in script.read_text()
