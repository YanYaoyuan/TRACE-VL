import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trace_vl.core_vlm.native import NativeVideoQALoss, encode_native
from trace_vl.core_vlm.static_pool import make_report, matched_training
from trace_vl.core_vlm.temporal import CausalPatchPool, StaticPatchPool, attach_temporal, load_temporal
from trace_vl.core_vlm.train import tensor_digest


def test_static_and_causal_constructors_have_identical_initial_parameters_and_rng():
    torch.manual_seed(17)
    causal = CausalPatchPool(32)
    after_causal = torch.get_rng_state()
    torch.manual_seed(17)
    static = StaticPatchPool(32)
    assert torch.equal(torch.get_rng_state(), after_causal)
    assert tensor_digest(causal.state_dict()) == tensor_digest(static.state_dict())
    assert causal.history_enabled and not static.history_enabled


def test_static_has_no_past_dependency_and_learns_spatial_weights():
    torch.manual_seed(5)
    net = StaticPatchPool(12, width=8)
    x = torch.randn(3, 4, 4, 12)
    baseline = x.mean((1, 2))[:, None]
    assert torch.equal(net(x, (4, 4), (1, 1), baseline), baseline)
    with torch.no_grad():
        net.score.weight.normal_(std=.1)
    before = net(x, (4, 4), (1, 1), baseline)
    modified = x.clone()
    modified[0] += torch.randn_like(modified[0])
    new_baseline = modified.mean((1, 2))[:, None]
    after = net(modified, (4, 4), (1, 1), new_baseline)
    assert torch.equal(before[1:], after[1:])
    assert not torch.equal(before[0], after[0])
    predictor = net.predict.weight.detach().clone()
    scorer = net.score.weight.detach().clone()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.)
    for _ in range(3):
        optimizer.zero_grad()
        net(x, (4, 4), (1, 1), baseline).square().mean().backward()
        assert net.predict.weight.grad is not None and not net.predict.weight.grad.count_nonzero()
        assert net.appearance.weight.grad.abs().sum() > 0
        optimizer.step()
    assert torch.equal(net.predict.weight, predictor)
    assert not torch.equal(net.score.weight, scorer)


def test_static_checkpoint_restores_parameters_and_keeps_history_disabled(tmp_path):
    from test_core_vlm_native import tiny_example

    bb, layout, row = tiny_example()
    steps = encode_native(bb, layout)
    net = NativeVideoQALoss(bb, mode='grid', pool_grid=(1, 1))
    attach_temporal(net, 'static-pool')
    with torch.no_grad():
        net.temporal.score.weight.normal_(std=.2)
    prefix = net.prefix(steps, row)[0]
    state = {'parameters': {'temporal.' + k: v for k, v in net.temporal.state_dict().items()}}
    torch.save(state, tmp_path / 'training.pt')
    restored = NativeVideoQALoss(bb, mode='grid', pool_grid=(1, 1))
    attach_temporal(restored, 'static-pool')
    load_temporal(restored, tmp_path)
    assert not restored.temporal.history_enabled
    assert torch.equal(prefix, restored.prefix(steps, row)[0])


def training_fixture():
    run = dict(model={'weights': 'bytes'}, manifest_sha256='same', ids=['q'], videos={'v': 'hash'},
               world_size=4, objective='CE + KL', initial_checkpoint=None,
               config=dict(model='/nas-data/original', temporal='causal-pool', updates=2500,
                           seed=17, feature_cache='/data/cache', lr=2e-5))
    current = copy.deepcopy(run)
    current['config'].update(model='/data/verified-copy', temporal='static-pool')
    initial = dict(lora_sha256='same', temporal_history_enabled=False)
    logs = [dict(update=i, question_ids_by_rank=[['q']], lr=2e-5, answer_tokens=2,
                 teacher_questions=1, loss=.4, gradient_norm=1., temporal_gradient_norm_after_clip=.1,
                 temporal_history_enabled=False, vision_cache_hits_rank0=i,
                 vision_cache_writes_rank0=0) for i in (1, 2)]
    return run, current, initial, logs


def test_training_audit_allows_relocated_same_bytes_but_refuses_recipe_and_sample_drift():
    reference, current, initial, logs = training_fixture()
    assert matched_training(reference, current, initial, initial, logs, logs)['passed']
    # Prefix of the exact full schedule, not a separate two-update LR schedule.
    assert matched_training(reference, current, initial, initial, logs, logs[:1])['updates'] == 1
    for field, value in [('lr', 1e-3), ('question_ids_by_rank', [['other']]), ('teacher_questions', 0)]:
        changed = copy.deepcopy(logs)
        changed[1][field] = value
        with pytest.raises(ValueError, match='schedule'):
            matched_training(reference, current, initial, initial, logs, changed)
    for field, value in [('temporal_history_enabled', True), ('vision_cache_writes_rank0', 1),
                          ('temporal_gradient_norm_after_clip', 0.)]:
        changed = copy.deepcopy(logs)
        changed[1][field] = value
        with pytest.raises(ValueError):
            matched_training(reference, current, initial, initial, logs, changed)
    changed = copy.deepcopy(current)
    changed['model']['weights'] = 'different'
    with pytest.raises(ValueError, match='model'):
        matched_training(reference, changed, initial, initial, logs, logs)
    changed = copy.deepcopy(current)
    changed['config']['seed'] = 29
    with pytest.raises(ValueError, match='recipe'):
        matched_training(reference, changed, initial, initial, logs, logs)


def report_fixture():
    from test_core_pool_probe import sample_report

    cases, _, references = sample_report()
    static = {}
    for ds, refs in references.items():
        for p in refs.values():
            for value in p.values():
                value['visual_tokens'] = 128
        static[ds] = {k: {**copy.deepcopy(p), 'run': 'static', 'history_enabled': False,
                         'architecture': 'static-pool'} for k, p in refs['causal_pool'].items()}
    return cases, static, references


def test_report_pairs_independently_trained_static_with_both_frozen_references():
    cases, static, references = report_fixture()
    r = make_report(cases, static, references)
    assert r['completed'] and not r['architecture_accepted']
    assert r['newly_trained_arms'] == ['static_pool']
    assert set(r['results']['ovo']['paired']) == {
        'causal_pool_minus_static_pool', 'static_pool_minus_plain', 'causal_pool_minus_plain'}
    changed = copy.deepcopy(static)
    changed['ovo']['0']['choice'] = 1
    report = make_report(cases, changed, references)
    assert report['results']['ovo']['paired']['causal_pool_minus_static_pool']['primary']['choice']['delta_pp'] == 25
    for field, value in [('history_enabled', True), ('visual_tokens', 160), ('choice', -1)]:
        bad = copy.deepcopy(static)
        bad['ovo']['0'][field] = value
        with pytest.raises(ValueError):
            make_report(cases, bad, references)
    del static['nextqa']['0']
    with pytest.raises(ValueError, match='coverage'):
        make_report(cases, static, references)


def test_launcher_shell_and_train_parser_accept_static_architecture(monkeypatch):
    from trace_vl.core_vlm import train

    path = Path(__file__).parents[1] / 'scripts/launch_core_static_pool_a100.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    seen = []
    monkeypatch.setattr(train, 'run', seen.append)
    monkeypatch.setattr(sys, 'argv', ['train', '--model', '/data/model', '--manifest', '/data/manifest',
                                     '--out', '/data/new-run', '--temporal', 'static-pool'])
    train.main()
    assert seen[0].temporal == 'static-pool'


@pytest.mark.parametrize('smoke_passes', [False, True])
def test_orchestrator_uses_full_recipe_prefix_and_gates_resume(tmp_path, monkeypatch, smoke_passes):
    from trace_vl.core_vlm import static_pool

    args = SimpleNamespace(base=tmp_path / 'base', observations=tmp_path / 'observations',
                           out=tmp_path / 'new-control', smoke_only=False)
    cfg = dict(model='/data/model', updates=2500, accumulation=1, microbatch_size=8,
               rank=8, lr=2e-5, seed=17, save_every=250, cache_mb=4096, native_check=True,
               pixels=65536, steps=8, pool_grid=[4, 4], timestamp_policy='native-index-fps',
               gpus=[0, 1, 2, 3], minimum_free_gib=30, preservation_weight=.5)
    inputs = dict(config=cfg, manifest='/data/manifest', feature_cache='/data/cache', base=str(args.base))
    monkeypatch.setattr(static_pool, 'prepare', lambda args: inputs)
    jobs = []
    monkeypatch.setattr(static_pool, 'run_jobs', lambda tasks, env, log_root: jobs.extend(tasks))
    monkeypatch.setattr(static_pool, 'audit_training', lambda base, out, n: {'passed': True, 'updates': n})

    def evaluation(inputs, out, smoke, env):
        if smoke and not smoke_passes:
            raise RuntimeError('GPU parity failed')
        if smoke:
            (out / 'smoke-ok.json').write_text(json.dumps(dict(completed=True, smoke=True, questions=8)))

    monkeypatch.setattr(static_pool, 'evaluate', evaluation)
    if smoke_passes:
        static_pool.run(args)
        assert [x[0] for x in jobs] == ['gpu-check', 'train-smoke', 'train-full']
        assert jobs[-1][1] == jobs[-2][1][:-2]
    else:
        with pytest.raises(RuntimeError, match='GPU parity'):
            static_pool.run(args)
        assert [x[0] for x in jobs] == ['gpu-check', 'train-smoke']
    cmd = jobs[1][1]
    assert cmd[-2:] == ['--stop-after', '2']
    assert cmd[cmd.index('--updates') + 1] == '2500'
    assert '--examples' not in cmd and cmd[cmd.index('--seed') + 1] == '17'
