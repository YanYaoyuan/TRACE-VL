import pytest
import torch
from test_core_vlm_native import tiny_example
from torch.nn import functional as F

from trace_vl.core_vlm.native import NativeVideoQALoss, encode_native
from trace_vl.core_vlm.preservation import original_answer_logits
from trace_vl.core_vlm.temporal import attach_temporal


def example():
    bb, layout, row = tiny_example()
    bb.add_lora(2, 4, 0.)
    with torch.no_grad():
        steps = encode_native(bb, layout)
    net = NativeVideoQALoss(bb, mode='grid', pool_grid=(1, 1), answer_end_token_id=2)
    return bb, net, steps, row


def test_original_teacher_is_invariant_and_restores_adapter_and_mixed_training_flags():
    bb, net, steps, row = example()
    net.train()
    bb.visual.eval()
    flags = [m.training for m in bb.model.modules()]
    teacher = original_answer_logits(net, steps, row)
    assert not teacher.requires_grad
    with torch.no_grad():
        for n, p in net.named_parameters():
            if 'lora_B' in n:
                p.fill_(.05)
    changed, _, _ = net.answer_logits(steps, row)
    assert not torch.allclose(teacher, changed)
    torch.testing.assert_close(original_answer_logits(net, steps, row), teacher, atol=0, rtol=0)
    assert [m.training for m in bb.model.modules()] == flags
    assert all(not m.disable_adapters for m in bb.model.modules() if isinstance(getattr(m, 'disable_adapters', None), bool))


def test_teacher_exception_restores_state(monkeypatch):
    bb, net, steps, row = example()
    net.train()
    flags = [m.training for m in bb.model.modules()]
    monkeypatch.setattr(NativeVideoQALoss, 'answer_logits', lambda *a: (_ for _ in ()).throw(RuntimeError('fail')))
    with pytest.raises(RuntimeError, match='fail'):
        original_answer_logits(net, steps, row)
    assert [m.training for m in bb.model.modules()] == flags
    assert all(not m.disable_adapters for m in bb.model.modules() if isinstance(getattr(m, 'disable_adapters', None), bool))


def test_kl_direction_and_checkpointed_gradients_match_uncheckpointed():
    bb, net, steps, row = example()
    attach_temporal(net, 'causal')
    with torch.no_grad():
        for n, p in net.named_parameters():
            if 'lora_B' in n:
                p.normal_(0, .03)
    net.train()
    teacher = original_answer_logits(net, steps, row)
    logits, targets, _ = net.answer_logits(steps, row)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets)
    expected = ce + .5 * 4 * F.kl_div(F.log_softmax(logits / 2, -1),
        F.log_softmax(teacher / 2, -1), reduction='none', log_target=True).sum(-1).mean()
    loss, meta = net(steps, row, teacher_logits=teacher, preservation_weight=.5)
    torch.testing.assert_close(loss, expected)
    assert meta['preservation_kl'] > 0 and meta['teacher_used']
    loss.backward()
    grads = {n:p.grad.clone() for n,p in net.named_parameters() if p.grad is not None}
    net.zero_grad()
    bb.enable_checkpointing()
    teacher2 = original_answer_logits(net, steps, row)
    loss2, _ = net(steps, row, teacher_logits=teacher2, preservation_weight=.5)
    loss2.backward()
    torch.testing.assert_close(loss2, loss)
    assert any(n.startswith('temporal.') and g.abs().sum()>0 for n,g in grads.items())
    assert any('lora_' in n and g.abs().sum()>0 for n,g in grads.items())
    for n,p in net.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=1e-5, rtol=1e-4)
    # Neither the shared frozen backbone nor teacher accumulates trainable grads.
    assert all(p.grad is None for p in bb.model.parameters() if not p.requires_grad)


def test_compute_control_teacher_zero_weight_preserves_ce_gradients():
    _, net, steps, row = example()
    with torch.no_grad():
        for n, p in net.named_parameters():
            if 'lora_B' in n:
                p.normal_(0, .02)
    net.train()
    ce, _ = net(steps, row)
    ce.backward()
    expected = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    net.zero_grad()
    teacher = original_answer_logits(net, steps, row)
    control, _ = net(steps, row, teacher_logits=teacher, preservation_weight=0.)
    control.backward()
    torch.testing.assert_close(control, ce, atol=0, rtol=0)
    for n, p in net.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n], atol=0, rtol=0)


def test_three_arm_audit_rejects_teacher_or_weight_drift(tmp_path, monkeypatch):
    import json

    from test_core_vlm_mixed import make_training

    from trace_vl.core_vlm import mixed_experiment
    from trace_vl.core_vlm import preservation_experiment as runner
    monkeypatch.setattr(mixed_experiment, 'ARMS', runner.ARMS)
    for name in runner.ARMS:
        make_training(tmp_path, name)
        runfile = tmp_path / name / 'run.json'
        data = json.loads(runfile.read_text())
        data['config'].update(preservation_teacher=True, preservation_temperature=2.,
                              preservation_weight=0. if name == 'ce' else .5)
        runfile.write_text(json.dumps(data))
        logs = tmp_path / name / 'metrics.jsonl'
        entries = [json.loads(x) for x in logs.read_text().splitlines()]
        for r in entries:
            r.update(teacher_questions=1, answer_ce=.5, preservation_kl_per_question=.01)
        logs.write_text(''.join(json.dumps(r) + '\n' for r in entries))
    assert runner.audit_training(tmp_path, 2)['passed']
    runfile = tmp_path / 'ce/run.json'
    data = json.loads(runfile.read_text())
    data['config']['preservation_weight'] = .1
    runfile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='weights changed'):
        runner.audit_training(tmp_path, 2)
