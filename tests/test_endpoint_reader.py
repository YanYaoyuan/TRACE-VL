import pytest
import torch
from torch.nn import functional as F

pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_core_vlm_native import tiny_example  # noqa: E402

from trace_vl.complementary_memory.state_stream import Block, block_inputs, state_tensors  # noqa: E402
from trace_vl.core_vlm.endpoint_reader import EndpointReaderEpisode  # noqa: E402
from trace_vl.v3.positions import PositionCounter  # noqa: E402

HEAD, TAIL, ANSWER = [5, 6], [10, 11, 12], [40, 41, 2]


@pytest.fixture
def bb():
    from peft import LoraConfig, get_peft_model
    backbone, _, _ = tiny_example()
    backbone.text_config = backbone.model.config.text_config
    backbone.full_layers, backbone.linear_layers = [1], [0]
    backbone.vision_start, backbone.vision_end = 1, 2
    backbone.peft = get_peft_model(backbone.model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                                            target_modules=["q_proj", "v_proj"]))
    backbone.model.eval()
    return backbone


def blocks(n):
    generator = torch.Generator().manual_seed(34)
    return [Block(i, float(i + 1), (8, 9), torch.randn(4, 32, generator=generator).requires_grad_(),
                  torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), (2, 2)) for i in range(n)]


def episode(bb, items, policy="reservoir"):
    result = EndpointReaderEpisode(bb, HEAD, policy, recent=2, capacity=2, seed=17)
    for item in items:
        result.append(item)
    result.seal(items[-1].end)
    return result


def test_answer_only_cached_ce_matches_uncached_native_teacher_forcing(bb):
    items = blocks(4)
    reader = episode(bb, items, "full")
    actual, meta = reader.answer_loss(TAIL, ANSWER)
    counter = PositionCounter()
    parts = [(bb.embed_ids(HEAD), counter.text(len(HEAD)))]
    parts += [block_inputs(bb, block, counter) for block in items]
    teacher = [*TAIL, *ANSWER[:-1]]
    parts.append((bb.embed_ids(teacher), counter.text(len(teacher))))
    hidden = bb.lm(torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1),
                   cache=None, use_cache=False)
    expected = F.cross_entropy(bb.logits(hidden[:, -len(ANSWER):]).reshape(len(ANSWER), -1),
                               torch.tensor(ANSWER))
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-4)
    assert meta["answer_tokens"] == 3 and not meta["prefix_gradients"]


def test_answer_gradients_reach_lora_but_not_prefix_or_frozen_base(bb):
    items = blocks(8)
    reader = episode(bb, items)
    before, rng = reader.stream.snapshot(), reader.stream.rng.getstate()
    loss, _ = reader.answer_loss(TAIL, ANSWER)
    loss.backward()
    reader.stream.verify(before)
    assert reader.stream.rng.getstate() == rng
    assert reader.stream.audit()["kv_steps"] == 4
    assert all(b.tokens.grad is None for b in items)
    assert all(not t.requires_grad for t in state_tensors(bb, reader.stream.cache))
    trainable = [p for p in bb.model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert sum(float(p.grad.abs().sum()) for p in trainable) > 0
    assert all(p.grad is None for p in bb.model.parameters() if not p.requires_grad)


def test_source_update_averages_questions_closes_cache_and_changes_lora(bb):
    reader = episode(bb, blocks(8))
    questions = [(TAIL, ANSWER), ([12, 13], [42, 2])]
    expected = sum(float(reader.answer_loss(t, a)[0].detach()) for t, a in questions) / 2
    before = [p.detach().clone() for p in reader.trainable]
    optimizer = torch.optim.SGD(reader.trainable, lr=.1)
    # Outer no_grad contexts must not silently discard the normalized loss graph.
    with torch.no_grad():
        result = reader.train_step(questions, optimizer)
    assert result["mean_question_answer_ce"] == pytest.approx(expected)
    assert result["answer_tokens"] == 5 and result["optimizer_steps"] == 1
    assert result["gradient_norm_before_clip"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, reader.trainable, strict=True))
    with pytest.raises(RuntimeError, match="closed"):
        reader.answer_loss(TAIL, ANSWER)
    fresh = episode(bb, blocks(3))
    assert fresh.answer_loss(TAIL, ANSWER)[0].requires_grad


def test_external_parameter_update_invalidates_the_prefix(bb):
    reader = episode(bb, blocks(3))
    with torch.no_grad():
        reader.trainable[0].add_(.01)
    with pytest.raises(RuntimeError, match="stale"):
        reader.answer_loss(TAIL, ANSWER)


def test_gold_and_question_forks_cannot_change_live_writes(bb):
    reader = episode(bb, blocks(7))
    before = reader.stream.snapshot()
    reader.answer_loss(TAIL, ANSWER)
    reader.answer_loss([14, 15, 16, 17], [43, 44, 45, 2])
    reader.stream.verify(before)
    assert reader.stream.seen == 7 and reader.stream.audit()["kv_steps"] == 4


def test_rejects_future_endpoint_late_writes_and_silent_answer_truncation(bb):
    reader = EndpointReaderEpisode(bb, HEAD, recent=2, capacity=2)
    item = blocks(1)[0]
    reader.append(item)
    with pytest.raises(ValueError, match="causal prefix"):
        reader.seal(item.end - .1)
    reader.seal(item.end)
    with pytest.raises(RuntimeError, match="append"):
        reader.append(blocks(2)[-1])
    with pytest.raises(ValueError, match="answer budget"):
        reader.answer_loss(TAIL, ANSWER, max_answer_tokens=2)


def test_optimizer_cannot_update_base_or_unrelated_parameters(bb):
    reader = episode(bb, blocks(3))
    frozen = next(p for p in bb.model.parameters() if not p.requires_grad)
    optimizer = torch.optim.SGD([*reader.trainable, frozen], lr=.1)
    with pytest.raises(ValueError, match="exactly"):
        reader.train_step([(TAIL, ANSWER)], optimizer)


def test_inference_mode_is_rejected_before_prefix_creation_or_use(bb):
    with torch.inference_mode(), pytest.raises(ValueError, match="inference"):
        EndpointReaderEpisode(bb, HEAD)
    reader = episode(bb, blocks(2))
    with torch.inference_mode(), pytest.raises(ValueError, match="inference"):
        reader.answer_loss(TAIL, ANSWER)


def test_failed_update_closes_episode_without_reusing_partial_gradients(bb):
    reader = episode(bb, blocks(3))
    optimizer = torch.optim.SGD(reader.trainable, lr=.1)
    with pytest.raises(ValueError, match="answer budget"):
        reader.train_step([(TAIL, ANSWER)], optimizer, max_answer_tokens=1)
    with pytest.raises(RuntimeError, match="closed"):
        reader.answer_loss(TAIL, ANSWER)


def test_source_checkpoint_resume_matches_uninterrupted_adam_updates(bb, tmp_path, monkeypatch):
    from trace_vl.core_vlm.train import restore_checkpoint, save_checkpoint, tensor_digest

    # CPU CI exercises actual checkpoint/Adam/LoRA recovery. Runtime uses CUDA RNG.
    monkeypatch.setattr(torch.cuda, 'get_rng_state', torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', torch.set_rng_state)
    params = {n: p for n, p in bb.model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(list(params.values()), lr=.001, weight_decay=0.)
    first = episode(bb, blocks(8))
    metrics = [first.train_step([(TAIL, ANSWER)], optimizer)]
    save_checkpoint(tmp_path, 1, 'resume-test', bb, params, optimizer, 0, 1, metrics)
    episode(bb, blocks(9)).train_step([([13, 14], [42, 43, 2])], optimizer)
    expected = tensor_digest(params)
    restored_optimizer = torch.optim.AdamW(list(params.values()), lr=.001, weight_decay=0.)
    update, recovered = restore_checkpoint(tmp_path / 'checkpoint-000001', 'resume-test',
                                           params, restored_optimizer, 0, 1)
    assert update == 1 and recovered == metrics
    episode(bb, blocks(9)).train_step([([13, 14], [42, 43, 2])], restored_optimizer)
    assert tensor_digest(params) == expected
    assert all(int(s['step']) == 2 for s in restored_optimizer.state.values())
