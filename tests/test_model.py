from dataclasses import replace

import pytest
import torch

from trace_vl.config import ModelConfig
from trace_vl.data.dataset import VideoDataset, collate_samples
from trace_vl.models.model import TraceVL


def test_normalized_memory_stays_bounded_and_backpropagates_long_stream():
    cfg = ModelConfig(normalize_memory=True)
    model = TraceVL(cfg).train()
    batch = {
        "frames": torch.rand(2, 24, 3, 32, 32),
        "timestamps": torch.arange(24, dtype=torch.float32)[None].expand(2, -1) / 4,
        "questions": ["What changed?", "What happened?"],
        "answers": ["yes", "no"],
    }
    result = model(batch)
    result["loss"].backward()
    memory = result["state"].memory
    assert memory.working.square().mean(-1).max() <= 1.001
    assert memory.events[memory.valid].square().mean(-1).max() <= 1.001
    assert torch.isfinite(model.blocks[0].q.weight.grad).all()


def video(batch=1, steps=5):
    return torch.rand(batch, steps, 3, 32, 32), torch.arange(steps).float()[None].expand(batch, -1) / 2


def test_prefix_causality_and_query_independent_observation():
    model = TraceVL(ModelConfig()).eval()
    frames, times = video()
    changed = frames.clone()
    changed[:, 3:] = torch.rand_like(changed[:, 3:]) * 100
    with torch.no_grad():
        a, aux_a = model.observe(frames, times, rules=["watch object"])
        _, aux_b = model.observe(changed, times, rules=["watch object"])
        torch.testing.assert_close(aux_a["alert_logits"][:, :3], aux_b["alert_logits"][:, :3], rtol=0, atol=0)
        snapshot = a.memory.events.clone()
        working = a.memory.working.clone()
        model.evidence(a, ["Where did the person go?"])
        model.evidence(a, ["What color was the object?"])
        torch.testing.assert_close(a.memory.events, snapshot, rtol=0, atol=0)
        torch.testing.assert_close(a.memory.working, working, rtol=0, atol=0)


def test_state_is_bounded_and_timestamps_are_recorded():
    cfg = ModelConfig(event_capacity=3, cache_frames=2)
    model = TraceVL(cfg).eval()
    frames, times = video(steps=11)
    with torch.no_grad():
        state, _ = model.observe(frames, times)
    assert state.memory.events.shape == (1, 3, 4, 64)
    assert state.memory.valid.sum() == 3
    assert all(len(cache) == 2 for cache in state.caches)
    assert sorted(state.memory.known_at.flatten().tolist()) == [4.0, 4.5, 5.0]
    assert state.memory.before_valid.all()
    assert state.memory.pointer.item() == 2


def test_past_query_and_nonmonotonic_input_rejected():
    model = TraceVL(ModelConfig()).eval()
    frames, times = video(steps=2)
    with torch.no_grad():
        state, _ = model.observe(frames, times)
    with pytest.raises(ValueError, match="query precedes"):
        model.evidence(state, ["what?"], torch.tensor([0.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        model.observe_step(frames[:, 1], times[:, 1], state)


def test_alerts_are_invariant_to_timestamp_origin():
    model = TraceVL(ModelConfig()).eval()
    frames, times = video()
    with torch.no_grad():
        a, aux_a = model.observe(frames, times)
        b, aux_b = model.observe(frames, times + 64)
    torch.testing.assert_close(aux_a["alert_logits"], aux_b["alert_logits"])
    torch.testing.assert_close(a.memory.events, b.memory.events)
    torch.testing.assert_close(a.memory.known_at[a.memory.valid] + 64, b.memory.known_at[b.memory.valid])


def test_padding_holds_memory_and_remains_finite():
    model = TraceVL(ModelConfig()).eval()
    frames, times = video(steps=3)
    with torch.no_grad():
        state, _ = model.observe(frames[:, :1], times[:, :1])
        padded, aux = model.observe_step(frames[:, 1], times[:, 1], state, frame_valid=torch.tensor([False]))
    for name in state.memory.__dataclass_fields__:
        torch.testing.assert_close(getattr(state.memory, name), getattr(padded.memory, name))
    torch.testing.assert_close(padded.last_visual, state.last_visual)
    assert torch.isfinite(aux["alert_logits"]).all()


def test_future_capsule_is_masked():
    model = TraceVL(ModelConfig()).eval()
    frames, times = video(steps=2)
    with torch.no_grad():
        state, _ = model.observe(frames, times)
        modified = replace(state.memory, known_at=torch.full_like(state.memory.known_at, 100))
        _, info = model.reader(torch.zeros(1, 64), state.last_visual, modified, times[:, -1])
    assert not info["valid"].any()


@pytest.mark.parametrize("mode", ["capsule", "summary", "recent"])
def test_forward_backward_real_gradients(fixture_manifest, mode):
    cfg = ModelConfig(memory_mode=mode, capsule_tokens=1 if mode == "summary" else 4)
    model = TraceVL(cfg)
    dataset = VideoDataset(fixture_manifest, split="train")
    batch = collate_samples([dataset[0], dataset[1]])
    output = model(batch)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    for parameter in [
        model.blocks[0].q.weight,
        model.reader.layers[0]["attn"].in_proj_weight,
        model.projector[0].weight,
        model.memory.write_head[0].weight,
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    if mode == "recent":
        assert not output["state"].memory.valid.any()


def test_frozen_backbone_still_backpropagates_to_prefix(fixture_manifest):
    model = TraceVL(ModelConfig(freeze_vision=True, freeze_language=True)).train()
    assert not model.language.training and not model.vision.training
    sample = VideoDataset(fixture_manifest, split="train")[0]
    model(collate_samples([sample]))["loss"].backward()
    assert all(p.grad is None for p in model.language.parameters())
    assert model.projector[0].weight.grad.abs().sum() > 0


def test_learned_write_and_priority_run():
    cfg = ModelConfig(write_policy="learned", retention_policy="priority", event_capacity=2)
    model = TraceVL(cfg)
    frames, times = video(steps=7)
    state, aux = model.observe(frames, times, detach_every=3)
    assert state.memory.valid.sum() <= 2
    assert torch.isfinite(aux["utility_logits"]).all()
