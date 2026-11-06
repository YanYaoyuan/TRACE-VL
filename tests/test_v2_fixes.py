import math
from dataclasses import replace

import pytest
import torch

from trace_vl.config import ModelConfig, factor_grid
from trace_vl.models.backbones import ByteTokenizer, PromptFormat, chat_template_parts, training_sequences
from trace_vl.models.memory import EvidenceReader, TransitionMemory
from trace_vl.models.model import TraceVL
from trace_vl.runtime import initialize_weights
from trace_vl.training import learning_rate_factor, parameter_groups

V2 = dict(
    prompt_format="chat",
    state_update="bounded",
    state_kv_norm=True,
    retrieval_gradient=True,
    query_projection="separate",
    token_selection="pool",
    motion_layout="grid",
)


def batch(size=2, steps=6, scale=1.0):
    return {
        "frames": torch.rand(size, steps, 3, 32, 32) * scale,
        "timestamps": torch.arange(steps, dtype=torch.float32)[None].expand(size, -1) / 2,
        "questions": ["What changed?", "Who left?"][:size],
        "answers": ["the door opened", "nobody"][:size],
    }


def test_factor_grid_is_most_square():
    assert factor_grid(96) == (8, 12)
    assert factor_grid(224) == (14, 16)
    assert factor_grid(16) == (4, 4)
    assert factor_grid(7) == (1, 7)


def test_invalid_v2_choice_rejected():
    with pytest.raises(ValueError, match="state_update"):
        ModelConfig(state_update="gru")


def test_bounded_state_update_cannot_explode():
    torch.manual_seed(0)
    cfg = ModelConfig(state_update="bounded")
    model = TraceVL(cfg).eval()
    with torch.no_grad():
        state, _ = model.observe(*[batch(steps=40, scale=100.0)[k] for k in ("frames", "timestamps")])
    # A convex mix of zeros and LayerNorm outputs stays within the LayerNorm elementwise bound.
    assert state.memory.working.abs().max() <= math.sqrt(cfg.dim) + 1e-3


def test_full_v2_model_trains_with_contrast_loss_and_stays_causal():
    torch.manual_seed(0)
    model = TraceVL(ModelConfig(**V2)).train()
    output = model(batch(), {"contrast": 0.5, "contrast_margin": 0.2})
    assert set(output["losses"]) == {"answer", "contrast"}
    output["loss"].backward()
    assert torch.isfinite(model.query_projection.weight.grad).all()
    assert model.blocks[0].k.weight.grad.abs().sum() > 0
    model.eval()
    frames, times = batch(1)["frames"], batch(1)["timestamps"]
    changed = frames.clone()
    changed[:, 3:] = torch.rand_like(changed[:, 3:]) * 100
    with torch.no_grad():
        _, a = model.observe(frames, times)
        _, b = model.observe(changed, times)
    torch.testing.assert_close(a["alert_logits"][:, :3], b["alert_logits"][:, :3], rtol=0, atol=0)
    with torch.no_grad():
        state, _ = model.observe(frames, times)
        answers, _ = model.answer(state, ["What changed?"], max_new_tokens=3)
    assert isinstance(answers[0], str)


def test_contrast_loss_skipped_for_single_sample():
    output = TraceVL(ModelConfig(**V2))(batch(1), {"contrast": 0.5})
    assert "contrast" not in output["losses"]


def test_pooled_tokens_have_fixed_positions_and_motion_grid_positions_differ():
    model = TraceVL(ModelConfig(token_selection="pool", motion_layout="grid")).eval()
    with torch.no_grad():
        _, a = model._visual_tokens(torch.rand(1, 3, 32, 32), None)
        _, b = model._visual_tokens(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    torch.testing.assert_close(a, b)
    rgb, motion = a[0, :8], a[0, 8:]
    assert rgb.unique(dim=0).shape[0] == 8
    assert motion.unique(dim=0).shape[0] == 4
    assert ((a >= 0) & (a <= 1)).all()


def test_straight_through_retrieval_keeps_forward_and_adds_score_gradient():
    torch.manual_seed(0)
    cfg = ModelConfig(event_capacity=6, read_events=2)
    memory = TransitionMemory(cfg).initial(1, "cpu", torch.float32)
    memory.valid[:] = True
    memory.known_at[:] = torch.arange(6.0)
    events = torch.randn(1, 6, 4, cfg.dim)
    current, query_time = torch.randn(1, 12, cfg.dim), torch.tensor([10.0])
    results = []
    for flag in (False, True):
        torch.manual_seed(1)
        reader = EvidenceReader(replace(cfg, retrieval_gradient=flag))
        memory.events = events.clone().requires_grad_(True)
        query = torch.randn(1, cfg.dim, generator=torch.Generator().manual_seed(2)).requires_grad_(True)
        out, info = reader(query, current, memory, query_time)
        out.square().sum().backward()
        results.append((out.detach(), info["indices"], memory.events.grad.clone()))
    (out_a, idx_a, grad_a), (out_b, idx_b, grad_b) = results
    torch.testing.assert_close(out_a, out_b)
    assert torch.equal(idx_a, idx_b)
    unselected = torch.ones(6, dtype=torch.bool)
    unselected[idx_a[0]] = False
    # Hard top-k never sends gradient to unselected events; the straight-through softmax does.
    assert grad_a[0, unselected].abs().sum() == 0
    assert grad_b[0, unselected].abs().sum() > 0


def test_raw_prompt_format_matches_original_layout():
    tokenizer = ByteTokenizer()
    prompts, answers = ["what is it?"], ["a cat"]
    plain = training_sequences(tokenizer, prompts, answers, 64, "cpu")
    formatted = training_sequences(tokenizer, prompts, answers, 64, "cpu", PromptFormat(tokenizer, "raw"))
    for a, b in zip(plain, formatted):
        assert torch.equal(a, b)


def test_chat_format_keeps_suffix_when_question_is_truncated():
    tokenizer = ByteTokenizer()
    fmt = PromptFormat(tokenizer, "chat")
    ids = fmt.prompt("x" * 500, limit=20)
    assert tokenizer.decode(ids).endswith("\nAnswer: ")
    assert tokenizer.decode(fmt.head) == "Video:\n"


def test_chat_template_parts_splits_around_user_content():
    class Fake:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            assert not enable_thinking
            return f"<user>{messages[0]['content']}</user><assistant>"

    assert chat_template_parts(Fake()) == ("<user>", "</user><assistant>")


def test_learning_rate_schedule():
    settings = {"lr_schedule": "cosine", "warmup_steps": 10, "max_steps": 110, "min_lr_ratio": 0.1}
    assert learning_rate_factor(0, settings) == pytest.approx(0.1)
    assert learning_rate_factor(9, settings) == pytest.approx(1.0)
    assert learning_rate_factor(10, settings) == pytest.approx(1.0)
    assert learning_rate_factor(109, settings) == pytest.approx(0.1, abs=1e-3)
    assert learning_rate_factor(50, {}) == 1.0


def test_norm_and_query_parameters_are_exempt_from_decay():
    model = TraceVL(ModelConfig())
    decay, other = parameter_groups(model, 0.01, exclude_norm_decay=True)
    assert other["weight_decay"] == 0.0
    exempt = {id(p) for p in other["params"]}
    assert id(model.memory.state_queries) in exempt and id(model.blocks[0].temporal_norm.weight) in exempt
    assert id(model.blocks[0].q.weight) not in exempt
    assert len(decay["params"]) + len(other["params"]) == sum(1 for _ in model.parameters())


def test_warm_start_from_original_checkpoint_into_v2(tmp_path):
    torch.manual_seed(0)
    old = TraceVL(ModelConfig())
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "config": {"model": ModelConfig().to_dict()}, "step": 5}, path)
    new = TraceVL(ModelConfig(**V2))
    report = initialize_weights(new, path)
    assert "query_projection" in report["changed_context_settings"]
    with torch.no_grad():
        torch.testing.assert_close(new.encode_text(["what?"]), old.encode_text(["what?"]))


def test_prefix_diagnostic_separates_shared_offset_from_video_differences():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/diagnose_prefix.py"
    spec = importlib.util.spec_from_file_location("diagnose_prefix", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shared = torch.full((4, 8), 10.0)
    collapsed = [shared + 1e-3 * torch.randn(4, 8) for _ in range(3)]
    distinct = [torch.randn(4, 8) for _ in range(3)]
    assert module.cross_video_cosine(collapsed) > 0.999
    assert module.between_video_fraction(collapsed) < 1e-6
    assert module.between_video_fraction(distinct) > 0.3
