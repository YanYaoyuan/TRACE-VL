"""CPU tests for v3 components that do not need the Qwen3.5 weights.

GPU/backbone checks are `trace-vl v3 check` (equality gate) and `trace-vl v3 profile`.
"""

import json

import numpy as np
import pytest
import torch

from trace_vl.v3.config import MemoryConfig, V3Config, load_v3_config
from trace_vl.v3.data import convert_ovo, step_times
from trace_vl.v3.kv import KVMemory
from trace_vl.v3.model import TraceV3Modules
from trace_vl.v3.modules import CSTCache, TECache, grid_coords
from trace_vl.v3.positions import PositionCounter


def small_config(**transition):
    return V3Config.from_dict(
        {"cst": {"dim": 32, "layers": 2, "heads": 4, "window": 3},
         "transition": {"slots": 4, "gist": 2, "max_active": 2, "mass_threshold": 0.0, "before_lag": 1, **transition}}
    )  # fmt: skip


def test_positions_follow_qwen35_rope_rule():
    pos = PositionCounter()
    text = pos.text(3)
    assert text.tolist() == [[0, 1, 2]] * 3
    grid = pos.grid(2, 3)
    assert grid[0].tolist() == [3] * 6  # temporal = block start
    assert grid[1].tolist() == [3, 3, 3, 4, 4, 4]  # height = start + row
    assert grid[2].tolist() == [3, 4, 5, 3, 4, 5]  # width = start + col
    assert pos.state() == 3 + 3  # advance by max(h, w)
    sparse = pos.points(torch.tensor([1]), torch.tensor([2]), 2, 3)
    assert sparse[:, 0].tolist() == [6, 7, 8] and pos.state() == 9  # same advance as a full block


def test_cst_is_identity_at_init_and_cache_is_bounded():
    cfg = small_config()
    modules = TraceV3Modules(cfg, model_dim=16).eval()
    cache = CSTCache()
    for k in range(5):
        x = torch.randn(6, 16)
        out, z = modules.cst(x, (2, 3), float(k), cache)
        assert torch.equal(out, x)
        assert z.shape == (6, 32)
    assert all(len(layer) == cfg.cst.window for layer in cache.keys)
    assert len(cache.times) == cfg.cst.window


def test_cst_gate_receives_gradient_at_init():
    modules = TraceV3Modules(small_config(), model_dim=16)
    out, _ = modules.cst(torch.randn(6, 16), (2, 3), 0.0, CSTCache())
    out.sum().backward()
    assert modules.cst.gate.grad is not None and modules.cst.gate.grad.abs().sum() > 0


def test_transition_encoder_i_and_p_steps():
    cfg = small_config(i_step_every=3)
    modules = TraceV3Modules(cfg, model_dim=16).eval()
    cst, te = CSTCache(), TECache()
    kinds = []
    for k in range(6):
        x = torch.randn(6, 16)
        adapted, z = modules.cst(x, (2, 3), float(k), cst)
        info = modules.transition(adapted, z, (2, 3), te)
        kinds.append(info.is_i_step)
        assert info.rows.max() <= 1 and info.cols.max() <= 2 and info.rows.min() >= 0
        assert info.tokens.shape[0] == info.rows.shape[0] == len(info.kinds) == len(info.groups)
        if info.is_i_step:
            assert torch.equal(info.tokens, adapted) and set(info.kinds) == {"frame"}
        else:
            assert info.kinds[: cfg.transition.gist] == ["gist"] * cfg.transition.gist
            n_trans = info.tokens.shape[0] - cfg.transition.gist
            assert n_trans % 3 == 0 and n_trans // 3 <= cfg.transition.max_active
            assert "pred_loss" in info.aux
    assert kinds[0] is True  # first step is always an I-step
    assert kinds[1:3] == [False, False] and kinds[3] is True


def test_synthesized_tokens_are_norm_matched_and_nonzero_on_static_input():
    """Regression (PR #3 review): static content made delta = after - before exactly zero -> 1e12 grads."""
    cfg = small_config(i_step_every=100)
    modules = TraceV3Modules(cfg, model_dim=16).eval()
    cst, te = CSTCache(), TECache()
    x = torch.rand(6, 16) + 0.5
    for k in range(4):  # identical frames every step
        adapted, z = modules.cst(x, (2, 3), float(k), cst)
        info = modules.transition(adapted, z, (2, 3), te)
    assert not info.is_i_step and info.aux["active_slots"] > 0
    ref = adapted.norm(dim=-1).mean()
    norms = info.tokens.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, float(ref)), rtol=1e-4)
    deltas = [tok for tok, kind in zip(info.tokens, info.kinds) if kind == "delta"]
    assert deltas and all(float(d.norm()) > 0.5 * float(ref) for d in deltas)


def test_transition_tracker_is_bounded_and_keeps_onsets():
    from trace_vl.v3.modules import TransitionTracker

    tracker = TransitionTracker(resolve_seconds=5.0, capacity=3)
    vec = torch.tensor([[1.0, 0.0, 0.0]])
    for t in range(10):  # the same transition re-emitted every second
        tracker.update(float(t), [(2.0, 2.0)], vec)
    assert len(tracker.tracks) == 1 and tracker.tracks[0]["onset"] == 0.0
    assert float(tracker.clock(9.0, 0.0)[0]) == pytest.approx(np.log1p(9.0))
    for t in range(10, 20):  # many different transitions
        tracker.update(float(t), [(float(t % 7), 0.0)], torch.randn(1, 3))
    assert len(tracker.tracks) <= 3
    tracker.update(100.0, [], None)  # nothing seen for > resolve_seconds
    assert tracker.tracks == []


def test_eval_fingerprint_covers_adapter_content_and_videos(tmp_path):
    from trace_vl.v3.harness import run_fingerprint

    records = tmp_path / "r.jsonl"
    records.write_text(json.dumps({"id": "a", "video": "v.mp4"}) + "\n")
    (tmp_path / "v.mp4").write_bytes(b"0" * 10)
    adapter = tmp_path / "lora"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"one")
    rows = [{"id": "a", "video": "v.mp4"}]
    a = run_fingerprint(V3Config(), records, "v3", None, str(adapter), str(tmp_path), rows)
    b = run_fingerprint(V3Config(), records, "v3", None, None, str(tmp_path), rows)
    (adapter / "adapter_model.safetensors").write_bytes(b"two")
    c = run_fingerprint(V3Config(), records, "v3", None, str(adapter), str(tmp_path), rows)
    (tmp_path / "v.mp4").write_bytes(b"0" * 11)
    d = run_fingerprint(V3Config(), records, "v3", None, str(adapter), str(tmp_path), rows)
    assert len({a["id"], b["id"], c["id"], d["id"]}) == 4


def test_score_rejects_mixed_runs(tmp_path):
    from trace_vl.v3.harness import score

    base = {"source": "s", "task": "t", "answer": 0, "prediction": 0, "lm_tokens_per_step": 1, "kv_mb": 1,
            "answer_seconds": 0.1, "video_key": "v"}  # fmt: skip
    (tmp_path / "predictions.0of1.jsonl").write_text(
        json.dumps({**base, "id": "a", "run": "r1"}) + "\n" + json.dumps({**base, "id": "b", "run": "r2"}) + "\n"
    )
    with pytest.raises(ValueError, match="mixes"):
        score(str(tmp_path))


def test_full_v3_parameter_budget_is_small():
    report = TraceV3Modules(V3Config()).parameter_report()
    assert 30e6 < report["total"] < 150e6, report


def test_kv_policies():
    mem = KVMemory(MemoryConfig(policy="drift", window_seconds=2.0, budget_tokens=4))
    mem.append(3, "prompt", -1.0, ("prompt",), protected=True)
    for step in range(6):
        mem.append(2, "frame", float(step), ("step", step))
        mem.scores[("step", step)] = [0.5, 0.9, 0.1, 0.0, 0.0, 0.0][step]
    mask = mem.keep_mask(now=5.0)
    kept = {g for g, m in zip(mem.group, mask.tolist()) if m}
    # Prompt and the 2 s window (steps 3-5) always stay; the 4-token budget takes the two highest-drift
    # older steps (1, then 0); step 2 is evicted.
    assert {("prompt",), ("step", 3), ("step", 4), ("step", 5)} <= kept
    assert ("step", 1) in kept and ("step", 0) in kept and ("step", 2) not in kept
    mem.apply(mask)
    assert len(mem) == int(mask.sum())
    window = KVMemory(MemoryConfig(policy="window", window_seconds=1.0))
    window.append(2, "frame", 0.0, ("step", 0))
    window.append(2, "frame", 5.0, ("step", 1))
    assert window.keep_mask(5.0).tolist() == [False, False, True, True]
    assert KVMemory(MemoryConfig(policy="full")).keep_mask(10.0) is None


def test_step_times_are_causal_and_subsample():
    times = step_times(10.0, 1.0, 0.5)
    assert times[0] == 0.25 and all(t + 0.25 <= 10.0 + 1e-6 for t in times)
    assert len(step_times(100.0, 1.0, 0.5, max_steps=10)) == 10


def test_convert_ovo_keeps_mcq_tasks(tmp_path):
    items = [
        {"id": 1, "task": "EPM", "video": "a.mp4", "realtime": 12, "question": "q", "options": ["x", "y"], "gt": 1},
        {"id": 2, "task": "REC", "video": "b.mp4", "test_info": []},
    ]
    src = tmp_path / "ovo.json"
    src.write_text(json.dumps(items))
    report = convert_ovo(src, tmp_path / "out.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert report["records"] == 1 and report["skipped_forward_tasks"] == {"REC": 1}
    assert rows[0]["task"] == "backward/EPM" and rows[0]["query_time"] == 12.0 and rows[0]["answer"] == 1


def test_configs_load():
    for name in ("native_full", "native_window", "native_drift", "native_fifo", "native_budget16k", "v3_tkm", "stage1"):
        cfg, _ = load_v3_config(f"configs/v3/{name}.yaml")
        assert cfg.backbone.name == "Qwen/Qwen3.5-2B"
    with pytest.raises(ValueError):
        V3Config.from_dict({"memory": {"policy": "magic"}})


def _encoded(dim=16):
    from trace_vl.v3.alert import EncodedQuestions, Question

    questions = [
        Question("abnormal", "noul", "Is anything abnormal happening?"),
        Question("incident", "choice", "Which incident is this?", ["none", "fight", "fall"]),
    ]
    return EncodedQuestions(questions, torch.randn(2, dim), [None, torch.randn(3, dim)])


def test_alert_head_outputs_isolation_and_state_read():
    from trace_vl.v3.alert import DecisionMemory, EncodedQuestions, alert_loss, public

    torch.manual_seed(0)
    modules = TraceV3Modules(small_config(), model_dim=16).eval()
    enc = _encoded()
    taps = torch.randn(len(modules.cfg.alert.taps), 6, 16)
    xy = grid_coords(2, 3)
    memory = DecisionMemory(8)
    memory.add(torch.randn(5, 16), 1.0)
    states = torch.randn(*modules.alert.state_shape)
    with torch.no_grad():
        both = modules.alert(enc, taps, xy, memory, 3.0, torch.zeros(4), states)
        alone = modules.alert(EncodedQuestions(enc.questions[:1], enc.text[:1], enc.options[:1]), taps, xy, memory,
                              3.0, torch.zeros(4), states)  # fmt: skip
        no_state = modules.alert(enc, taps, xy, memory, 3.0, torch.zeros(4), None)
    assert 0.0 <= float(both[0]["value"]) <= 1.0
    assert torch.allclose(both[1]["probabilities"].sum(), torch.tensor(1.0)) and 0 <= float(both[1]["confidence"]) <= 1
    # Isolation: adding another question does not change this question's answer.
    assert torch.allclose(both[0]["value"], alone[0]["value"], atol=1e-6)
    # The Gated DeltaNet state read is used when available.
    assert not torch.allclose(both[0]["value"], no_state[0]["value"])
    out = public(both)
    assert out[0]["kind"] == "noul" and out[1]["choice"] in (0, 1, 2) and len(out[1]["probabilities"]) == 3
    modules.train()
    results = modules.alert(enc, taps, xy, memory, 3.0, torch.zeros(4), None)
    loss, stats = alert_loss(results, {"abnormal": 1.0, "incident": 2})
    loss.backward()
    assert torch.isfinite(loss) and "alert_noul" in stats and "alert_choice" in stats
    assert modules.alert.noul.weight.grad is not None


def test_alert_question_validation_and_targets():
    from trace_vl.v3.alert import Question
    from trace_vl.v3.train import alert_questions, alert_target

    with pytest.raises(ValueError):
        Question("x", "box", "where?")  # only classification kinds
    with pytest.raises(ValueError):
        Question("x", "choice", "which?", [str(i) for i in range(256)])
    with pytest.raises(ValueError):
        Question("x", "noul", "is it?", ["yes"])
    record = {"rules": ["A bag is left unattended."], "alert_spans": [[0, 10.0, 20.0]],
              "alerts": [{"id": "inc", "kind": "choice", "text": "Which?", "options": ["none", "fall"],
                          "label_spans": [[1, 5.0, 6.0]]}]}  # fmt: skip
    questions, specs = alert_questions(record)
    assert [q.id for q in questions] == ["inc", "rule0"]
    assert alert_target(specs[1], 15.0) == 1.0 and alert_target(specs[1], 25.0) == 0.0
    assert alert_target(specs[0], 5.5) == 1 and alert_target(specs[0], 7.0) == 0


def test_grid_coords_range():
    c = grid_coords(3, 4)
    assert c.shape == (12, 2) and float(c.min()) == 0.0 and float(c.max()) == 1.0
    assert np.isclose(float(c[5, 0]), 1 / 3)


def test_cst_checkpointing_matches_plain_forward_and_gradients():
    """M1 capacity fix (PR #3): per-layer activation checkpointing must not change results."""
    torch.manual_seed(0)
    outs = []
    for flag in (True, False):
        cfg = small_config()
        cfg.cst.checkpoint = flag
        torch.manual_seed(1)
        modules = TraceV3Modules(cfg, model_dim=16).train()
        with torch.no_grad():
            modules.cst.gate.fill_(0.5)
        cache, total = CSTCache(), 0
        torch.manual_seed(2)
        for k in range(4):
            out, z = modules.cst(torch.randn(6, 16), (2, 3), float(k), cache)
            total = total + out.square().sum() + z.square().sum()
        total.backward()
        outs.append((total.detach(), modules.cst.layers[0].qkv.weight.grad.clone(), modules.cst.gate.grad.clone()))
    for a, b in zip(*outs):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_transition_tracker_expires_before_matching_across_gaps():
    """Regression (PR #3 review): a stale track must not be revived after an observation gap."""
    from trace_vl.v3.modules import TransitionTracker

    tracker = TransitionTracker(resolve_seconds=12.0)
    vec = torch.tensor([[1.0, 0.0, 0.0]])
    tracker.update(0.0, [(2.0, 2.0)], vec)
    tracker.update(100.0, [(2.0, 2.0)], vec)
    assert len(tracker.tracks) == 1 and tracker.tracks[0]["onset"] == 100.0
    assert float(tracker.clock(100.0, 0.0)[0]) == 0.0  # age 0, not 100 s
    tracker.update(105.0, [(2.0, 2.0)], vec)  # within resolve_seconds: same track continues
    assert tracker.tracks[0]["onset"] == 100.0


def test_letter_logprobs_match_log_softmax_in_fp32():
    """fp32 letter scoring (bf16 tie fix, PR #3) equals the plain log-softmax when everything is fp32."""
    from types import SimpleNamespace

    from trace_vl.v3.backbone import Backbone

    torch.manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    fake = SimpleNamespace(lm_head=head)
    hidden = torch.randn(16)
    letters = [3, 7, 11, 42]
    ours = Backbone.letter_logprobs(fake, hidden, letters)
    ref = head(hidden).log_softmax(-1)[letters]
    assert torch.allclose(ours, ref, atol=1e-5)
