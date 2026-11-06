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


def test_transition_tokens_are_convex_combinations_at_init():
    cfg = small_config(i_step_every=100)
    modules = TraceV3Modules(cfg, model_dim=16).eval()
    cst, te = CSTCache(), TECache()
    x0 = torch.rand(6, 16)
    for k in range(3):
        x = x0 + 0.5 * k
        adapted, z = modules.cst(x, (2, 3), float(k), cst)
        info = modules.transition(adapted, z, (2, 3), te)
    gist = info.tokens[: cfg.transition.gist]
    assert (gist >= adapted.min(0).values - 1e-5).all() and (gist <= adapted.max(0).values + 1e-5).all()


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


def test_alert_head_shapes():
    modules = TraceV3Modules(small_config(), model_dim=16)
    logits = modules.alert(torch.randn(16), torch.randn(3, 16), torch.zeros(4))
    assert logits.shape == (3,)
    emb = modules.event(torch.randn(5, 16))
    assert torch.allclose(emb.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_grid_coords_range():
    c = grid_coords(3, 4)
    assert c.shape == (12, 2) and float(c.min()) == 0.0 and float(c.max()) == 1.0
    assert np.isclose(float(c[5, 0]), 1 / 3)
