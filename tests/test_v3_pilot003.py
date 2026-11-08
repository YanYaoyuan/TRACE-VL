"""Pilot-003: fixed P-step budget (top-k slots), after-only arm, identical budgets across arms (CPU)."""

import pytest
import torch

from trace_vl.v3.config import V3Config, load_v3_config
from trace_vl.v3.model import TraceV3Modules
from trace_vl.v3.modules import CSTCache, TECache, slot_part_pools


def small(mode, fixed=2, threshold=100.0):
    # mass_threshold 100: a learned gate would emit nothing; fixed_active must still emit exactly `fixed` slots.
    return V3Config.from_dict(
        {"cst": {"dim": 32, "layers": 2, "heads": 4, "window": 3},
         "transition": {"slots": 4, "gist": 2, "max_active": 2, "mass_threshold": threshold, "before_lag": 1,
                        "i_step_every": 100, "scene_cut_surprise": 1.01, "fixed_active": fixed, "p_step_mode": mode},
         "train": {"target_p_step_tokens": 2 + 3 * fixed}}
    )  # fmt: skip


def run(mode, train=False, steps=3):
    cfg = small(mode)
    torch.manual_seed(5)
    modules = TraceV3Modules(cfg, model_dim=16)
    modules.train(train)
    cst, te = CSTCache(), TECache()
    torch.manual_seed(6)
    infos = []
    for step in range(steps):
        adapted, z = modules.cst(torch.rand(12, 16) + 0.5, (3, 4), float(step), cst)
        infos.append(modules.transition(adapted, z, (3, 4), te))
    return infos, adapted


@pytest.mark.parametrize("train", [False, True])
def test_three_arms_emit_identical_budgets_and_share_slot_memory(train):
    out = {mode: run(mode, train) for mode in ("transition", "after", "pooled")}
    p_steps = {mode: [i for i in infos if not i.is_i_step] for mode, (infos, _) in out.items()}
    assert all(len(v) == 2 for v in p_steps.values())  # first step is the I-step; scene cuts are off
    for mode, infos in p_steps.items():
        for info in infos:
            assert info.tokens.shape[0] == 2 + 3 * 2, mode  # gist + 3 x fixed_active, despite the high threshold
            assert info.aux["active_slots"] == 2
            assert float(info.aux["expected_tokens"]) == 8.0 and not info.aux["expected_tokens"].requires_grad
    last = {mode: infos[-1] for mode, infos in p_steps.items()}
    ref = last["transition"].aux
    for mode in ("after", "pooled"):  # same slots, same CST memory and centroids: only LM-facing tokens differ
        assert torch.allclose(last[mode].aux["slot_memory"], ref["slot_memory"])
        assert last[mode].aux["active_centroids"] == ref["active_centroids"]
    assert last["after"].kinds == ["gist"] * 2 + ["after"] * 6
    assert last["after"].groups[2:] == last["transition"].groups[2:]  # retention groups per slot, like trios
    assert last["transition"].kinds[2:] == ["before", "after", "delta"] * 2


def test_fixed_mode_has_no_sampling_or_straight_through():
    a, _ = run("transition", train=True)
    b, _ = run("transition", train=True)
    assert [i.aux["active_centroids"] for i in a] == [i.aux["active_centroids"] for i in b]  # deterministic
    info = a[-1]
    assert torch.equal(info.tokens, run("transition", train=False)[0][-1].tokens)  # train == eval tokens


def test_after_tokens_are_norm_matched_real_token_pools():
    (infos, adapted) = run("after")
    info = infos[-1]
    ref = adapted.norm(dim=-1).mean()
    assert torch.allclose(info.tokens.norm(dim=-1), torch.full_like(info.tokens[:, 0], float(ref)), rtol=1e-4)
    assert info.rows.max() <= 2 and info.cols.max() <= 3 and info.rows.min() >= 0


def test_slot_part_pools_partition_and_fallback():
    x = torch.eye(6)
    rows, cols = torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 1, 0, 1, 0, 1])
    a = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10, 0.0])
    pools, r, c = slot_part_pools(a, x, rows, cols)
    assert pools.shape == (3, 6) and torch.allclose(pools.sum(-1), torch.ones(3))  # convex combinations
    assert torch.allclose(pools[0], torch.tensor([1.0, 0, 0, 0, 0, 0]))  # heaviest third: token 0 alone
    assert (pools[1] > 0).sum() >= 1 and (pools[2] > 0).sum() >= 1
    dominant = torch.tensor([0.9, 0.05, 0.05, 0, 0, 0])  # one token holds > 2/3: later parts fall back
    pools, _, _ = slot_part_pools(dominant, x, rows, cols)
    assert torch.allclose(pools[0], x[0]) and torch.isfinite(pools).all()
    assert all(torch.count_nonzero(p) >= 1 for p in pools)


def test_config_validation():
    with pytest.raises(ValueError, match="fixed_active"):
        V3Config.from_dict({"transition": {"fixed_active": 9}})
    with pytest.raises(ValueError, match="fixed_active"):
        V3Config.from_dict({"transition": {"fixed_active": -1}})
    with pytest.raises(ValueError, match="p_step_mode"):
        V3Config.from_dict({"transition": {"p_step_mode": "delta"}})
    assert V3Config.from_dict({"transition": {"p_step_mode": "after"}}).transition.fixed_active == 0


def test_pilot003_configs_differ_only_in_p_step_mode():
    dicts = {}
    for arm, mode in (("transition", "transition"), ("after", "after"), ("b4", "pooled")):
        cfg, _ = load_v3_config(f"configs/v3/pilot003_{arm}.yaml")
        t = cfg.transition
        assert t.p_step_mode == mode and t.fixed_active == 4 and t.scene_cut_surprise > 1.0
        assert cfg.train.target_p_step_tokens == t.gist + 3 * t.fixed_active == 20
        assert cfg.train.steps == 100 and cfg.train.grad_accum == 1 and cfg.train.warmup == 10
        d = cfg.to_dict()
        d["transition"]["p_step_mode"] = None
        dicts[arm] = d
    assert dicts["transition"] == dicts["after"] == dicts["b4"]
