"""CPU tests for pilot diagnostics (PR #3 review of pilot-003)."""

import json
import math

import pytest
import torch

from trace_vl.v3.diagnose import (
    ablation_report,
    mcnemar_exact,
    paired_delta,
    prediction_agreement,
    training_curves,
    validation_nll,
)
from trace_vl.v3.modules import effective_coverage, pool_weights, slot_part_pools


def preds_from(answers, predictions, videos):
    return {f"q{i}": {"id": f"q{i}", "answer": a, "prediction": p, "video_key": v, "scores": [-1.0, -2.0, -3.0]}
            for i, (a, p, v) in enumerate(zip(answers, predictions, videos))}  # fmt: skip


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    assert mcnemar_exact(3, 1) == pytest.approx(2 * (1 + 4) / 16)


def test_prediction_agreement_counts():
    preds = {"a": preds_from([0, 0, 0, 0], [0, 0, 1, 2], "vvww"), "b": preds_from([0, 0, 0, 0], [0, 1, 0, 1], "vvww")}
    out = prediction_agreement(preds, "a", "b")
    assert (out["both_correct"], out["only_a"], out["only_b"], out["both_wrong"]) == (1, 1, 1, 1)
    assert out["n"] == 4 and out["same_prediction_fraction"] == 0.25 and out["mcnemar_p"] == 1.0
    preds["b"].pop("q0")
    with pytest.raises(ValueError):
        prediction_agreement(preds, "a", "b")


def test_validation_nll_full_and_option_renormalized():
    preds = preds_from([0, 1], [0, 0], "vw")
    out = validation_nll(preds)
    assert out["n"] == 2 and out["nll_full_vocab"] == pytest.approx((1.0 + 2.0) / 2)
    log_z = math.log(math.exp(-1) + math.exp(-2) + math.exp(-3))
    assert out["nll_options"] == pytest.approx(((log_z + 1) + (log_z + 2)) / 2)


def test_training_curves_windows():
    metrics = [{"step": k, "ce": float(k), "gate_abs_mean": 0.1 * k, "note": "x"} for k in range(1, 11)]
    out = training_curves(metrics, window=3)
    assert out["first"]["ce"] == 2.0 and out["last"]["ce"] == 9.0 and "note" not in out["first"]


def test_paired_delta_and_ablation_report(tmp_path):
    full = preds_from([0] * 6, [0, 0, 0, 0, 1, 1], "aabbcc")
    gist = preds_from([0] * 6, [0, 0, 0, 0, 1, 1], "aabbcc")  # identical: the slot tokens changed nothing
    b4 = preds_from([0] * 6, [0] * 6, "aabbcc")
    delta = paired_delta({"x": full, "y": gist}, "x", "y")
    assert delta["delta_pp"] == 0.0 and delta["ci95_pp"] == [0.0, 0.0] and delta["videos"] == 3
    for name, rows in (("after", full), ("after_gistonly", gist), ("b4", b4)):
        (tmp_path / "eval" / name).mkdir(parents=True)
        (tmp_path / "eval" / name / "predictions.0of1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    report = ablation_report(tmp_path)
    assert list(report) == ["after"]
    assert report["after"]["agreement"]["same_prediction_fraction"] == 1.0
    assert report["after"]["gist_only_minus_b4"]["delta_pp"] == pytest.approx(-100 * 2 / 6, abs=1e-3)


def test_effective_coverage_bounds():
    uniform = pool_weights(4, 4, 16)[0]
    assert effective_coverage(uniform) == pytest.approx(1.0)
    one_cell = torch.zeros(3, 16)
    one_cell[:, 5] = 1.0
    assert effective_coverage(one_cell) == pytest.approx(1 / 16)
    assert 0.9 < effective_coverage(pool_weights(14, 14, 20)[0]) <= 1.0  # B4 regions cover the grid


def test_slot_part_weights_reproduce_the_pools():
    torch.manual_seed(0)
    x, a = torch.randn(12, 5), torch.rand(12)
    rows, cols = torch.arange(12) // 4, torch.arange(12) % 4
    tokens, _, _, weights = slot_part_pools(a, x, rows, cols, return_weights=True)
    assert torch.allclose(weights @ x, tokens, atol=1e-5)
    assert torch.allclose(weights.sum(-1), torch.ones(3))


def test_emit_slots_false_sends_gist_only():
    from trace_vl.v3.config import V3Config
    from trace_vl.v3.model import TraceV3Modules
    from trace_vl.v3.modules import CSTCache, TECache

    for mode in ("transition", "after"):
        cfg = V3Config.from_dict(
            {"cst": {"dim": 32, "layers": 2, "heads": 4, "window": 3},
             "transition": {"slots": 4, "gist": 2, "max_active": 2, "before_lag": 1, "i_step_every": 100,
                            "scene_cut_surprise": 1.01, "fixed_active": 2, "p_step_mode": mode, "emit_slots": False},
             "train": {"target_p_step_tokens": 8}}
        )  # fmt: skip
        torch.manual_seed(5)
        modules = TraceV3Modules(cfg, model_dim=16).eval()
        cst, te = CSTCache(), TECache()
        for step in range(3):
            adapted, z = modules.cst(torch.rand(12, 16) + 0.5, (3, 4), float(step), cst)
            info = modules.transition(adapted, z, (3, 4), te)
        assert not info.is_i_step and info.tokens.shape[0] == 2 and info.kinds == ["gist", "gist"]
        assert info.aux["slot_memory"] is not None  # CST memory unchanged by the ablation
        assert 0 < info.aux["coverage"] <= 1 and 0 < info.aux["gist_entropy"] <= 1
