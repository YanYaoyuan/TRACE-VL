"""Pilot-004: duration-only change, intermediate checkpoints, model-neutral selector diagnostics (CPU)."""

import json
from pathlib import Path

import pytest
import torch

from trace_vl.v3.config import V3Config, load_v3_config
from trace_vl.v3.model import TraceV3Modules
from trace_vl.v3.modules import CSTCache, TECache, row_entropy, token_stats
from trace_vl.v3.pilot import training_complete, write_completion
from trace_vl.v3.train import selector_params


def test_row_entropy_and_token_stats():
    uniform = torch.ones(2, 16)
    one_hot = torch.zeros(1, 16)
    one_hot[0, 3] = 1
    assert torch.allclose(row_entropy(uniform), torch.ones(2)) and float(row_entropy(one_hot)) == pytest.approx(0, abs=1e-6)
    same = torch.ones(5, 8)
    assert token_stats(same)["cos_mean"] == pytest.approx(1.0) and token_stats(same)["effective_rank"] == pytest.approx(1.0)
    basis = torch.eye(5, 8)
    assert token_stats(basis)["cos_mean"] == pytest.approx(0.0) and token_stats(basis)["effective_rank"] == pytest.approx(5.0)


def small(mode):
    return V3Config.from_dict(
        {"cst": {"dim": 32, "layers": 2, "heads": 4, "window": 3},
         "transition": {"slots": 4, "gist": 2, "max_active": 2, "before_lag": 1, "i_step_every": 100,
                        "scene_cut_surprise": 1.01, "fixed_active": 2, "p_step_mode": mode},
         "train": {"target_p_step_tokens": 8}}
    )  # fmt: skip


def run(mode, diagnostics):
    torch.manual_seed(5)
    modules = TraceV3Modules(small(mode), model_dim=16).eval()
    modules.transition.diagnostics = diagnostics
    cst, te = CSTCache(), TECache()
    torch.manual_seed(6)
    for step in range(3):
        adapted, z = modules.cst(torch.rand(12, 16) + 0.5, (3, 4), float(step), cst)
        info = modules.transition(adapted, z, (3, 4), te)
    return info


@pytest.mark.parametrize("mode", ["transition", "after", "pooled"])
def test_diagnostics_do_not_change_outputs_and_report_every_mode(mode):
    off, on = run(mode, False), run(mode, True)
    assert torch.equal(off.tokens, on.tokens) and off.kinds == on.kinds and torch.equal(off.rows, on.rows)
    assert not any(k.startswith("diag_") for k in off.aux)
    for key in ("diag_gist_entropy_min", "diag_gist_logit_std", "diag_gist_token_cos", "diag_slot_entropy_mean",
                "diag_slot_entropy_min", "diag_slot_attn_spread", "diag_emitted_cos", "diag_emitted_rank"):
        assert key in on.aux, (mode, key)
    assert 0 <= on.aux["diag_slot_entropy_min"] <= on.aux["diag_slot_entropy_mean"] <= 1 + 1e-6
    assert 1 <= on.aux["diag_emitted_rank"] <= on.tokens.shape[0] + 1e-4
    assert ("diag_part_entropy_mean" in on.aux) == (mode == "after")


def test_selector_groups_cover_the_pooling_parameters():
    groups = selector_params(TraceV3Modules(small("after"), model_dim=16))
    assert set(groups) == {"gist", "slots", "surprise", "cst_gate", "cst_layers"} and all(groups.values())


def test_checkpoint_steps_validation():
    with pytest.raises(ValueError, match="checkpoint_steps"):
        V3Config.from_dict({"train": {"steps": 10, "checkpoint_steps": [5, 11]}})
    with pytest.raises(ValueError, match="checkpoint_steps"):
        V3Config.from_dict({"train": {"steps": 10, "checkpoint_steps": [5, 5]}})
    with pytest.raises(ValueError, match="diag_every"):
        V3Config.from_dict({"train": {"diag_every": -1}})


def test_pilot004_changes_only_duration_relative_to_pilot003():
    for arm in ("transition", "after", "b4"):
        old, _ = load_v3_config(f"configs/v3/pilot003_{arm}.yaml")
        new, _ = load_v3_config(f"configs/v3/pilot004_{arm}.yaml")
        a, b = old.to_dict(), new.to_dict()
        assert b["train"]["steps"] == 1000 and b["train"]["checkpoint_steps"] == [100, 300, 1000]
        assert b["train"]["diag_every"] == 10
        for key in ("steps", "checkpoint_steps", "diag_every"):
            a["train"].pop(key), b["train"].pop(key)
        assert a == b, arm  # same optimizer, warmup, budget, seed, data settings


def fake_pair(root: Path, tag: bytes):
    (root / "lora").mkdir(parents=True)
    (root / "modules.pt").write_bytes(b"modules" + tag)
    (root / "lora" / "adapter_config.json").write_text("{}")
    (root / "lora" / "adapter_model.safetensors").write_bytes(b"weights" + tag)


def test_completion_requires_and_pins_intermediate_checkpoints(tmp_path):
    config = {"train": {"steps": 3, "checkpoint_steps": [1, 3]}}
    out = tmp_path / "arm"
    fake_pair(out, b"final")
    (out / "metrics.jsonl").write_text("".join(json.dumps({"step": k}) + "\n" for k in (1, 2, 3)))
    fake_pair(out / "checkpoints" / "step-00001", b"1")
    with pytest.raises(FileNotFoundError):  # step-00003 snapshot missing: never marked complete
        write_completion(out, 3, config)
    fake_pair(out / "checkpoints" / "step-00003", b"3")
    write_completion(out, 3, config)
    assert training_complete(out, config, 3) == (True, "")
    record = json.loads((out / "complete.json").read_text())
    assert "checkpoints/step-00001/modules.pt" in record["artifacts"]
    (out / "checkpoints" / "step-00001" / "modules.pt").write_bytes(b"changed")
    assert not training_complete(out, config, 3)[0]
