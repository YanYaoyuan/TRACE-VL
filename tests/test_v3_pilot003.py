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
    assert [i.is_i_step for i in a] == [True, False, False]  # I-step first, then P-steps (scene cuts off)
    p_a, p_b = [i for i in a if not i.is_i_step], [i for i in b if not i.is_i_step]
    assert [i.aux["active_centroids"] for i in p_a] == [i.aux["active_centroids"] for i in p_b]  # deterministic
    evaluated = [i for i in run("transition", train=False)[0] if not i.is_i_step]
    for train_info, eval_info in zip(p_a, evaluated):  # train == eval: no sampling, no straight-through scaling
        assert torch.allclose(train_info.tokens, eval_info.tokens)


def test_after_tokens_are_norm_matched_real_token_pools():
    (infos, adapted) = run("after")
    info = infos[-1]
    ref = adapted.norm(dim=-1).mean()
    assert torch.allclose(info.tokens.norm(dim=-1), torch.full_like(info.tokens[:, 0], float(ref)), rtol=1e-4)
    assert info.rows.max() <= 2 and info.cols.max() <= 3 and info.rows.min() >= 0


def test_slot_part_pools_partition_and_fallback():
    x = torch.eye(6)
    rows, cols = torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 1, 0, 1, 0, 1])
    # Definition: a token joins part floor(3 x weight ranked above it), i.e. the part in which its weight starts.
    a = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10, 0.0])  # above: 0, .30, .55, .75, .90, 1.0 -> parts 0,0,1,2,2,2
    pools, r, c = slot_part_pools(a, x, rows, cols)
    assert pools.shape == (3, 6) and torch.allclose(pools.sum(-1), torch.ones(3))  # convex combinations
    assert torch.allclose(pools[0], torch.tensor([0.30, 0.25, 0, 0, 0, 0]) / 0.55, atol=1e-6)
    assert torch.allclose(pools[1], x[2])
    assert torch.allclose(pools[2], torch.tensor([0, 0, 0, 0.15, 0.10, 0]) / 0.25, atol=1e-6)
    assert r.tolist() == [0.0, 1.0, 1.0] and c.tolist() == [0.0, 0.0, 1.0]  # weighted centroids, rounded
    dominant = torch.tensor([0.9, 0.05, 0.05, 0, 0, 0])  # above: 0, .9, .95 -> part 1 is empty
    pools, _, _ = slot_part_pools(dominant, x, rows, cols)
    assert torch.allclose(pools[0], x[0]) and torch.isfinite(pools).all()
    assert torch.count_nonzero(pools[1]) == 1 and int(pools[1].argmax()) in (1, 2)  # fallback: next-ranked token


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


# ---------------------------------------------------------------------------- runner identity / completion (review)
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

from trace_vl.v3.pilot import preflight, training_complete, write_completion  # noqa: E402

CONFIG = {"train": {"steps": 3}, "x": 1}


def fake_training(out: Path, steps=3, weights=True):
    (out / "lora").mkdir(parents=True)
    (out / "modules.pt").write_bytes(b"modules")
    (out / "lora" / "adapter_config.json").write_text("{}")
    if weights:
        (out / "lora" / "adapter_model.safetensors").write_bytes(b"weights")
    (out / "metrics.jsonl").write_text("".join(json.dumps({"step": k}) + "\n" for k in range(1, steps + 1)))


def test_completion_record_is_required_and_pins_artifacts(tmp_path):
    out = tmp_path / "arm"
    fake_training(out)
    # Interrupted during/after the final save: metrics and files exist but no completion record.
    assert training_complete(out, CONFIG, 3) == (False, "no completion record (interrupted or never finished)")
    write_completion(out, 3, CONFIG)
    assert training_complete(out, CONFIG, 3) == (True, "")
    assert not training_complete(out, {**CONFIG, "x": 2}, 3)[0]  # different resolved config
    assert not training_complete(out, CONFIG, 4)[0]
    (out / "modules.pt").write_bytes(b"truncated")  # artifact changed after completion
    assert "differ" in training_complete(out, CONFIG, 3)[1]


def test_missing_adapter_weights_or_metric_gaps_are_incomplete(tmp_path):
    out = tmp_path / "noweights"
    fake_training(out, weights=False)
    with pytest.raises(FileNotFoundError):
        write_completion(out, 3, CONFIG)
    out2 = tmp_path / "gaps"
    fake_training(out2)
    write_completion(out2, 3, CONFIG)
    (out2 / "metrics.jsonl").write_text("".join(json.dumps({"step": k}) + "\n" for k in (1, 3, 3)))
    assert "metric log" in training_complete(out2, CONFIG, 3)[1]
    shutil.rmtree(out2 / "lora")
    assert not training_complete(out2, CONFIG, 3)[0]


def pilot_inputs(tmp_path):
    data, media, configs = tmp_path / "data", tmp_path / "media", tmp_path / "configs"
    data.mkdir()
    media.mkdir()
    shutil.copytree(Path("configs/v3"), configs)
    for split, videos in (("train", ["a.mp4", "b.mp4"]), ("val", ["c.mp4"])):
        rows = [{"id": f"{split}{i}", "video": v, "split": split} for i, v in enumerate(videos)]
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for v in videos:
            (media / v).write_bytes(b"video-" + v.encode())
    return data, media, configs


def run_preflight(tmp_path, data, media, configs, runs="runs"):
    return preflight(data, media, tmp_path / runs, configs, None, 30.0, 4, Path("src/trace_vl/v3"))


def test_restart_with_changed_inherited_config_or_media_is_rejected(tmp_path):
    data, media, configs = pilot_inputs(tmp_path)
    assert run_preflight(tmp_path, data, media, configs)["val_questions"] == 1
    run_preflight(tmp_path, data, media, configs)  # unchanged restart passes
    base = configs / "pilot003_base.yaml"
    original = base.read_text()
    base.write_text(original.replace("train:\n", "train:\n  lr: 0.0009\n"))  # inherited learning rate
    with pytest.raises(ValueError, match="configs"):
        run_preflight(tmp_path, data, media, configs)
    base.write_text(original.replace("train:\n", "train:\n  seed: 11\n"))  # inherited seed
    with pytest.raises(ValueError, match="configs"):
        run_preflight(tmp_path, data, media, configs)
    base.write_text(original)
    run_preflight(tmp_path, data, media, configs)
    (media / "c.mp4").write_bytes(b"replaced video with other bytes")  # media replaced in place
    with pytest.raises(ValueError, match="media"):
        run_preflight(tmp_path, data, media, configs)


def test_preflight_rejects_arm_configs_differing_beyond_mode(tmp_path):
    data, media, configs = pilot_inputs(tmp_path)
    after = configs / "pilot003_after.yaml"
    after.write_text(after.read_text() + "  slots: 6\n")
    with pytest.raises(ValueError, match="differ beyond"):
        run_preflight(tmp_path, data, media, configs)
