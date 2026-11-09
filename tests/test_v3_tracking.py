"""wandb tracking, demos and run upload, tested with a stub wandb module (no network, no package needed)."""

import json
import sys
import types

import numpy as np
import pytest
import torch

from trace_vl.v3.tracking import Tracker, enabled_from_env


class FakeRun:
    def __init__(self, **kwargs):
        self.kwargs, self.logged, self.summary, self.artifacts, self.url = kwargs, [], {}, [], "https://wandb.example/run"

    def log(self, values, step=None):
        self.logged.append((step, values))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished = True


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name, self.type, self.files = name, type, []

    def add_file(self, path, name=None):
        self.files.append(name)

    def add_dir(self, path, name=None):
        self.files.append(name + "/")


class FakeTable:
    def __init__(self, columns, data=None):
        self.columns, self.data = columns, list(data or [])

    def add_data(self, *row):
        self.data.append(row)


@pytest.fixture
def fake_wandb(monkeypatch):
    runs = []
    module = types.SimpleNamespace(
        init=lambda **kw: runs.append(FakeRun(**kw)) or runs[-1], Artifact=FakeArtifact, Table=FakeTable,
        Video=lambda path: ("video", path), Image=lambda path: ("image", path))  # fmt: skip
    monkeypatch.setitem(sys.modules, "wandb", module)
    return runs


def test_disabled_tracker_is_a_no_op(monkeypatch):
    monkeypatch.delenv("TRACE_WANDB", raising=False)
    assert not enabled_from_env()
    t = Tracker.start(False, "x", None, "train")
    t.log({"a": 1.0})
    t.upload("x", "y", ["/nonexistent"])
    t.finish()
    assert not t.active


def test_missing_wandb_warns_and_disables(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import raises ImportError
    with pytest.warns(UserWarning, match="not installed"):
        assert not Tracker.start(True, "x", None, "train").active


def test_tracker_logs_finite_numbers_only(fake_wandb, tmp_path):
    t = Tracker.start(True, "arm", "group", "train", config={"a": 1}, directory=tmp_path / "w")
    t.log({"ce": 0.5, "flag": True, "nan": float("nan"), "text": "x", "step_tokens": 20}, step=3)
    assert fake_wandb[0].logged == [(3, {"ce": 0.5, "step_tokens": 20})]
    assert fake_wandb[0].kwargs["group"] == "group" and fake_wandb[0].kwargs["job_type"] == "train"
    (tmp_path / "m.jsonl").write_text("{}")
    t.upload("pilot/004 logs", "training-logs", [tmp_path / "m.jsonl", tmp_path / "missing"])
    assert fake_wandb[0].artifacts[0].name == "pilot-004-logs" and fake_wandb[0].artifacts[0].files == ["m.jsonl"]


def build_run(tmp_path):
    runs = tmp_path / "pilot004-x"
    (runs / "train" / "transition").mkdir(parents=True)
    (runs / "train" / "b4.incomplete-1").mkdir(parents=True)
    (runs / "train" / "transition" / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": k, "ce": 1.0 / k, "lr": 1e-4}) + "\n" for k in (1, 2, 3)))
    (runs / "identity.json").write_text(json.dumps({"configs": {"transition": {"x": 1}}, "train_sha256": "abc"}))
    summary = {"per_checkpoint": {"100": {"accuracy": {"b4": 0.88, "transition": 0.85},
                                          "ci95": {"b4": [0.85, 0.91], "transition": [0.82, 0.88]},
                                          "paired": {"transition_minus_b4": {"delta_pp": -2.7, "ci95_pp": [-4.5, -1.0]}}}},
               "decision": {"M2": False}, "note": "x"}  # fmt: skip
    (runs / "summary.json").write_text(json.dumps(summary))
    (runs / "selector-diagnostics.json").write_text(json.dumps({"transition": {"diag_gist_entropy_min": [[1, 1.0], [10, 0.9]]}}))
    (runs / "pilot004-digest.txt").write_text("digest")
    return runs


def test_upload_run_replays_training_and_reports(fake_wandb, tmp_path):
    from trace_vl.v3.wandb_upload import upload_run

    report = upload_run(build_run(tmp_path))
    assert report["train_runs"] == ["transition"]  # incomplete attempts are skipped
    train, evaluation = fake_wandb
    assert [step for step, _ in train.logged] == [1, 2, 3] and train.kwargs["name"] == "pilot004-x-train-transition"
    assert train.summary["updates"] == 3
    assert evaluation.summary["per_checkpoint/100/accuracy/b4"] == 0.88 and evaluation.summary["decision/M2"] is False
    tables = {k: v for _, values in evaluation.logged for k, v in values.items()}
    assert tables["accuracy"].data[0][:3] == ["100", "b4", 0.88]
    assert tables["paired_differences_pp"].data == [["100", "transition_minus_b4", -2.7, -4.5, -1.0]]
    assert tables["selector_diagnostics"].data[1] == ["transition", "diag_gist_entropy_min", 10, 0.9]
    names = evaluation.artifacts[0].files
    assert "summary.json" in names and "pilot004-digest.txt" in names and report["eval_run"].startswith("https://")


def test_overlay_gif_strip_and_question_selection(tmp_path):
    from PIL import Image

    from trace_vl.v3.demo import overlay, select_questions, write_gif, write_strip

    frame = np.full((96, 128, 3), 100, dtype=np.uint8)
    weights = np.zeros(12)
    weights[5] = 1.0
    hot = overlay(frame, weights, (3, 4), "t=0.5s P-step 20 tok")
    plain = overlay(frame, None, (3, 4), "I-step")
    assert hot.shape == plain.shape == (240, 320, 3) and hot[20:, :, 0].max() > plain[20:, :, 0].max()  # below the label bar
    write_gif([hot, plain, hot], tmp_path / "a.gif")
    write_strip([hot, plain, hot], tmp_path / "a.png")
    assert Image.open(tmp_path / "a.gif").n_frames == 3 and Image.open(tmp_path / "a.png").size == (960, 240)
    records = [{"id": f"q{i}", "split": "val", "options": ["a", "b"], "answer": 0} for i in range(30)]
    assert [r["id"] for r in select_questions(records, 5)] == [r["id"] for r in select_questions(list(reversed(records)), 5)]


def test_weight_map_is_a_distribution_over_the_grid():
    from trace_vl.v3.config import V3Config
    from trace_vl.v3.model import TraceV3Modules
    from trace_vl.v3.modules import CSTCache, TECache

    for mode in ("transition", "after", "pooled"):
        cfg = V3Config.from_dict(
            {"cst": {"dim": 32, "layers": 2, "heads": 4, "window": 3},
             "transition": {"slots": 4, "gist": 2, "max_active": 2, "before_lag": 1, "i_step_every": 100,
                            "scene_cut_surprise": 1.01, "fixed_active": 2, "p_step_mode": mode},
             "train": {"target_p_step_tokens": 8}})  # fmt: skip
        torch.manual_seed(5)
        modules = TraceV3Modules(cfg, model_dim=16).eval()
        modules.transition.diagnostics = True
        cst, te = CSTCache(), TECache()
        for step in range(3):
            adapted, z = modules.cst(torch.rand(12, 16) + 0.5, (3, 4), float(step), cst)
            info = modules.transition(adapted, z, (3, 4), te)
        w = info.aux["weight_map"]
        assert w.shape == (12,) and float(w.sum()) == pytest.approx(1.0, abs=1e-5) and float(w.min()) >= 0


def test_log_demos_builds_one_table_per_arm(fake_wandb, tmp_path):
    from trace_vl.v3.demo import log_demos

    tracker = Tracker.start(True, "eval", "g", "eval")
    rows = [{"id": "q1", "gif": "a.gif", "strip": "a.png", "question": "Q?", "options": ["x", "y"], "gold": 0,
             "prediction": 1, "correct": False, "probabilities": [0.4, 0.6], "tokens_per_step": 46.9}]  # fmt: skip
    log_demos(tracker, rows, "transition")
    table = fake_wandb[0].logged[0][1]["demos/transition"]
    assert table.data[0][2] == ("video", "a.gif") and table.data[0][6:8] == ("A", "B")
