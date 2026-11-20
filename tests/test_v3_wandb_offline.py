"""Real wandb SDK in offline mode (no network): artifact paths, multi-arm/multi-checkpoint upload, operator log,
init-failure fallback, guarded failures, offline collection for a later sync. Install with `--extra track`."""

import json
import warnings

import pytest

wandb = pytest.importorskip("wandb")

from trace_vl.v3.tracking import Tracker, offline_runs, sync_offline  # noqa: E402
from trace_vl.v3.wandb_upload import upload_run  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb-home"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_PROJECT", "trace-vl-test")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def pilot004_like(tmp_path):
    """Three arms, three checkpoints each: nine eval/<arm>/step-*/score.json with the same file name."""
    runs = tmp_path / "pilot004-a3f8c29-001"
    arms, steps = ("transition", "after", "b4"), (100, 300, 1000)
    for arm in arms:
        (runs / "train" / arm).mkdir(parents=True)
        (runs / "train" / arm / "metrics.jsonl").write_text(
            "".join(json.dumps({"step": k, "ce": 1.0 / k, "diag_emitted_rank": 2.0}) + "\n" for k in range(1, 21)))
        (runs / "train" / arm / "run.json").write_text("{}")
        for step in steps:
            out = runs / "eval" / arm / f"step-{step:05d}"
            out.mkdir(parents=True)
            (out / "score.json").write_text(json.dumps({"arm": arm, "step": step, "accuracy": 0.88}))
            (out / "run.json").write_text(json.dumps({"id": f"{arm}{step}"}))
    (runs / "eval" / "native").mkdir(parents=True)
    (runs / "eval" / "native" / "score.json").write_text(json.dumps({"accuracy": 0.8832}))
    (runs / "identity.json").write_text(json.dumps({"configs": {a: {"arm": a} for a in arms}, "train_sha256": "x"}))
    summary = {"per_checkpoint": {str(s): {"accuracy": {a: 0.88 for a in ("native", *arms)},
                                           "paired": {"transition_minus_b4": {"delta_pp": -2.0, "ci95_pp": [-4.3, 0.0]}}}
                                  for s in steps}, "decision": {"M2": False}}  # fmt: skip
    (runs / "summary.json").write_text(json.dumps(summary))
    (runs / "selector-diagnostics.json").write_text(json.dumps({a: {"diag_emitted_rank": [[1, 2.0], [10, 2.1]]} for a in arms}))
    (runs / "pilot004-digest.txt").write_text("digest")
    operator_log = tmp_path / "TRACE-VL" / "pilot-004.log"  # outside the run directory
    operator_log.parent.mkdir()
    operator_log.write_text("VERIFIED\n")
    return runs, operator_log


def test_upload_keeps_relative_paths_for_all_arms_and_checkpoints(tmp_path):
    runs, operator_log = pilot004_like(tmp_path)
    report = upload_run(runs, demos=0, extra_files=[str(operator_log), str(tmp_path / "missing.log")])
    entries = report["artifacts"]["reports"]
    scores = [e for e in entries if e.endswith("score.json")]
    assert len(scores) == 10 and len(set(scores)) == 10  # 3 arms x 3 checkpoints + native: no collisions
    assert "eval/transition/step-00100/score.json" in entries and "eval/b4/step-01000/score.json" in entries
    assert "operator/pilot-004.log" in entries and "summary.json" in entries and "pilot004-digest.txt" in entries
    for arm in ("transition", "after", "b4"):
        assert "train/" + arm + "/metrics.jsonl" in report["artifacts"]["train-" + arm]
    assert report["errors"] == ["extra file not found: " + str(tmp_path / "missing.log")]
    assert report["mode"] == "offline" and len(report["offline_runs"]) == 4  # 3 training replays + 1 eval run
    assert set(sync_offline(runs / "wandb", dry_run=True)) == set(report["offline_runs"])


def test_online_init_failure_falls_back_to_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "online")
    real_init = wandb.init

    def failing_online(*args, **kwargs):
        if kwargs.get("mode") == "online":
            raise RuntimeError("network unreachable")
        return real_init(*args, **kwargs)

    monkeypatch.setattr(wandb, "init", failing_online)
    with pytest.warns(UserWarning, match="collecting offline"):
        tracker = Tracker.start(True, "arm", "g", "train", directory=tmp_path / "w")
    assert tracker.active and tracker.mode == "offline"
    for step in range(1, 6):
        tracker.log({"ce": 1.0 / step}, step=step)
    tracker.finish()
    assert len(offline_runs(tmp_path / "w")) == 1


def test_failures_are_reported_once_and_never_raise(tmp_path):
    tracker = Tracker.start(True, "arm", "g", "train", directory=tmp_path / "w")
    a, b = tmp_path / "a" / "x.json", tmp_path / "b" / "x.json"
    for path in (a, b):
        path.parent.mkdir()
        path.write_text("{}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert tracker.upload("dup", "reports", [a, b], names={str(a): "x.json", str(b): "x.json"}) is None
    ours = [w for w in caught if "tracking disabled" in str(w.message)]
    assert "collision" in tracker.error and not tracker.active and len(ours) == 1
    tracker.log({"ce": 1.0}, step=1)  # disabled after the failure: silently ignored
    tracker.finish()


def test_training_style_logging_two_arms(tmp_path):
    for arm in ("transition", "b4"):
        tracker = Tracker.start(True, f"pilot-{arm}", "pilot", "train", config={"arm": arm}, directory=tmp_path / arm)
        for step in range(1, 51):
            tracker.log({"ce": 1.0 / step, "lr": 1e-4, "diag_gist_entropy_min": 0.99, "flag": True}, step=step)
        (tmp_path / arm / "metrics.jsonl").write_text("{}\n")
        artifact = tracker.upload(f"pilot-{arm}-logs", "training-logs", [tmp_path / arm / "metrics.jsonl"], base=tmp_path)
        assert sorted(artifact.manifest.entries) == [f"{arm}/metrics.jsonl"]
        tracker.finish()
    assert len(offline_runs(tmp_path)) == 2
