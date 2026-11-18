import argparse
import hashlib
import json
import types

import pytest


def test_report_entry_logs_choice_and_strict_scores(tmp_path, monkeypatch):
    from trace_vl.core_vlm import evaluate

    video = tmp_path / "v.mp4"
    video.write_bytes(b"dummy")
    row = {"id": "q", "video": str(video), "question": "Q?", "options": ["x", "y"], "answer": 0}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    directory = tmp_path / "experiment/eval/ovo/plain"
    calls = {}
    tracker = types.SimpleNamespace(log=lambda x: calls.update(metrics=x), summary=lambda x: None,
        upload=lambda *a, **k: calls.update(artifact=a[2]), finish=lambda: calls.update(finished=True))
    monkeypatch.setattr(evaluate.Tracker, "start", lambda *a, **k: tracker)
    monkeypatch.setattr(evaluate, "checkpoint_identity", lambda *a: {"fixed": "model"})
    monkeypatch.setattr(evaluate, "read_split", lambda *a: [row])

    def save_identity(path, identity):
        path.write_text(json.dumps(identity))
        units = directory / "questions"
        units.mkdir()
        (units / (hashlib.sha256(b"q").hexdigest() + ".json")).write_text(
            json.dumps({"id": "q", "run": identity["id"], "answer": "A", "choice": 0}))

    monkeypatch.setattr(evaluate, "ensure_run_manifest", save_identity)
    args = argparse.Namespace(shards=1, shard=0, max_new_tokens=4, manifest=str(manifest), split="val", limit=0,
        out=str(directory), model="unused", checkpoint=None, mode="grid", pool_grid=[1, 1], report_only=True)
    evaluate.run(args)
    assert calls["metrics"]["eval/choice_accuracy"] == 1
    assert calls["metrics"]["eval/strict_accuracy"] == 1
    assert calls["finished"] and len(calls["artifact"]) == 3
    assert (directory / "report.json").is_file()


def test_real_wandb_offline_records_curves_and_artifact(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    from trace_vl.v3.tracking import Tracker
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_PROJECT", "trace-core-regression")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache/wandb"))
    tracker = Tracker.start(True, "batch4", "test", "core-train", config={"microbatch_size": 4}, directory=tmp_path)
    assert tracker.active and tracker.mode == "offline"
    tracker.log({"loss": 1., "samples_per_second": 3., "peak_allocated_gib_max": 12.}, step=1)
    report = tmp_path / "report.json"
    report.write_text('{"completed": true}')
    artifact = tracker.upload("core-test-report", "core-evaluation", [report], base=tmp_path)
    assert artifact is not None and "report.json" in artifact.manifest.entries
    tracker.finish()
    assert list(tmp_path.rglob("run-*.wandb"))
