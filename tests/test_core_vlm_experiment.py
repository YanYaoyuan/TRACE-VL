import json
import os
import sys

import pytest

from trace_vl.core_vlm.experiment import (
    ARMS,
    paired,
    report_experiment,
    run_jobs,
    training_command,
    validate_config,
)


def test_training_arms_share_recipe_data_and_sample_budget():
    cfg = json.loads(open("configs/core_vlm/baselines_a100.json").read())
    validate_config(cfg)
    full = training_command(cfg, "same.jsonl", "same", "full")
    grid = training_command(cfg, "same.jsonl", "same", "grid")
    assert [(a, b) for a, b in zip(full, grid, strict=True) if a != b] == [("full", "grid")]
    assert "--nproc_per_node=4" in full
    assert full[full.index("--updates") + 1] == "1000"
    flow = training_command(cfg, "same.jsonl", "flow", "grid", flow=True)
    assert flow[flow.index("--updates") + 1] == "2"


def test_source_cluster_interval_does_not_treat_each_question_as_independent():
    rows = [{"id": str(i), "source_group": "same", "video": "v", "question": "Q",
             "options": ["a", "b"], "answer": 0} for i in range(100)]
    left = {r["id"]: {"answer": "A"} for r in rows}
    right = {r["id"]: {"answer": "B"} for r in rows}
    result = paired(rows, left, right)
    assert result["delta_pp"] == 100 and result["source_videos"] == 1
    assert result["ci95_pp"] is None


def test_queue_stops_on_failed_child_and_retains_error_log(tmp_path):
    with pytest.raises(RuntimeError, match="exited 3"):
        run_jobs([("fail", [sys.executable, "-c", "print('specific failure'); raise SystemExit(3)"], [])],
                 os.environ.copy(), tmp_path)
    assert "specific failure" in (tmp_path / "fail.log").read_text()


def test_final_comparison_rejects_unmatched_pixels_even_with_same_question_ids(tmp_path):
    row = {"id": "q", "question": "Q", "video": "v", "source_group": "v", "split": "val",
           "options": ["a", "b"], "answer": 0}
    manifest = tmp_path / "qa.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    observation = {"video_sha256": "v", "frame_indices": [0, 1], "decoded_rgb_sha256": "pixels",
                   "input_ids_sha256": "prompt", "timestamp_policy": "native-index-fps",
                   "timestamp_text_seconds": [0.]}
    for arm in ARMS:
        directory = tmp_path / "eval" / arm
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps({"id": arm}))
        (directory / "predictions.jsonl").write_text(json.dumps({"run": arm, "id": "q", "answer": "A",
                                                               "observation": observation}) + "\n")
    result = report_experiment(manifest, tmp_path, 0)
    assert result["completed"]
    path = tmp_path / "eval" / "trained_grid" / "predictions.jsonl"
    prediction = json.loads(path.read_text())
    prediction["observation"]["decoded_rgb_sha256"] = "different pixels"
    path.write_text(json.dumps(prediction) + "\n")
    with pytest.raises(ValueError, match="unmatched observation"):
        report_experiment(manifest, tmp_path, 0)
