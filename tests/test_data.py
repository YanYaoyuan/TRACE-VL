import numpy as np
import pytest
import torch

from trace_vl.config import load_config
from trace_vl.data.adapters import import_videomme
from trace_vl.data.dataset import VideoDataset, collate_samples
from trace_vl.data.schema import check_split_leakage, read_jsonl, validate_record
from trace_vl.data.video import sample_video


def test_manifest_split_and_no_source_leakage(fixture_manifest):
    rows = read_jsonl(fixture_manifest)
    assert check_split_leakage(rows) == {"train": 4, "val": 2, "test": 2}
    rows[1]["source_group"], rows[1]["split"] = rows[0]["source_group"], "test"
    with pytest.raises(ValueError, match="multiple splits"):
        check_split_leakage(rows)


def test_prefix_decoder_ignores_future_pixels(fixture_manifest):
    path = fixture_manifest.parent / "train_0000.npz"
    a, t, _ = sample_video(path, 0, 1.5, 2, 100, 32)
    with np.load(path) as z:
        frames, times = z["frames"].copy(), z["timestamps"]
    frames[times > 1.5] = 255
    np.savez_compressed(path, frames=frames, timestamps=times)
    b, tb, _ = sample_video(path, 0, 1.5, 2, 100, 32)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert t.max() <= 1.5
    torch.testing.assert_close(t, tb)


def test_video_cannot_be_relabelled_to_bypass_split_isolation(fixture_manifest):
    rows = read_jsonl(fixture_manifest)
    rows[1]["video"], rows[1]["split"] = rows[0]["video"], "test"
    with pytest.raises(ValueError, match="video .*multiple splits"):
        check_split_leakage(rows)


def test_real_video_decoder_cutoff(fixture_manifest):
    frames, times, cutoff = sample_video(fixture_manifest.parent / "train_0000.avi", 0, 1.25, 4, 100, 32)
    assert frames.shape[1:] == (3, 32, 32)
    assert times.tolist() == [0.0, 0.5, 1.0]
    assert cutoff == 1.25


def test_targets_and_collation(fixture_manifest):
    dataset = VideoDataset(fixture_manifest, "train")
    positive, negative = dataset[0], dataset[1]
    assert positive["alert_labels"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert positive["write_labels"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert negative["alert_labels"].sum() == 0
    for key in ("frames", "timestamps", "alert_labels", "write_labels", "utility_labels"):
        negative[key] = negative[key][:3]
    batch = collate_samples([positive, negative])
    assert not batch["frame_mask"][1, 3:].any()
    assert (batch["alert_labels"][1, 3:] == -1).all()


def test_invalid_schema_cutoffs_and_monitoring_inputs(fixture_manifest):
    row = read_jsonl(fixture_manifest)[0]
    row["observation"]["end"] = float("nan")
    with pytest.raises(ValueError):
        validate_record(row)
    row = read_jsonl(fixture_manifest)[0]
    del row["monitor_category"]
    with pytest.raises(ValueError, match="monitor_category"):
        validate_record(row)


def test_official_videomme_import_preserves_duration_class(tmp_path):
    row = {
        "videoID": "abc",
        "question_id": "001-1",
        "question": "What?",
        "options": ["A. yes", "B. no"],
        "answer": "A",
        "duration": "long",
    }
    output = tmp_path / "manifest.jsonl"
    assert import_videomme([row], output, tmp_path) == 1
    parsed = read_jsonl(output)[0]
    assert parsed["duration_group"] == "long"
    assert parsed["observation"]["end"] == "video_end"
    assert parsed["protocol"] == "offline"


def test_config_inheritance_and_cycles(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("model:\n  dim: 32\ntraining:\n  batch_size: 3\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\ntraining:\n  max_steps: 5\n")
    config = load_config(child)
    assert config["training"] == {"batch_size": 3, "max_steps": 5}
    assert config["model"]["dim"] == 32
    base.write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(child)
