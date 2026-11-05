import json
from pathlib import Path

import pytest

from trace_vl.cli import main
from trace_vl.config import ModelConfig, load_config
from trace_vl.data.dataset import VideoDataset, collate_samples
from trace_vl.data.llava import import_llava_video
from trace_vl.data.schema import check_split_leakage, read_jsonl
from trace_vl.models.model import TraceVL


def annotation_row(video, index):
    return {
        "id": str(index),
        "video": video,
        "conversations": [
            {"from": "human", "value": "<image>\nWhat happened?"},
            {"from": "gpt", "value": "A person entered."},
            {"from": "human", "value": "What happened next?"},
            {"from": "gpt", "value": "They left."},
        ],
    }


def test_llava_video_disjoint_split_and_dialogue_context(tmp_path):
    rows = []
    for index in range(5):
        (tmp_path / f"{index}.mp4").touch()
        rows.extend([annotation_row(f"{index}.mp4", index)] * 2)
    rows.append(annotation_row("missing.mp4", 6))
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps(rows))
    output = tmp_path / "manifest.jsonl"
    report = import_llava_video([source], output, tmp_path, max_videos=4, val_fraction=0.25)
    prepared = read_jsonl(output)
    assert report["missing_media_rows"] == 1
    assert report["available_videos"] == 5
    assert report["train_videos"] == 3 and report["val_videos"] == 1
    assert check_split_leakage(prepared) == {"train": 12, "val": 4, "test": 0}
    assert len({r["id"] for r in prepared}) == 16
    first, second = prepared[:2]
    assert first["question"] == "What happened?"
    assert first["answer"] == "A person entered."
    assert "A person entered." in second["question"]
    assert "They left." not in second["question"]
    assert all("events" not in r and r["protocol"] == "offline" for r in prepared)
    assert json.loads(Path(str(output) + ".report.json").read_text()) == report
    # Reordering annotation rows must not change which videos belong to each split.
    source.write_text(json.dumps(list(reversed(rows))))
    other = tmp_path / "other.jsonl"
    import_llava_video([source], other, tmp_path, max_videos=4, val_fraction=0.25)
    assert {r["video"]: r["split"] for r in prepared} == {r["video"]: r["split"] for r in read_jsonl(other)}


def test_llava_rejects_bad_paths_and_missing_media(tmp_path):
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps([annotation_row("../outside.mp4", 0)]))
    output = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match="within video_root"):
        import_llava_video([source], output, tmp_path)
    source.write_text(json.dumps([annotation_row("not_here.mp4", 0)]))
    with pytest.raises(ValueError, match="at least two"):
        import_llava_video([source], output, tmp_path)
    assert not output.exists()


def test_llava_cli_decoding_and_answer_only_gradients(fixture_manifest, tmp_path, capsys):
    root = fixture_manifest.parent
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps([annotation_row(f"train_{index:04d}.avi", index) for index in range(4)]))
    output = tmp_path / "pilot.jsonl"
    main(["data", "import-llava", str(source), "--video-root", str(root), "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["selected_videos"] == 4
    dataset = VideoDataset(output, "train", max_frames=4)
    sample = dataset[0]
    assert sample["frames"].shape == (4, 3, 32, 32)
    assert (sample["alert_labels"] == -1).all()
    model = TraceVL(ModelConfig(text_limit=256))
    result = model(collate_samples([sample]), {"alert": 0, "write": 0, "utility": 0})
    result["loss"].backward()
    assert set(result["losses"]) == {"answer"}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.blocks.parameters())


def test_local_pilot_configuration_keeps_reference_architecture():
    project = Path(__file__).resolve().parents[1]
    pilot = load_config(project / "configs/local_2b_pilot.yaml")
    reference = load_config(project / "configs/reference_2b.yaml")
    assert pilot["model"]["freeze_vision"] and pilot["model"]["freeze_language"]
    assert {k: v for k, v in pilot["model"].items() if k != "text_limit"} == {
        k: v for k, v in reference["model"].items() if k != "text_limit"
    }
    assert pilot["training"]["precision"] == "bf16"
    assert pilot["data"]["max_frames"] == 4
    assert pilot["training"]["loss_weights"]["alert"] == 0
