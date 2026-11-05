import importlib.util
from pathlib import Path

from trace_vl.config import ModelConfig
from trace_vl.data.schema import read_jsonl, write_jsonl
from trace_vl.training import train

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_visual_dependency.py"
spec = importlib.util.spec_from_file_location("visual_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_all_question_diagnostic_keeps_targets_and_changes_source_video(fixture_manifest, tmp_path):
    rows = read_jsonl(fixture_manifest)
    rows.extend(
        {**row, "id": row["id"] + ":another", "question": "What changes later?"}
        for row in list(rows)
        if row["split"] == "val"
    )
    rows.sort(key=lambda row: (row["split"], row["source_group"]))
    write_jsonl(fixture_manifest, rows)
    config = {
        "model": ModelConfig().to_dict(),
        "data": {"manifest": str(fixture_manifest), "max_frames": 2},
        "training": {"device": "cpu", "max_steps": 1, "batch_size": 2},
        "evaluation": {"device": "cpu", "max_new_tokens": 2, "reuse_video_state": True},
    }
    checkpoint = train(config, tmp_path / "train")
    report = diagnostic.diagnose(
        config,
        checkpoint,
        tmp_path / "diagnostic.json",
        max_videos=2,
        all_questions=True,
        generate=True,
    )
    truth = {row["id"]: row for row in rows if row["split"] == "val"}
    assert report["videos"] == 2 and report["questions"] == 4
    assert report["observed_prefix_pairs"] == 2
    assert {row["id"] for row in report["results"]} == set(truth)
    for result in report["results"]:
        assert result["question"] == truth[result["id"]]["question"]
        assert result["answer"] == truth[result["id"]]["answer"]
        assert result["video"] != result["donor_video"]
        assert isinstance(result["matched_prediction"], str)
        assert isinstance(result["mismatched_prediction"], str)
    assert 0 <= report["generation"]["fraction_answer_changed"] <= 1
