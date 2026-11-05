import json

from trace_vl.config import ModelConfig
from trace_vl.data.schema import read_jsonl, write_jsonl
from trace_vl.inference import evaluate
from trace_vl.training import train


def test_video_state_reuse_preserves_answers_and_evidence(fixture_manifest, tmp_path):
    rows = []
    for row in read_jsonl(fixture_manifest):
        if row["split"] == "val":
            row = {k: v for k, v in row.items() if k not in {"events", "rule", "monitor_category"}}
            rows.extend([row, {**row, "id": row["id"] + ":q2", "question": "What happened?"}])
        else:
            rows.append(row)
    write_jsonl(fixture_manifest, rows)
    config = {
        "model": ModelConfig(normalize_memory=True).to_dict(),
        "data": {"manifest": str(fixture_manifest), "max_frames": 2},
        "training": {"device": "cpu", "max_steps": 1, "batch_size": 2},
        "evaluation": {"device": "cpu", "max_new_tokens": 2},
    }
    checkpoint = train(config, tmp_path / "train")
    first = evaluate(config, checkpoint, tmp_path / "uncached", split="val")
    config["evaluation"]["reuse_video_state"] = True
    second = evaluate(config, checkpoint, tmp_path / "cached", split="val")
    assert first["provenance"]["observed_prefixes"] == 4
    assert second["provenance"]["observed_prefixes"] == 2
    uncached = [
        json.loads(line) for line in (tmp_path / "uncached/predictions.jsonl").read_text().splitlines()
    ]
    cached = [json.loads(line) for line in (tmp_path / "cached/predictions.jsonl").read_text().splitlines()]
    keys = ("id", "answer", "observed_until", "evidence")
    assert [{k: r[k] for k in keys} for r in uncached] == [{k: r[k] for k in keys} for r in cached]
    assert sum(r["observation_reused"] for r in cached) == 2
