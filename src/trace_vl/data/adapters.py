"""Explicit public-annotation adapters; never create answers or timing labels that are not supplied."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import validate_record, write_jsonl


def import_videomme(rows, output, video_root, split="test", extension=".mp4"):
    root = Path(video_root).resolve()
    records = []
    for row in rows:
        video_id = str(row["videoID"])
        path = root / (video_id + extension)
        # Official rows contain duration classes, not seconds. Resolve duration from media at decoding.
        r = {
            "schema_version": 1,
            "id": "videomme:" + str(row["question_id"]),
            "split": split,
            "source": "Video-MME",
            "source_group": "videomme:" + video_id,
            "video": str(path),
            "protocol": "offline",
            "observation": {"start": 0, "end": "video_end"},
            "question": row["question"],
            "answer": str(row["answer"]).strip(),
            "answer_type": "choice",
            "options": list(row["options"]),
            "category": row.get("task_type", "unknown"),
            "duration_group": row.get("duration", "unknown"),
            "domain": row.get("domain", "unknown"),
            "annotation_protocol": "Video-MME visual-only, without subtitles/audio; full-video cutoff",
        }
        records.append(validate_record(r))
    write_jsonl(output, records)
    return len(records)


def import_alert_csv(csv_path, output, video_root, split="train"):
    """One row per episode/question, events_json holds independently annotated event records."""
    rows = []
    with Path(csv_path).open(newline="") as f:
        for x in csv.DictReader(f):
            r = {
                "schema_version": 1,
                "id": x["id"],
                "split": x.get("split") or split,
                "source": x.get("source") or "custom",
                "source_group": x["source_group"],
                "video": str((Path(video_root) / x["video"]).resolve()),
                "protocol": "streaming",
                "observation": {"start": float(x["start"]), "end": float(x["observed_until"])},
                "rule": x["rule"],
                "monitor_category": x["monitor_category"],
                "question": x["question"],
                "answer": x["answer"],
                "answer_type": "text",
                "events": json.loads(x["events_json"] or "[]"),
            }
            rows.append(validate_record(r))
    write_jsonl(output, rows)
    return len(rows)


def read_annotation_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError("expected a JSON list of official Video-MME rows")
        return value
    if path.suffix == ".parquet":
        try:
            from datasets import Dataset
        except ImportError as e:
            raise ImportError("Install trace-vl[data] to read parquet annotations") from e
        return list(Dataset.from_parquet(str(path)))
    raise ValueError("supported annotation formats: json, jsonl, parquet")
