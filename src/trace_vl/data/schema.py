"""Normalized records explicitly declare the visible video prefix."""

from __future__ import annotations

import json
import math
from pathlib import Path


def finite_number(value, name, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def validate_record(record: dict) -> dict:
    r = dict(record)
    if r.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    for name in ("id", "video", "question", "answer", "source_group"):
        if not isinstance(r.get(name), str) or not r[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if r.get("split") not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if r.get("protocol", "streaming") not in {"streaming", "offline"}:
        raise ValueError("protocol must be streaming or offline")
    obs = r.get("observation")
    if not isinstance(obs, dict):
        raise ValueError("observation.start/end is required")
    start = finite_number(obs.get("start"), "observation.start")
    end = obs.get("end")
    if end == "video_end":
        if r.get("protocol") != "offline":
            raise ValueError("streaming records must have an explicit numeric observation.end")
    else:
        end = finite_number(end, "observation.end")
        if end < start:
            raise ValueError("observation.end precedes start")
    r["observation"] = {"start": start, "end": end}
    if r.get("answer_type", "text") not in {"text", "choice"}:
        raise ValueError("answer_type must be text or choice")
    if r.get("answer_type") == "choice":
        options = r.get("options")
        if (
            not isinstance(options, list)
            or not 2 <= len(options) <= 26
            or not all(isinstance(x, str) for x in options)
        ):
            raise ValueError("choice options must be a list of 2..26 strings")
        if r["answer"] not in [chr(65 + i) for i in range(len(options))]:
            raise ValueError("choice answer is outside options")
    if "events" in r:
        if not isinstance(r["events"], list):
            raise ValueError("events must be a list, including [] for verified negatives")
        if not r.get("rule") or not isinstance(r.get("monitor_category"), str) or not r["monitor_category"]:
            raise ValueError("monitoring episodes require rule and monitor_category, including negatives")
    event_ids = set()
    for event in r.get("events", []):
        for name in ("id", "category"):
            if not isinstance(event.get(name), str) or not event[name]:
                raise ValueError(f"event.{name} is required")
        if event["id"] in event_ids:
            raise ValueError("duplicate event id within an episode")
        event_ids.add(event["id"])
        a = finite_number(event.get("start"), "event.start")
        b = finite_number(event.get("end"), "event.end")
        known = finite_number(event.get("known_at"), "event.known_at")
        deadline = finite_number(event.get("deadline"), "event.deadline")
        if b < a or known < a or deadline < known:
            raise ValueError("event interval, known_at, or deadline is inconsistent")
        if event["category"] != r["monitor_category"]:
            raise ValueError("one monitoring rule/category per episode is required")
    if "utility_targets" in r:
        for point in r["utility_targets"]:
            finite_number(point.get("time"), "utility time")
            v = finite_number(point.get("value"), "utility value")
            if v > 1:
                raise ValueError("utility targets must be in [0,1]")
    return r


def read_jsonl(path):
    records, ids = [], set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = validate_record(json.loads(line))
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"{path}:{line_number}: {e}") from e
        if r["id"] in ids:
            raise ValueError(f"duplicate sample id: {r['id']}")
        ids.add(r["id"])
        records.append(r)
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def check_split_leakage(records):
    seen, videos = {}, {}
    for r in records:
        group = r["source_group"]
        if group in seen and seen[group] != r["split"]:
            raise ValueError(f"source_group {group!r} appears in multiple splits")
        seen[group] = r["split"]
        video = str(Path(r["video"]))
        if video in videos and videos[video] != r["split"]:
            raise ValueError(
                f"video {video!r} appears in multiple splits, even with different source_group labels"
            )
        videos[video] = r["split"]
    return {split: sum(r["split"] == split for r in records) for split in ("train", "val", "test")}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
