"""Duration rules ("object left for at least N minutes") replayed causally from transition logs.

Ground truth comes from linked MEVA put-down/pick-up intervals. The detector also fires on things the
annotations do not cover (a person sitting still, a chair moved), so precision against annotated
put-downs is a lower bound; optional VLM verification is the intended fix for those false alarms.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .evidence import fit
from .meva import _same_place


def predicted_alerts(log, min_duration, zone=None):
    """One alert per region transition that stays changed for `min_duration`, raised causally."""
    state = {}
    for record in log:
        if record["kind"] != "region":
            continue
        item = state.setdefault(record["id"], dict(record))
        item.update({k: record[k] for k in ("t_end", "t_end_known", "status", "crops")})
    alerts = []
    for item in state.values():
        if zone is not None and not _same_place(item["box"], zone):
            continue
        due = item["t_start"] + min_duration
        ended = item["t_end"] if item["t_end"] is not None else float("inf")
        if ended < due:
            continue
        alerts.append({**item, "alert_time": max(due, item["t_known"])})
    return alerts


def verify(alerts, root, vlm, unit=32, crop_tokens=128, threshold=0.5):
    """Keep alerts where the VLM sees an object that was left behind (before vs after crop)."""
    import math

    kept = []
    for alert in alerts:
        if "before" not in alert["crops"] or "after" not in alert["crops"]:
            continue

        def load(name):
            image = Image.open(Path(root) / alert["crops"][name]).convert("RGB")
            return fit(image, crop_tokens, unit)

        segments = [
            {"type": "text", "text": "Two crops of the same place from a fixed security camera."},
            {"type": "text", "text": "Before:"},
            {"type": "image", "image": load("before")},
            {"type": "text", "text": "After:"},
            {"type": "image", "image": load("after")},
            {"type": "text", "text": "Was an object (bag, box, item) left behind in the After crop that is not in "
             "the Before crop? A person standing or sitting does not count."},  # fmt: skip
            {"type": "text", "text": "A. Yes\nB. No\nAnswer with the option letter only."},
        ]
        _, scores, _ = vlm.choose(segments, 2)
        p_yes = math.exp(scores[0]) / (math.exp(scores[0]) + math.exp(scores[1]))
        if p_yes >= threshold:
            kept.append({**alert, "p_yes": round(p_yes, 4)})
    return kept


def score_alerts(events_path, root, min_duration=120.0, tolerance=60.0, split_streams=None, vlm=None, unit=32):
    events = json.loads(Path(events_path).read_text())
    root = Path(root)
    totals = {"tp": 0, "fp": 0, "fn": 0, "latency": []}
    per_stream = {}
    for stream_id, data in events.items():
        if split_streams is not None and stream_id not in split_streams:
            continue
        log_path = root / "streams" / stream_id / "transitions.jsonl"
        if not log_path.exists():
            continue
        log = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
        alerts = predicted_alerts(log, min_duration)
        if vlm is not None:
            alerts = verify(alerts, root, vlm, unit)
        truth = [
            {"box": r["box"], "due": r["start"] + min_duration}
            for r in data["rests"]
            if r["end"] - r["start"] >= min_duration and not r["censored"]
        ]
        # Objects whose pick-up was never linked have unknown labels: alerts on them are ignored.
        unknown = [r for r in data["rests"] if r["censored"]]
        alerts = [
            a
            for a in alerts
            if not any(_same_place(a["box"], r["box"]) and a["alert_time"] >= r["start"] for r in unknown)
        ]
        matched, tp = set(), 0
        for alert in sorted(alerts, key=lambda a: a["alert_time"]):
            hit = next(
                (
                    i
                    for i, g in enumerate(truth)
                    if i not in matched
                    and _same_place(alert["box"], g["box"])
                    and g["due"] - tolerance <= alert["alert_time"] <= g["due"] + tolerance
                ),
                None,
            )
            if hit is not None:
                matched.add(hit)
                tp += 1
                totals["latency"].append(alert["alert_time"] - truth[hit]["due"])
        fp, fn = len(alerts) - tp, len(truth) - tp
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        per_stream[stream_id] = {"tp": tp, "fp": fp, "fn": fn, "alerts": len(alerts), "truth": len(truth)}
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    latency = sorted(totals["latency"])
    return {
        "min_duration": min_duration,
        "tolerance": tolerance,
        "verified": vlm is not None,
        "precision_lower_bound": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "median_latency_seconds": latency[len(latency) // 2] if latency else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_stream": per_stream,
        "note": "Truth = annotated, linked put-downs only; unlinked (censored) objects are excluded.",
    }
