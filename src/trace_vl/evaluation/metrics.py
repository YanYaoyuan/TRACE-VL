"""Local metrics, not substitutes for every upstream leaderboard protocol."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np


def normalized_text(value):
    value = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def choice_letter(value, count=4):
    m = re.match(r"^\s*(?:(?:the\s+)?answer\s*(?:is|:)?\s*)?([A-Z])(?:[\s.():]|$)", str(value), re.I)
    result = m.group(1).upper() if m else None
    return result if result in [chr(65 + i) for i in range(count)] else None


def token_f1(prediction, target):
    pred, truth = normalized_text(prediction).split(), normalized_text(target).split()
    if not pred or not truth:
        return float(pred == truth)
    shared = sum((Counter(pred) & Counter(truth)).values())
    return 2 * shared / (len(pred) + len(truth))


def auroc(labels, scores):
    """Mann-Whitney AUROC with average ranks for tied scores; undefined -> None."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid AUROC input")
    pos, neg = (labels == 1).sum(), (labels == 0).sum()
    if not pos or not neg:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = ((i + 1) + j) / 2
        i = j
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    if len(labels) != len(scores) or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid AP input")
    if not labels.sum():
        return None
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(y) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    recall = tp / labels.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def event_counts(predicted, truth, cutoff, observed_start=0):
    """One-to-one matching inside [known_at, deadline], no future GT added to denominator."""
    eligible = [e for e in truth if observed_start <= e["known_at"] <= cutoff]
    # Do not score windows whose deadline is beyond the visible prefix: unfinished opportunity.
    completed = [e for e in eligible if e["deadline"] <= cutoff]
    censored = [e for e in eligible if e["deadline"] > cutoff]
    unmatched = set(range(len(completed)))
    tp, fp, delays = 0, 0, []
    for prediction in sorted(predicted, key=lambda x: x["time"]):
        t = float(prediction["time"])
        if not math.isfinite(t) or t < observed_start or t > cutoff + 1e-6:
            raise ValueError("alert outside the observed prefix")
        category = prediction.get("category", "event")
        candidates = [
            i
            for i in unmatched
            if completed[i]["category"] == category
            and completed[i]["known_at"] <= t <= completed[i]["deadline"]
        ]
        if candidates:
            i = min(candidates, key=lambda i: completed[i]["deadline"])
            unmatched.remove(i)
            tp += 1
            delays.append(t - completed[i]["known_at"])
        elif any(e["category"] == category and e["known_at"] <= t <= cutoff for e in censored):
            continue
        else:
            fp += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": len(unmatched),
        "delays": delays,
        "censored_events": len(censored),
        "hours": max(0, cutoff - observed_start) / 3600,
    }


def score_predictions(records, predictions):
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate ground-truth ids")
    pred_map = {r["id"]: r for r in predictions}
    if len(pred_map) != len(predictions):
        raise ValueError("duplicate prediction ids")
    if set(by_id) != set(pred_map):
        raise ValueError(
            f"prediction/GT ids mismatch: missing={sorted(set(by_id) - set(pred_map))[:5]}, "
            f"extra={sorted(set(pred_map) - set(by_id))[:5]}"
        )
    if not records:
        raise ValueError("empty evaluation")
    episodes = set()
    groups, exact, f1 = defaultdict(list), [], []
    totals = dict(tp=0, fp=0, fn=0, censored_events=0, hours=0.0)
    delays, labels, scores, latencies = [], [], [], []
    for ident, r in by_id.items():
        p = pred_map[ident]
        cutoff = float(p["observed_until"])
        if not math.isfinite(cutoff) or cutoff < r["observation"]["start"]:
            raise ValueError("invalid prediction cutoff")
        expected = r["observation"]["end"]
        if expected != "video_end" and abs(cutoff - float(expected)) > 1e-5:
            raise ValueError("prediction uses a different observation cutoff")
        if r.get("answer_type") == "choice":
            correct = float(choice_letter(p.get("answer", ""), len(r["options"])) == r["answer"])
        else:
            correct = float(normalized_text(p.get("answer", "")) == normalized_text(r["answer"]))
        exact.append(correct)
        f1.append(token_f1(p.get("answer", ""), r["answer"]))
        groups[r.get("category", "unknown")].append(correct)
        groups["duration:" + r.get("duration_group", "unspecified")].append(correct)
        if "events" in r:
            episode = (r["source_group"], r["video"], r["observation"]["start"], cutoff, r["rule"])
            if episode in episodes:
                raise ValueError(
                    "duplicate monitoring episode; avoid counting observation hours multiple times"
                )
            episodes.add(episode)
            e = event_counts(p.get("alerts", []), r["events"], cutoff, r["observation"]["start"])
            delays.extend(e.pop("delays"))
            for k, value in e.items():
                totals[k] += value
        frame_times = set()
        for frame in p.get("frame_scores", []):
            t, score = float(frame["time"]), float(frame["score"])
            if not math.isfinite(t) or t < r["observation"]["start"] or t > cutoff + 1e-6:
                raise ValueError("frame outside visible prefix")
            if t in frame_times or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("duplicate timestamp or invalid frame score")
            frame_times.add(t)
            if "events" in r:
                # Derive truth here. Prediction files never supply their own labels.
                labels.append(int(any(e["known_at"] <= t <= e["end"] for e in r["events"])))
                scores.append(score)
        if p.get("query_latency_seconds") is not None:
            latency = float(p["query_latency_seconds"])
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("invalid latency")
            latencies.append(latency)
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {
        "samples": len(records),
        "qa_exact_match": float(np.mean(exact)),
        "qa_token_f1": float(np.mean(f1)),
        "qa_by_group": {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in groups.items()},
        "events": {
            **totals,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "false_alarms_per_hour": fp / totals["hours"] if totals["hours"] else None,
            "miss_rate": fn / (tp + fn) if tp + fn else None,
            "latency_p50_seconds": float(np.median(delays)) if delays else None,
            "latency_p95_seconds": float(np.percentile(delays, 95)) if delays else None,
        },
        "sampled_frame_auroc": auroc(labels, scores) if labels else None,
        "sampled_frame_ap": average_precision(labels, scores) if labels else None,
        "query_latency_p50_seconds": float(np.median(latencies)) if latencies else None,
        "query_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
        "protocol_note": "Local normalized QA/event metrics. Sampled-frame AUC/AP are not official dense-frame anomaly scores.",
    }
