"""Prespecified paired content-NLL endpoints; reused development cohort only."""

from __future__ import annotations

import numpy as np

from trace_vl.reader_controls.analysis import question_metrics

from . import policies

PRIMARY = ("spread16_minus8_content", "random16_minus8_content")
CONTRASTS = {
    "spread16_minus8_content": {"spread16": 1, "blank:spread16": -1, "spread8": -1, "blank:spread8": 1},
    "random16_minus8_content": {"random16": 1, "blank:random16": -1, "random8": -1, "blank:random8": 1},
    "all36_minus_spread8_content": {"all36": 1, "blank:all36": -1, "spread8": -1, "blank:spread8": 1},
    "all36_minus_random8_content": {"all36": 1, "blank:all36": -1, "random8": -1, "blank:random8": 1},
    "all36_minus_spread16_content": {"all36": 1, "blank:all36": -1, "spread16": -1, "blank:spread16": 1},
    "all36_minus_random16_content": {"all36": 1, "blank:all36": -1, "random16": -1, "blank:random16": 1},
}
for _policy in ("spread8", "spread16", "random8", "random16", "all36"):
    CONTRASTS[_policy + "_content"] = {_policy: 1, "blank:" + _policy: -1}


def arm_values(record):
    groups = {}
    for name, logits in record["logits"].items():
        parts = name.rsplit(":", 1)
        base = parts[0] if parts[-1].isdigit() else name
        groups.setdefault(base, []).append(question_metrics(logits, record["gold"]))
    return {name: {metric: float(np.mean([v[metric] for v in values])) for metric in ("nll", "acc")}
            for name, values in groups.items()}


def paired(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite source values required")
    indices = np.random.default_rng(7).integers(0, n, (10000, n))
    low, high = np.quantile(values[indices].mean(1), [.025, .975]).tolist()
    se = float(values.std(ddof=1) / np.sqrt(n))
    return {"sources": n, "mean": float(values.mean()), "ci95": [low, high], "se": se,
            "mde80": 2.8 * se, "reading": "helps" if low > 0 else "harms" if high < 0 else "inconclusive"}


def analyze(records, questions):
    expected = {q["id"]: q for q in questions}
    if len(records) != len(expected) or len({r["id"] for r in records}) != len(expected):
        raise ValueError("incomplete or duplicate source scores")
    by_id = {r["id"]: r for r in records}
    if set(by_id) != set(expected) or len({q["source_group"] for q in questions}) != len(questions):
        raise ValueError("source coverage differs")
    values = {}
    for q in questions:
        r = by_id[q["id"]]
        names = {"text_only", *q["reads"], *("blank:" + n for n in q["reads"])}
        if set(r["logits"]) != names or set(r["visual_tokens"]) != names:
            raise ValueError("policy coverage differs")
        if r["gold"] != q["gold"] or r["source_group"] != q["source_group"]:
            raise ValueError("question identity differs")
        if any(r["visual_tokens"][n] != policies.expected_tokens(n) for n in names):
            raise ValueError("visual budget differs")
        values[q["id"]] = arm_values(r)
    arms = {name: {metric: float(np.mean([values[q["id"]][name][metric] for q in questions]))
                   for metric in ("nll", "acc")} for name in next(iter(values.values()))}
    contrasts = {}
    for name, weights in CONTRASTS.items():
        contrasts[name] = {}
        for metric in ("nll", "acc"):
            # Positive is improvement: lower NLL, higher accuracy.
            sign = -1 if metric == "nll" else 1
            contrasts[name][metric] = paired([
                sign * sum(c * values[q["id"]][arm][metric] for arm, c in weights.items()) for q in questions
            ])
    task_metrics = {}
    for task in sorted({q["task"] for q in questions}):
        subset = [q for q in questions if q["task"] == task]
        task_metrics[task] = {"sources": len(subset), "arms": {
            name: {metric: float(np.mean([values[q["id"]][name][metric] for q in subset]))
                   for metric in ("nll", "acc")} for name in arms}}
    balanced = {name: float(np.mean([v["arms"][name]["acc"] for v in task_metrics.values()])) for name in arms}
    readings = {name: contrasts[name]["nll"]["reading"] for name in PRIMARY}
    return {"sources": len(questions), "arms": arms, "contrasts": contrasts, "tasks": task_metrics,
            "balanced_accuracy": balanced, "primary_readings": readings,
            "both_primary_intervals_positive": all(value == "helps" for value in readings.values()),
            "scope": "Reused development diagnostic; no official, unseen, writer or training claim."}
