"""Source-paired B0 reading; negative NLL favors the first named policy."""

from __future__ import annotations

import numpy as np

from trace_vl.complementary_memory.analysis import question_metrics

SEEDS = (17, 29, 43)
CONTRASTS = {
    "reservoir_minus_window40": {"reservoir": 1, "window40": -1},
    "window40_minus_window8": {"window40": 1, "window8": -1},
    "reservoir_minus_window8": {"reservoir": 1, "window8": -1},
    "window8_reproduction_change": {"window8": 1, "reference_window8": -1},
}


def paired(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite source values required")
    n = len(values)
    index = np.random.default_rng(7).integers(0, n, (10000, n))
    se = float(values.std(ddof=1) / np.sqrt(n))
    return {"sources": n, "mean": float(values.mean()),
            "ci95": np.quantile(values[index].mean(1), [.025, .975]).tolist(), "se": se, "mde80": 2.8 * se}


def analyze(rows, reference):
    result = {}
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate question")
    for delay in sorted({r["delay_seconds"] for r in rows}):
        subset = sorted((r for r in rows if r["delay_seconds"] == delay), key=lambda r: r["source_group"])
        if len({r["source_group"] for r in subset}) != len(subset):
            raise ValueError("one question per source/delay required")
        values = []
        for row in subset:
            old = reference[row["id"]]
            if any(row[k] != old[k] for k in ("id", "source_group", "gold", "delay_seconds")):
                raise ValueError("reference identity differs")
            if set(row["predictions"]) != {"window8", "window40"}:
                raise ValueError("new prediction coverage differs")
            item = {name: question_metrics(p["option_logits_fp32"], row["gold"])
                    for name, p in row["predictions"].items()}
            item["reference_window8"] = question_metrics(old["predictions"]["window_stream"]["option_logits_fp32"], row["gold"])
            seeded = [question_metrics(old["predictions"][f"reservoir{s}:S1K1"]["option_logits_fp32"], row["gold"])
                      for s in SEEDS]
            item["reservoir"] = {metric: float(np.mean([v[metric] for v in seeded])) for metric in ("nll", "acc")}
            values.append(item)
        arms = {name: {metric: float(np.mean([v[name][metric] for v in values])) for metric in ("nll", "acc")}
                for name in values[0]}
        contrasts = {}
        for name, weights in CONTRASTS.items():
            contrasts[name] = {metric: paired([sum(c * v[arm][metric] for arm, c in weights.items())
                                               for v in values]) for metric in ("nll", "acc")}
        result[str(delay)] = {"sources": len(subset), "arms": arms, "contrasts": contrasts}
    return result


def outcome(analysis):
    primary = analysis["300"]["contrasts"]["reservoir_minus_window40"]["nll"]
    low, high = primary["ci95"]
    reading = "favours_reservoir" if high < 0 else "favours_window40" if low > 0 else "inconclusive"
    integrity = analysis["0"]["contrasts"]["window40_minus_window8"]["nll"]
    return {"primary_300s": reading, "primary_difference": primary,
            "window40_0s_damage_trigger": integrity["ci95"][0] > .10,
            "scope": "Equal retained-step policy comparison, separately evolved states; reused development cohort.",
            "automatic_training": False}
