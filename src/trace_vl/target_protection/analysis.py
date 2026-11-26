"""Fixed 300 s primary; all other horizons descriptive and source-paired."""

from __future__ import annotations

import numpy as np

from trace_vl.b0_window40.analysis import paired
from trace_vl.complementary_memory.analysis import question_metrics

SEEDS = (17, 29, 43)
RELEVANT_EFFECT = .05


def analyze(rows, reference, b0):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate question")
    result = {}
    for delay in sorted({r["delay_seconds"] for r in rows}):
        subset = sorted((r for r in rows if r["delay_seconds"] == delay), key=lambda r: r["source_group"])
        if len({r["source_group"] for r in subset}) != len(subset):
            raise ValueError("one question per source/delay required")
        values = []
        for row in subset:
            old, window = reference[row["id"]], b0[row["id"]]
            if any(row[k] != old[k] or row[k] != window[k] for k in ("id", "source_group", "gold", "delay_seconds")):
                raise ValueError("reference identity differs")
            item = {}
            for name, predictions in (
                ("protected", [row["predictions"][f"protected{s}"] for s in SEEDS]),
                ("reservoir", [old["predictions"][f"reservoir{s}:S1K1"] for s in SEEDS]),
                ("window40", [window["predictions"]["window40"]]),
            ):
                metrics = [question_metrics(p["option_logits_fp32"], row["gold"]) for p in predictions]
                item[name] = {m: float(np.mean([v[m] for v in metrics])) for m in ("nll", "acc")}
            values.append(item)
        result[str(delay)] = {
            "sources": len(subset),
            "arms": {arm: {m: float(np.mean([v[arm][m] for v in values])) for m in ("nll", "acc")}
                     for arm in ("protected", "reservoir", "window40")},
            "contrasts": {f"protected_minus_{arm}": {
                m: paired([v["protected"][m] - v[arm][m] for v in values]) for m in ("nll", "acc")}
                for arm in ("reservoir", "window40")},
        }
    return result


def outcome(analysis):
    primary = analysis["300"]["contrasts"]["protected_minus_reservoir"]["nll"]
    low, high = primary["ci95"]
    reading = "favours_protection" if high < 0 else "favours_reservoir" if low > 0 else "inconclusive"
    return {"primary_300s": reading, "primary_difference": primary,
            "relevant_improvement_005_excluded": low > -RELEVANT_EFFECT,
            "relevant_effect_nats": RELEVANT_EFFECT,
            "short_horizons": "descriptive only; no noninferiority margin or claim",
            "automatic_training": False,
            "scope": "Privileged equal-step headroom on reused development sources, separately evolved states."}
