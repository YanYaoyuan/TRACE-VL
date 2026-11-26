"""One frozen fresh-arm primary; unreproduced cross-run references remain secondary."""
from __future__ import annotations

import numpy as np

from trace_vl.b0_window40.analysis import paired
from trace_vl.complementary_memory.analysis import question_metrics

SEEDS = (17, 29, 43)
CAPACITIES = (32, 128)
SMOKE_SECONDS = 3600
FULL_SECONDS = 15600


def analyze(rows, windows, protected):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate question")
    result = {}
    for delay in sorted({r["delay_seconds"] for r in rows}):
        subset = sorted((r for r in rows if r["delay_seconds"] == delay), key=lambda r: r["source_group"])
        if len({r["source_group"] for r in subset}) != len(subset):
            raise ValueError("one question per source/delay required")
        values = []
        for row in subset:
            item = {}
            predictions = {f"R{cap}": [row["predictions"][f"r{cap}s{s}"] for s in SEEDS] for cap in CAPACITIES}
            if delay in (15, 300):
                window, target = windows[row["id"]], protected[row["id"]]
                if any(row[k] != ref[k] for ref in (window, target) for k in ("id", "source_group", "gold", "delay_seconds")):
                    raise ValueError("cross-run reference identity differs")
                predictions.update(saved_window40=[window["predictions"]["window40"]],
                                   saved_protected=[target["predictions"][f"protected{s}"] for s in SEEDS])
            for name, preds in predictions.items():
                metrics = [question_metrics(p["option_logits_fp32"], row["gold"]) for p in preds]
                item[name] = {m: float(np.mean([v[m] for v in metrics])) for m in ("nll", "acc")}
            values.append(item)
        contrasts = {"R128_minus_R32": "R32"}
        if delay in (15, 300):
            contrasts.update(R128_minus_saved_window40="saved_window40", R128_minus_saved_protected="saved_protected")
        result[str(delay)] = {
            "sources": len(subset),
            "arms": {a: {m: float(np.mean([v[a][m] for v in values])) for m in ("nll", "acc")} for a in values[0]},
            "contrasts": {name: {m: paired([v["R128"][m] - v[other][m] for v in values]) for m in ("nll", "acc")}
                          for name, other in contrasts.items()},
        }
    return result


def outcome(analysis):
    primary = analysis["300"]["contrasts"]["R128_minus_R32"]["nll"]
    low, high = primary["ci95"]
    reading = "favours_R128" if high < 0 else "favours_R32" if low > 0 else "inconclusive"
    return {"primary_300s": reading, "primary_difference": primary, "automatic_training": False,
            "scope": "Unequal-budget capacity-policy comparison on reused development sources; separately evolved states.",
            "secondary_scope": "Pointwise intervals; no noninferiority, monotonicity or optimal-set claim."}
