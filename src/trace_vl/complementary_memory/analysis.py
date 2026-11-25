"""Prespecified Test A source-paired analysis. No accuracy gate controls execution."""
from __future__ import annotations

import math

import numpy as np

FACTORS = ("S1K1", "S1K0", "S0K1", "S0K0")
DRAWS, SEED = 10000, 7


def question_metrics(logits, gold):
    top = max(logits)
    norm = top + math.log(sum(math.exp(x - top) for x in logits))
    return {"nll": norm - logits[gold], "acc": float(max(range(len(logits)), key=logits.__getitem__) == gold)}


def arm_values(row, seeds):
    """Memory seeds are averaged within the question/source before any comparison."""
    m = {a: question_metrics(p["option_logits_fp32"], row["gold"]) for a, p in row["predictions"].items()}
    out = {a: v for a, v in m.items() if not a.startswith("reservoir")}
    for factor in FACTORS:
        out[factor] = {k: float(np.mean([m[f"reservoir{s}:{factor}"][k] for s in seeds])) for k in ("nll", "acc")}
    return out


# (weights over arms); positive = first named quantity has the larger metric value
CONTRASTS = {
    "carried_state": {"window_stream": 1, "recent_replay": -1},
    "history_kv_given_state": {"S1K1": 1, "window_stream": -1},
    "history_kv_fork_S1": {"S1K1": 1, "S1K0": -1},
    "history_kv_fork_S0": {"S0K1": 1, "S0K0": -1},
    "state_fork_K0": {"S1K0": 1, "S0K0": -1},
    "state_fork_K1": {"S1K1": 1, "S0K1": -1},
    "interaction": {"S1K1": 1, "S1K0": -1, "S0K1": -1, "S0K0": 1},
    "full_kv_headroom": {"full_stream": 1, "window_stream": -1},
    "reservoir_gap_to_full": {"full_stream": 1, "S1K1": -1},
    "snapshot_headroom": {"snapshot_replay": 1, "window_stream": -1},
}


def paired(values, rng):
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), (DRAWS, len(values)))].mean(1)
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
    return {"mean": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "se": se,
            "mde80": None if se is None else 2.8 * se, "sources": len(values)}


def label(value, *, lower_is_better):
    low, high = value["ci95"]
    if lower_is_better:
        low, high = -high, -low
    return "helps" if low > 0 else "harms" if high < 0 else "inconclusive"


def analyze(rows, seeds):
    out = {}
    for delay in sorted({r["delay_seconds"] for r in rows}):
        subset = sorted((r for r in rows if r["delay_seconds"] == delay), key=lambda r: r["source_group"])
        if len({r["source_group"] for r in subset}) != len(subset):
            raise ValueError("one question per source and delay required")
        values = [arm_values(r, seeds) for r in subset]
        arms = {a: {k: float(np.mean([v[a][k] for v in values])) for k in ("nll", "acc")} for a in values[0]}
        contrasts = {}
        for name, weights in CONTRASTS.items():
            contrasts[name] = {}
            for metric in ("nll", "acc"):
                rng = np.random.default_rng(SEED)
                result = paired([sum(w * v[a][metric] for a, w in weights.items()) for v in values], rng)
                # An interaction is signed evidence, not a benefit.
                if name != "interaction":
                    result["reading"] = label(result, lower_is_better=metric == "nll")
                contrasts[name][metric] = result
        out[str(delay)] = {"sources": len(subset), "arms": arms, "contrasts": contrasts}
    return out


def outcome(analysis):
    """Frozen reading of the 300 s endpoint plus the 0 s integrity check; informs, never proves novelty."""
    late, now = analysis["300"]["contrasts"], analysis["0"]["contrasts"]
    r = {k: late[k]["nll"]["reading"] for k in CONTRASTS if k != "interaction"}
    low, high = late["interaction"]["nll"]["ci95"]
    interaction = "kv_worth_more_with_carried_state" if high < 0 else (
        "kv_worth_less_with_carried_state" if low > 0 else "inconclusive")
    broken = now["carried_state"]["nll"]["ci95"][0] > .10
    headroom = "helps" in (r["full_kv_headroom"], r["snapshot_headroom"])
    if broken:
        decision = "reader_unfit: carried stream clearly damages 0s answers; train the reader before judging"
    elif r["history_kv_given_state"] == "helps":
        decision = "go_B: historical KV adds answer value beyond the carried state"
    elif headroom:
        decision = "go_B_for_selection_only: headroom exists but Reservoir does not recover it"
    else:
        decision = "hold: no demonstrated history headroom; inconclusive unless mde80 is small"
    return {"decision": decision, "readings_nll_300s": r, "interaction_nll_300s": interaction,
            "state_conditional_claim_supported": interaction != "inconclusive"}
