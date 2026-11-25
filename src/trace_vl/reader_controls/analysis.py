"""Prespecified source-paired analysis. Positive gain always favours the first-named arm."""

from __future__ import annotations

import math

import numpy as np

DRAWS, SEED = 10000, 7
WEIGHTS = ("start", "trained_reservoir", "trained_learned")
PRIMARY = ("evidence_vs_text", "evidence_vs_latest")
SCRUTINY_ACCURACY = 0.85
RELEVANT_NLL = 0.05

# name: (metric coefficients); NLL is negated in `gain`, so positive = more useful reading.
CONTRASTS = {
    "evidence_vs_text": {"bank_evidence8": 1, "text_only": -1},
    "evidence_vs_latest": {"bank_evidence8": 1, "latest8": -1},
    # Same timestamps, positions and token count; only visual values differ.
    "evidence_content": {"bank_evidence8": 1, "blank:bank_evidence8": -1},
    "latest_content": {"latest8": 1, "blank:latest8": -1},
    "random_content": {"random8": 1, "blank:random8": -1},
    "evidence_minus_latest_content": {
        "bank_evidence8": 1, "blank:bank_evidence8": -1, "latest8": -1, "blank:latest8": 1,
    },
    # Cues available without any visual content.
    "evidence_timing_vs_text": {"blank:bank_evidence8": 1, "text_only": -1},
    "evidence_timing_vs_latest_timing": {"blank:bank_evidence8": 1, "blank:latest8": -1},
    "latest_vs_text": {"latest8": 1, "text_only": -1},
    "recent_vs_text": {"recent4": 1, "text_only": -1},
    "random_vs_text": {"random8": 1, "text_only": -1},
    "random_vs_latest": {"random8": 1, "latest8": -1},
    "spread_vs_latest": {"spread8": 1, "latest8": -1},
    "evidence_vs_random": {"bank_evidence8": 1, "random8": -1},
    "learned_vs_latest": {"learned8": 1, "latest8": -1},
    "learned_vs_random": {"learned8": 1, "random8": -1},
    # Matched subset only; residual age imbalance is reported separately.
    "pair_evidence_vs_distractor": {"pair_evidence": 1, "pair_distractor": -1},
    "pair_timing": {"blank:pair_evidence": 1, "blank:pair_distractor": -1},
    "pair_content": {
        "pair_evidence": 1, "blank:pair_evidence": -1, "pair_distractor": -1, "blank:pair_distractor": 1,
    },
    # 576 visual tokens versus 128: separate budget and a reader context shift.
    "all_bank_576_vs_evidence8_128": {"all_bank": 1, "bank_evidence8": -1},
    "all_bank_576_vs_text": {"all_bank": 1, "text_only": -1},
    "all_bank_576_content": {"all_bank": 1, "blank:all_bank": -1},
}
TRAINING_CHANGE = ("evidence_vs_text", "evidence_vs_latest", "evidence_content", "evidence_minus_latest_content")


def question_metrics(logits, gold):
    """Option-normalized gold NLL and accuracy."""
    if not logits or not all(math.isfinite(x) for x in logits) or not 0 <= gold < len(logits):
        raise ValueError("finite option logits and a valid gold index required")
    top = max(logits)
    norm = top + math.log(sum(math.exp(x - top) for x in logits))
    return {"nll": norm - logits[gold], "acc": float(max(range(len(logits)), key=logits.__getitem__) == gold)}


def arm_values(record):
    """Average read seeds within the source before any between-source inference."""
    groups = {}
    for name, logits in record["logits"].items():
        blank = name.startswith("blank:")
        base = (name[6:] if blank else name).split(":")[0]
        groups.setdefault(("blank:" if blank else "") + base, []).append(
            question_metrics(logits, record["gold"])
        )
    return {a: {k: float(np.mean([m[k] for m in ms])) for k in ("nll", "acc")} for a, ms in groups.items()}


def gain(values, weights, metric):
    sign = -1 if metric == "nll" else 1
    return sign * sum(w * values[a][metric] for a, w in weights.items())


def paired(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"sources": len(values), "mean": None, "ci95": None, "mde80": None}
    generator = np.random.default_rng(SEED)
    draws = values[generator.integers(0, len(values), (DRAWS, len(values)))].mean(1)
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "sources": len(values),
        "mean": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "se": se,
        "mde80": 2.8 * se,
    }


def reading(value):
    if value["ci95"] is None:
        return "insufficient_sources"
    low, high = value["ci95"]
    return "favours_first" if low > 0 else "favours_second" if high < 0 else "inconclusive"


def analyze(records, expected):
    """records: list of score records for all weights; expected: the frozen source order."""
    table = {w: {} for w in WEIGHTS}
    for r in records:
        if r["source_group"] in table[r["weights"]]:
            raise ValueError("one question per source required")
        table[r["weights"]][r["source_group"]] = arm_values(r)
    for w in WEIGHTS:
        if set(table[w]) != set(expected):
            raise ValueError(f"incomplete or unexpected sources: {w}")
    out = {}
    for w in WEIGHTS:
        arms = sorted({a for v in table[w].values() for a in v})
        result = {"arms": {}, "contrasts": {}}
        for arm in arms:
            present = [table[w][s][arm] for s in expected if arm in table[w][s]]
            result["arms"][arm] = {
                "sources": len(present),
                **{k: float(np.mean([p[k] for p in present])) for k in ("nll", "acc")},
            }
        for name, weights in CONTRASTS.items():
            sources = [s for s in expected if all(a in table[w][s] for a in weights)]
            if not name.startswith("pair_") and len(sources) != len(expected):
                raise ValueError(f"non-matched contrast must keep every source: {name}")
            result["contrasts"][name] = {}
            for metric in ("nll", "acc"):
                value = paired([gain(table[w][s], weights, metric) for s in sources])
                result["contrasts"][name][metric] = {**value, "reading": reading(value)}
        out[w] = result
    changes = {}
    for w in WEIGHTS[1:]:
        changes[w] = {}
        for name in TRAINING_CHANGE:
            changes[w][name] = {}
            for metric in ("nll", "acc"):
                value = paired(
                    [
                        gain(table[w][s], CONTRASTS[name], metric)
                        - gain(table["start"][s], CONTRASTS[name], metric)
                        for s in expected
                    ]
                )
                changes[w][name][metric] = {**value, "reading": reading(value)}
    return {"weights": out, "trained_minus_start": changes}


def outcome(analysis):
    """Report flags, not automatic cohort retirement or training approval."""
    start = analysis["weights"]["start"]
    c = start["contrasts"]
    primary = {k: c[k]["nll"]["reading"] for k in PRIMARY}
    passed = all(v == "favours_first" for v in primary.values())
    content = c["evidence_content"]["nll"]["reading"]
    if passed and content == "favours_first":
        status = "headroom_supported: evidence beats both references and its timing-matched blank"
    elif passed:
        status = "headroom_with_timing_cue_unresolved: evidence beats references, not its timing-matched blank"
    elif any(c[k]["nll"]["ci95"][1] < RELEVANT_NLL for k in PRIMARY):
        status = "headroom_small: a primary 95% upper bound is below the prespecified relevant effect"
    else:
        status = "inconclusive: retain intervals and detectable effects; do not infer absence"
    text = start["arms"]["text_only"]["acc"]
    trained = {
        w: {
            k: "reduced_measured_visual_dependence" if v["nll"]["reading"] == "favours_second" else v["nll"]["reading"]
            for k, v in change.items()
        }
        for w, change in analysis["trained_minus_start"].items()
    }
    return {
        "primary_readings": primary,
        "status": status,
        "text_only_accuracy": text,
        "text_only_scrutiny": text >= SCRUTINY_ACCURACY,
        "oracle_accuracy_headroom_over_text": c["evidence_vs_text"]["acc"],
        "trained_minus_start": trained,
        "notes": [
            "Text-only scrutiny does not retire the cohort; weigh oracle headroom and its interval.",
            "Reduced measured visual dependence does not by itself establish shortcut learning.",
            "Only the two primary contrasts are confirmatory (both required); all others are descriptive.",
        ],
    }
