"""Frozen read policies over one stored 36-step bank. No model output enters any policy."""

from __future__ import annotations

import random
from math import comb

from trace_vl.core_vlm.prepare import digest

READ = 8
SEEDS = (17, 29, 43, 61, 83)
CALIPER_SECONDS = 10.0


def evidence_hits(times, evidence):
    """Same definition as the completed CPU audit: any sampled timestamp inside the interval."""
    a, b = evidence
    return [i for i, pair in enumerate(times) if any(a <= t <= b for t in pair)]


def spaced(items, count=READ):
    """Deterministic, evenly spaced subset; order preserved."""
    n = len(items)
    if n <= count:
        return list(items)
    return [items[(i * (n - 1) + (count - 1) // 2) // (count - 1)] for i in range(count)]


def rng(question, name, seed):
    return random.Random(int(digest(["reader-controls-v1", question, name, seed])[:16], 16))


def match_ages(hits, others, times, caliper=CALIPER_SECONDS):
    """One-to-one matching on step end time, without replacement.

    Maximum cardinality within the caliper, then minimum total absolute age
    difference. In one dimension an order-preserving optimum exists, so a small
    dynamic program is exact. Ties break deterministically.
    """
    hits, others = sorted(hits), sorted(others)
    end = [pair[-1] for pair in times]
    # value: (unmatched-first key, total distance, pairs)
    table = [[(0, 0.0, ()) for _ in range(len(others) + 1)] for _ in range(len(hits) + 1)]
    for i in range(1, len(hits) + 1):
        for j in range(1, len(others) + 1):
            choices = [table[i - 1][j], table[i][j - 1]]
            gap = abs(end[hits[i - 1]] - end[others[j - 1]])
            if gap <= caliper:
                n, cost, pairs = table[i - 1][j - 1]
                choices.append((n - 1, cost + gap, pairs + ((hits[i - 1], others[j - 1]),)))
            table[i][j] = min(choices)
    return list(table[-1][-1][2])


def build(row, times, learned):
    """Return (policies, audit). Seeded policies are named `<policy>:<seed>`."""
    n = len(times)
    if n < READ or len(learned) != READ or sorted(set(learned)) != list(learned):
        raise ValueError("expected a full bank and eight ordered saved selections")
    if learned[0] < 0 or learned[-1] >= n:
        raise ValueError("saved selection outside the bank")
    hits = evidence_hits(times, row["evidence"])
    others = [i for i in range(n) if i not in set(hits)]
    policies = {
        "latest8": list(range(n - READ, n)),
        "recent4": list(range(n - 4, n)),
        "spread8": spaced(list(range(n))),
        "learned8": list(learned),
        "all_bank": list(range(n)),
    }
    chosen = spaced(hits)
    matched = match_ages(hits, others, times)
    pairs = spaced(matched)
    used = {i for pair in pairs for i in pair}
    pool = [i for i in others if i not in used]
    for seed in SEEDS:
        policies[f"random8:{seed}"] = sorted(rng(row["id"], "random8", seed).sample(range(n), READ))
        fill = rng(row["id"], "evidence8", seed).sample(others, min(READ - len(chosen), len(others)))
        policies[f"bank_evidence8:{seed}"] = sorted(chosen + fill)
        if pairs:
            # Identical fill in both arms: only the matched steps differ.
            shared = rng(row["id"], "pair", seed).sample(pool, min(READ - len(pairs), len(pool)))
            policies[f"pair_evidence:{seed}"] = sorted([a for a, _ in pairs] + shared)
            policies[f"pair_distractor:{seed}"] = sorted([b for _, b in pairs] + shared)
    ages = [row["query_time"] - pair[-1] for pair in times]
    signed = [ages[a] - ages[b] for a, b in pairs]
    audit = {
        "bank_steps": n,
        "evidence_steps": len(hits),
        "evidence_steps_in_evidence8": len(chosen),
        "evidence_hit": {
            "bank": bool(hits),
            **{
                k: bool(set(v) & set(hits)) for k, v in policies.items() if k in ("latest8", "recent4", "learned8")
            },
        },
        "random8_expected_hit": 1 - comb(len(others), READ) / comb(n, READ),
        "pair": {
            "matched_within_caliper": len(matched),
            "unmatched_evidence_steps": len(hits) - len(matched),
            "pairs_read": len(pairs),
            "failure": None if pairs else ("no_stored_evidence" if not hits else "no_match_within_caliper"),
            "steps_per_arm": min(READ, len(pairs) + len(pool)) if pairs else 0,
            "age_difference_seconds": signed,
            "evidence_mean_age": sum(ages[a] for a, _ in pairs) / len(pairs) if pairs else None,
            "distractor_mean_age": sum(ages[b] for _, b in pairs) / len(pairs) if pairs else None,
        },
    }
    return policies, audit
