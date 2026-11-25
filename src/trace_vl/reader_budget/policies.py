"""Nested question-blind reads from a fixed 32-history + 4-recent bank."""

from __future__ import annotations

import math

import numpy as np

from trace_vl.core_vlm.prepare import digest

BUDGETS = (8, 16, 36)
SEEDS = (17, 29, 43, 61, 83)
RECENT, HISTORY = 4, 32


def validate_times(times, query_time):
    if len(times) != HISTORY + RECENT or not math.isfinite(query_time):
        raise ValueError("a fixed 36-step causal bank is required")
    previous = -math.inf
    for pair in times:
        if len(pair) != 2 or not all(math.isfinite(t) for t in pair):
            raise ValueError("finite two-frame steps required")
        start, end = pair
        if not 0 <= start <= end <= query_time or end <= previous:
            raise ValueError("bank times must be causal and strictly ordered")
        previous = end


def spread_order(times):
    """Oldest first, then maximin temporal distance; ties choose the oldest index."""
    ends = [pair[-1] for pair in times[:HISTORY]]
    selected = [0]
    remaining = set(range(1, HISTORY))
    while remaining:
        chosen = max(remaining, key=lambda i: (min(abs(ends[i] - ends[j]) for j in selected), -i))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def build(times, query_time, source_group):
    """Uses bank timestamps and source seed only, never question/answer/evidence."""
    validate_times(times, query_time)
    recent = list(range(HISTORY, HISTORY + RECENT))
    reads = {"all36": list(range(HISTORY + RECENT))}
    orders = {"spread": spread_order(times)}
    for seed in SEEDS:
        value = int(digest(["reader-budget-v1", source_group, seed])[:8], 16)
        orders[f"random:{seed}"] = np.random.default_rng(value).permutation(HISTORY).tolist()
    for family, order in orders.items():
        for budget in BUDGETS[:-1]:
            name, _, seed = family.partition(":")
            key = f"{name}{budget}" + (f":{seed}" if seed else "")
            reads[key] = sorted(order[:budget - RECENT] + recent)
    return reads


def expected_tokens(name):
    policy = name.removeprefix("blank:")
    if policy == "text_only":
        return 0
    base = policy.split(":")[0]
    sizes = {"all36": 36, "spread8": 8, "spread16": 16, "random8": 8, "random16": 16}
    if base not in sizes:
        raise ValueError("unknown read policy")
    return 16 * sizes[base]
