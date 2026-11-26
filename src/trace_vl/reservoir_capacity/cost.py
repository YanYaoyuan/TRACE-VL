"""Smoke-only time/capacity receipt; never reads question predictions."""
from __future__ import annotations

import math

from .analysis import FULL_SECONDS, SEEDS


def estimate_cost(inputs, sources, workers, gpu_count):
    base = inputs["reference"]
    if (len(sources) != 2 or {s["source_group"] for s in sources} != set(base["smoke_source_groups"])
            or any(not s["completed"] or s["input_id"] != inputs["id"] or s["live_streams"] != 6 for s in sources)):
        raise ValueError("both complete fixed six-stream smoke sources required")
    if len(workers) != 2 or any(w.get("exit_code") != 0 or not math.isfinite(w["wall_seconds"]) or w["wall_seconds"] <= 0 for w in workers.values()):
        raise ValueError("both smoke workers must exit normally")
    if gpu_count != 4:
        raise ValueError("cost protocol requires four full shards")
    step_fields = ("vision", "packing_audit", "lm_r32", "lm_r128")
    query_fields = ("answer_r32", "answer_r128", "question_verification")
    for s in sources:
        if (s["steps"] <= 0 or not math.isfinite(s["source_wall_seconds"]) or s["source_wall_seconds"] <= 0
                or any(not math.isfinite(t) or t < 0 for t in s["seconds"].values())):
            raise ValueError("finite positive smoke timing required")
    per_step = {k: sum(s["seconds"][k] for s in sources) / sum(s["steps"] for s in sources) for k in step_fields}
    per_query = {k: sum(s["seconds"][k] for s in sources) / 10 for k in query_fields}
    maximum_step = {k: max(s["seconds"][k] / s["steps"] for s in sources) for k in step_fields}
    maximum_query = {k: max(s["seconds"][k] / 5 for s in sources) for k in query_fields}
    residual_per_step = max(max(0., s["source_wall_seconds"] - sum(s["seconds"].values())) / s["steps"] for s in sources)
    smoke_order = [s["source_group"] for s in base["sources"] if s["source_group"] in base["smoke_source_groups"]]
    by_group = {s["source_group"]: s for s in sources}
    startup = max(max(0., workers[f"smoke-{i}"]["wall_seconds"] - by_group[group]["source_wall_seconds"])
                  for i, group in enumerate(smoke_order))
    done = set(base["smoke_source_groups"])
    shards = []
    for shard in range(gpu_count):
        queue = [s for s in base["sources"][shard::gpu_count] if s["source_group"] not in done]
        steps = sum(max(q["planned_encoder_calls"] for q in s["queries"]) for s in queue)
        questions = sum(len(s["queries"]) for s in queue)
        component = steps * sum(per_step.values()) + questions * sum(per_query.values())
        conservative = steps * (sum(maximum_step.values()) + residual_per_step) + questions * sum(maximum_query.values()) + startup + 120
        shards.append({"shard": shard, "sources": len(queue), "steps": steps, "questions": questions,
                       "component_seconds": component, "conservative_seconds": conservative})
    conservative = max(s["conservative_seconds"] for s in shards)
    if not math.isfinite(conservative) or conservative <= 0:
        raise ValueError("invalid conservative estimate")
    return {"inputs": inputs["id"], "remaining_shards": shards,
            "component_wall_seconds": max(s["component_seconds"] for s in shards),
            "conservative_wall_seconds": conservative, "full_ceiling_seconds": FULL_SECONDS,
            "within_ceiling": conservative <= FULL_SECONDS, "per_step_components": per_step,
            "per_query_components_three_seeds": per_query, "max_per_step_components": maximum_step,
            "max_per_query_components_three_seeds": maximum_query, "residual_seconds_per_step": residual_per_step,
            "startup_allowance_seconds": startup, "report_bundle_allowance_seconds": 120,
            "first_answer_lm_seconds_per_query_seed": {f"R{c}": per_query[f"answer_r{c}"] / len(SEEDS) for c in (32, 128)},
            "scope": "Measured two-source extrapolation, not a deadline guarantee. Max-source component rates plus scaled residual, conservative startup and 120s reporting allowance; actual watchdog enforces the ceiling."}
