"""Measure true microbatch throughput on prepared QA, at a fixed global batch.

Each trial starts from the original model and stops after a short prefix of the
target schedule. Trial checkpoints are separate from training runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest
from trace_vl.v3.tracking import Tracker, enabled_from_env

from .data import read_split, sha256
from .experiment import run_jobs, training_command
from .prepare import digest, write_json
from .train import checked_checkpoint

DEFAULT_LAYOUTS = {f"mb{size}": (size, 1) for size in (1, 2, 4)}


def trial_summary(folder, steps, warmup):
    run = json.loads((folder / "run.json").read_text())
    checkpoint = checked_checkpoint(folder / f"checkpoint-{steps:06d}", run["id"])
    rows = [json.loads(s) for s in (folder / "metrics.jsonl").read_text().splitlines()]
    if [r["update"] for r in rows] != list(range(1, steps + 1)):
        raise ValueError("incomplete benchmark update sequence")
    if any(not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"])
           or r["gradient_norm"] <= 0
           or not math.isfinite(r["seconds_per_update"]) or r["seconds_per_update"] <= 0 for r in rows):
        raise ValueError("nonfinite benchmark loss/gradient")
    seconds = statistics.median(r["seconds_per_update"] for r in rows[warmup:])
    sample_seconds = [r["sample_preparation_seconds_rank0"] for r in rows[warmup:]
                      if "sample_preparation_seconds_rank0" in r]
    return {"passed": True, "median_seconds_per_update": seconds,
            **({"median_sample_preparation_seconds_rank0": statistics.median(sample_seconds)}
               if sample_seconds else {}),
            "samples_per_second": rows[-1]["effective_batch_size"] / seconds,
            "peak_allocated_gib_max": max(r["peak_allocated_gib_max"] for r in rows),
            "peak_reserved_gib_max": max(r["peak_reserved_gib_max"] for r in rows),
            "first_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
            "initial_lora": json.loads((folder / "initialization.json").read_text())["lora_sha256"],
            "question_ids_by_update": [r["question_ids_by_rank"] for r in rows],
            "checkpoint": checkpoint}


def recommend(trials, memory_limit=32., loss_tolerance=.02, *, layouts=None, per_rank=4):
    choices = DEFAULT_LAYOUTS if layouts is None else layouts
    reference = next(iter(choices))
    candidates = {}
    for name in choices:
        measured = []
        for arm in ("plain", "temporal"):
            baseline, trial = trials[f"{arm}-{reference}"], trials[f"{arm}-{name}"]
            if not baseline["passed"] or not trial["passed"]:
                break
            if (baseline["question_ids_by_update"] != trial["question_ids_by_update"]
                    or baseline["initial_lora"] != trial["initial_lora"]
                    or abs(baseline["first_loss"] - trial["first_loss"]) > loss_tolerance
                    or trial["peak_reserved_gib_max"] > memory_limit):
                break
            measured.append(trial["median_seconds_per_update"])
        if len(measured) == 2:
            candidates[name] = sum(measured)
    best = min(candidates, key=candidates.get) if candidates else None
    size, workers = choices[best] if best else (None, None)
    return {"microbatch_size": size, "accumulation": per_rank // size if size else None,
            "effective_batch_size": 4 * per_rank,
            "candidate_seconds": {str(choices[name][0]) if layouts is None else name: seconds
                                  for name, seconds in candidates.items()},
            **({"layout": best, "decode_workers": workers} if layouts is not None else {}),
            "memory_limit_gib": memory_limit, "first_loss_absolute_tolerance": loss_tolerance,
            "scope": "Timing recommendation on this training prefix. GPU gradients require numerical tolerance. "
                     "Inspect first-loss differences and full training curves before scientific claims."}


def run(prepared, out, steps=12, warmup=2, recipe="preserved", *, layouts=None, per_rank=4):
    if not 0 <= warmup < steps or steps < 3 or recipe not in {"ce", "preserved"}:
        raise ValueError("invalid benchmark settings")
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    cfg = json.loads((prepared / "config.json").read_text())
    complete = json.loads((prepared / "mixture/complete.json").read_text())
    source_manifest = prepared / "mixture/manifest.jsonl"
    if sha256(source_manifest) != complete["files"]["manifest.jsonl"]:
        raise ValueError("prepared manifest changed")
    choices = DEFAULT_LAYOUTS if layouts is None else layouts
    if len(cfg["gpus"]) != 4 or not choices or any(
        min(size, workers) < 1 or per_rank % size for size, workers in choices.values()
    ):
        raise ValueError("layouts must divide the fixed per-rank batch on four GPUs")
    out.mkdir(parents=True, exist_ok=True)
    rows = read_split(source_manifest, "train")
    count = min(len(rows), max(256, steps * per_rank * 4))
    ordered = sorted(rows, key=lambda r: digest(["batch-speed-v1", cfg["seed"], r["id"]]))
    selected = sorted(ordered[:count], key=lambda r: r["id"])
    identity = {"config": cfg, "steps": steps, "warmup": warmup, "recipe": recipe,
                "manifest_sha256": sha256(source_manifest), "benchmark_ids": [r["id"] for r in selected],
                "prepared_complete_sha256": sha256(prepared / "mixture/complete.json"),
                "code": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "backbone_sha256": sha256(Path(__file__).parents[1] / "v3/backbone.py")}
    if layouts is not None:
        identity.update(layouts={k: list(v) for k, v in layouts.items()}, per_rank_batch=per_rank)
    identity["id"] = digest(identity)
    ensure_run_manifest(out / "run.json", identity)
    for video in sorted({r["video"] for r in selected}):
        if sha256(video) != complete["videos"][video]:
            raise ValueError("selected benchmark video changed")
    manifest = out / "benchmark-manifest.jsonl"
    atomic_text(manifest, "".join(json.dumps(r) + "\n" for r in selected))
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
           "TRACE_WANDB_GROUP": os.environ.get("TRACE_WANDB_GROUP", out.name)}
    trials = {}
    for arm in ("plain", "temporal"):
        for name, (size, workers) in choices.items():
            tag = f"{arm}-{name}"
            folder = out / "train" / tag
            trial_cfg = {**cfg, "microbatch_size": size, "accumulation": per_rank // size}
            if layouts is not None:
                trial_cfg["decode_workers"] = workers
            command = training_command(trial_cfg, manifest, folder, "grid")
            command += ["--temporal", "none" if arm == "plain" else "causal", "--stop-after", str(steps)]
            if recipe == "preserved":
                command += ["--preservation-teacher", "--preservation-weight", ".5", "--preservation-temperature", "2"]
            try:
                run_jobs([(tag, command, cfg["gpus"])], env, out / "logs")
                trials[tag] = trial_summary(folder, steps, warmup)
            except (RuntimeError, ValueError, OSError) as error:
                trials[tag] = {"passed": False, "error": str(error), "log": str(out / "logs" / (tag + ".log"))}
            write_json(out / "trials.json", trials)
            print("BATCH TRIAL", tag, json.dumps({k: v for k, v in trials[tag].items()
                  if k not in {"question_ids_by_update", "checkpoint"}}), flush=True)
    recommendation = recommend(trials, layouts=layouts, per_rank=per_rank)
    report = {"completed": True, "trials": trials, "recommendation": recommendation,
              "benchmark_questions": count,
              "note": "A fixed training-only subset avoids hashing the complete media set on every short trial. "
                      "Training time includes decoding, frozen vision, teacher (when enabled) and LM backward. "
                      "Warmup and checkpoint writes excluded from step timing. Memory is the maximum across ranks. "
                      "No evaluation scores or scale-training checkpoint are produced."}
    write_json(out / "batch-report.json", report)
    tracker = Tracker.start(enabled_from_env(), name=out.name + "-comparison", group=env["TRACE_WANDB_GROUP"],
        job_type="batch-benchmark", config={"recipe": recipe, "steps": steps, "warmup": warmup}, directory=out / "wandb")
    tracker.table("batch_comparison", ["arm", "layout", "seconds_per_update", "samples_per_second", "peak_gib"],
        [[arm, name, trials[f"{arm}-{name}"]["median_seconds_per_update"],
          trials[f"{arm}-{name}"]["samples_per_second"], trials[f"{arm}-{name}"]["peak_reserved_gib_max"]]
         for arm in ("plain", "temporal") for name in choices if trials[f"{arm}-{name}"]["passed"]])
    tracker.summary(recommendation)
    tracker.upload(out.name + "-report", "batch-benchmark", [out / "batch-report.json", out / "trials.json"], base=out)
    tracker.finish()
    print("BATCH BENCHMARK COMPLETE", json.dumps(recommendation), flush=True)
    if recommendation["microbatch_size"] is None:
        raise RuntimeError("no common verified configuration; inspect batch-report.json and logs")
    return recommendation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--recipe", choices=("ce", "preserved"), default="preserved")
    args = parser.parse_args()
    run(args.prepared, args.out, args.steps, args.warmup, args.recipe)


if __name__ == "__main__":
    main()
