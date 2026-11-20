"""Train matched scale arms using an audited prepared mixture and batch trial."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import statistics
from pathlib import Path

from trace_vl.v3.run_io import ensure_run_manifest

from .batch_benchmark import recommend
from .data import sha256
from .mixed_experiment import run as run_training
from .prepare import digest, write_json
from .preservation_experiment import audit_training
from .scale import queue_plan


def audit_benchmark(root):
    root = Path(root)
    report = json.loads((root / "batch-report.json").read_text())
    trials = json.loads((root / "trials.json").read_text())
    identity = json.loads((root / "run.json").read_text())
    if (report.get("completed") is not True or report["trials"] != trials
            or digest({k: v for k, v in identity.items() if k != "id"}) != identity["id"]):
        raise ValueError("batch report/provenance differs")
    reference = None
    for name in (f"{arm}-mb{size}" for arm in ("plain", "temporal") for size in (1, 2, 4)):
        trial = trials[name]
        rows = []
        for line in (root / "logs" / (name + ".log")).read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and "update" in item and "seconds_per_update" in item:
                rows.append(item)
        if (not trial["passed"] or [r["update"] for r in rows] != list(range(1, identity["steps"] + 1))
                or any(not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"])
                       or r["gradient_norm"] <= 0 or r["effective_batch_size"] != 16
                       or not math.isfinite(r["seconds_per_update"]) or r["seconds_per_update"] <= 0
                       for r in rows)):
            raise ValueError("incomplete/nonfinite batch trial: " + name)
        schedule = [(r["question_ids_by_rank"], r["lr"], r["answer_tokens"], r["teacher_questions"]) for r in rows]
        if reference is None:
            reference = schedule
        if schedule != reference or [r["question_ids_by_rank"] for r in rows] != trial["question_ids_by_update"]:
            raise ValueError("batch trial sample/token/teacher schedule differs: " + name)
        seconds = statistics.median(r["seconds_per_update"] for r in rows[identity["warmup"]:])
        if (not math.isclose(seconds, trial["median_seconds_per_update"], rel_tol=1e-9)
                or trial["first_loss"] != rows[0]["loss"]
                or trial["peak_reserved_gib_max"] != max(r["peak_reserved_gib_max"] for r in rows)):
            raise ValueError("batch trial summary differs from logs: " + name)
    recommendation = recommend(trials)
    if (recommendation != report["recommendation"] or recommendation["microbatch_size"] != 4
            or len({t["initial_lora"] for t in trials.values()}) != 1):
        raise ValueError("this scale queue requires the reviewed batch-4 recommendation")
    return identity, recommendation


def configure(prepared, benchmark, out, reviewed_pairs=None):
    prepared, benchmark, out = (Path(p).resolve() for p in (prepared, benchmark, out))
    cfg = json.loads((prepared / "config.json").read_text())
    preparation = json.loads((prepared / "preparation.json").read_text())
    if (preparation["config"] != cfg
            or digest({k: v for k, v in preparation.items() if k != "id"}) != preparation["id"]):
        raise ValueError("prepared configuration/provenance differs")
    inventory = json.loads((prepared / "mixture/inventory.json").read_text())
    plan = queue_plan(cfg, inventory)
    if len(cfg["gpus"]) != 4 or plan["effective_batch"] != 16:
        raise ValueError("this queue preserves global batch 16 on four GPUs")
    trial, batch = audit_benchmark(benchmark)
    if (trial["config"] != cfg or trial["recipe"] != "preserved"
            or trial["prepared_complete_sha256"] != sha256(prepared / "mixture/complete.json")
            or trial["manifest_sha256"] != sha256(prepared / "mixture/manifest.jsonl")):
        raise ValueError("benchmark and prepared training data differ")
    for name in ("train.py", "qa.py", "native.py", "temporal.py", "preservation.py"):
        if trial["code"][name] != sha256(Path(__file__).parent / name):
            raise ValueError("benchmarked training implementation changed: " + name)
    if trial["backbone_sha256"] != sha256(Path(__file__).parents[1] / "v3/backbone.py"):
        raise ValueError("benchmarked backbone implementation changed")
    pilot = Path(cfg["pilot_run"])
    report = json.loads((pilot / "report.json").read_text())
    pilot_cfg = json.loads((pilot / "experiment-config.json").read_text())
    if (report.get("completed") is not True or report.get("training_audit", {}).get("passed") is not True
            or audit_training(pilot / "train", pilot_cfg["updates"]) != report["training_audit"]):
        raise ValueError("preservation pilot checkpoint audit differs")
    for key in ("model", "steps", "pixels", "pool_grid", "timestamp_policy"):
        if cfg[key] != pilot_cfg[key]:
            raise ValueError("pilot observation/backbone differs: " + key)
    training = {**cfg, "mixture_data": str(prepared / "mixture"), "microbatch_size": 4,
                "accumulation": 1, "preservation_weight": .5,
                "reviewed_pairs": str(Path(reviewed_pairs).resolve()) if reviewed_pairs else None}
    record = {"prepared": str(prepared), "benchmark": str(benchmark), "batch": batch,
              "pilot_report_sha256": sha256(pilot / "report.json"), "pilot_decision": report["decision"],
              "prepared_complete_sha256": sha256(prepared / "mixture/complete.json"),
              "benchmark_report_sha256": sha256(benchmark / "batch-report.json"),
              "training_config": training, "reviewed_pairs_sha256": sha256(reviewed_pairs) if reviewed_pairs else None,
              "code": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
              "review": "Owner-authorized controlled expansion after the completed preservation pilot. "
                        "Existing architecture/capability gates failed. Both arms start from the original model. "
                        "This run tests expansion; it does not establish the temporal contribution."}
    record["id"] = digest(record)
    out.mkdir(parents=True, exist_ok=True)
    ensure_run_manifest(out / "release.json", record)
    write_json(out / "experiment-config.json", training)
    print("SCALE CONFIGURED", json.dumps({"questions": sum(inventory["families"].values()),
          "microbatch_size": 4, "accumulation": 1, "effective_batch": 16,
          "updates_per_arm": training["updates"], "reviewed_pairs": training["reviewed_pairs"]}), flush=True)
    return training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reviewed-pairs")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".scale-queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = configure(args.prepared, args.benchmark, out, args.reviewed_pairs)
        run_training(cfg, out, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
