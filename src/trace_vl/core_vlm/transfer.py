"""External transfer gate followed by a matched core-model temporal pilot.

All populations/configurations are frozen before GPU work. The published OVO
final half and NExT-QA test set are never opened. T0 is an ordinary baseline.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import score_answer, select_records
from .data import read_split, sha256
from .evaluate import summarize
from .experiment import ARMS, paired, run_jobs, shared_args, training_command
from .prepare import digest, write_json
from .train import checked_checkpoint
from .transfer_data import prepare

ENDPOINTS = {"ovo": {"primary": ("EPM", "ASI", "HLD"),
                      "no_harm": ("OCR", "ACR", "ATR", "STU", "FPD", "OJR")},
             "nextqa": {"primary": ("TN", "TP", "TC"), "no_harm": ("DC", "DL", "DO")}}


def collect(rows, directory):
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text())
    results = [json.loads(x) for x in (directory / "predictions.jsonl").read_text().splitlines()]
    summarize(rows, results, run["id"])
    mapped = {x["id"]: x for x in results}
    for r in rows:
        choice = mapped[r["id"]].get("choice")
        if type(choice) is not int or not 0 <= choice < len(r["options"]):
            raise ValueError("missing first-token option control; rerun in a new evaluation directory")
    return mapped


def audit_observations(rows, maps):
    keys = ("video_sha256", "frame_indices", "decoded_rgb_sha256", "input_ids_sha256",
            "timestamp_policy", "timestamp_text_seconds")
    for r in rows:
        observations = [v[r["id"]]["observation"] for v in maps.values()]
        if any(len({json.dumps(o[k], sort_keys=True) for o in observations}) != 1 for k in keys):
            raise ValueError(f"unmatched observations: {r['id']}")
        if r.get("query_time") is not None:
            if any(t > r["query_time"] for pair in observations[0]["frame_times"] for t in pair):
                raise ValueError("future frames in external evaluation")


def choice_map(predictions):
    return {k: {**v, "answer": chr(65 + v["choice"])} for k, v in predictions.items()}


def stats(rows, predictions):
    n = len(rows)
    if not n:
        return {"n": 0}
    valid = sum(score_answer(predictions[r["id"]]["answer"], r)["parsed"] is not None for r in rows)
    return {"n": n,
            "choice_accuracy": sum(predictions[r["id"]]["choice"] == r["answer"] for r in rows) / n,
            "strict_accuracy": sum(score_answer(predictions[r["id"]]["answer"], r)["correct"] for r in rows) / n,
            "format_valid_fraction": valid / n}


def comparison(rows, left, right, dataset):
    sets = {"all": rows, **{key: [r for r in rows if r["task"] in tasks]
                           for key, tasks in ENDPOINTS[dataset].items()}}
    sets.update({"task/" + t: [r for r in rows if r["task"] == t] for t in sorted({r["task"] for r in rows})})
    return {key: {"choice": paired(subset, choice_map(left), choice_map(right)),
                  "strict": paired(subset, left, right)} for key, subset in sets.items()}


def continuation_gate(comparisons, rule):
    checks = {}
    for dataset in ENDPOINTS:
        values = comparisons[dataset]
        gain, harm = values["primary"]["choice"], values["no_harm"]["choice"]
        checks[dataset] = bool(gain and harm and gain["ci95_pp"] and harm["ci95_pp"]
                               and gain["delta_pp"] >= rule["gain_pp"]
                               and gain["ci95_pp"][0] > 0
                               and harm["ci95_pp"][0] > -rule["noninferiority_pp"])
    return {"passed": all(checks.values()), "checks": checks, "rule": rule,
            "meaning": "Advance only if BOTH primary endpoints improve and both no-harm checks pass; "
                       "failure can mean insufficient precision, not necessarily degradation."}


def external_report(manifests, root, names, contrasts):
    results = {}
    for dataset, manifest in manifests.items():
        rows = read_split(manifest, "val")
        maps = {n: collect(rows, Path(root) / dataset / n) for n in names}
        audit_observations(rows, maps)
        results[dataset] = {"scores": {n: stats(rows, m) for n, m in maps.items()},
                            "paired": {f"{left}_minus_{right}": comparison(rows, maps[left], maps[right], dataset)
                                       for left, right in contrasts}}
    return results


def run(cfg, out, *, prepare_only=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".experiment.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base = Path(cfg["baseline_run"])
        old = json.loads((base / "experiment-config.json").read_text())
        for key in ("model", "steps", "pixels", "pool_grid", "timestamp_policy", "rank"):
            if cfg[key] != old[key]:
                raise ValueError(f"transfer protocol must match completed QA baseline: {key}")
        summary = json.loads((base / "summary.json").read_text())
        if not summary["completed"]:
            raise ValueError("baseline queue is incomplete")
        checkpoints = {}
        for mode in ("full", "grid"):
            training = json.loads((base / f"train/{mode}/run.json").read_text())
            path = base / f"train/{mode}/checkpoint-{old['updates']:06d}"
            complete = checked_checkpoint(path, training["id"])
            if complete["update"] != old["updates"]:
                raise ValueError("not the fixed baseline final checkpoint")
            checkpoints[mode] = path
        code = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
        identity = {"config": cfg, "code": code, "baseline_summary": sha256(base / "summary.json"),
                    "checkpoints": {k: sha256(v / "complete.json") for k, v in checkpoints.items()}}
        identity["id"] = digest(identity)
        ensure_run_manifest(out / "run.json", identity)
        print("[1/5] Frozen external populations, selective extraction, overlap/full-decode audit", flush=True)
        manifests = prepare(cfg, cfg["data_out"])
        if prepare_only:
            print("PREPARED: no GPU work", flush=True)
            return
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline",
               "TRACE_WANDB": "0", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        gpu_check = ("import torch; "
                     f"assert torch.cuda.device_count()=={len(cfg['gpus'])}; "
                     "free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(torch.cuda.device_count())]; "
                     f"print(free); assert min(free)>={cfg['minimum_free_gib']}")
        run_jobs([("gpu-check", [sys.executable, "-c", gpu_check], cfg["gpus"])], env, out / "logs")

        def evaluate(manifest, name, mode, checkpoint, directory):
            dest = directory / name
            command = [sys.executable, "-m", "trace_vl.core_vlm.evaluate", *shared_args(cfg, manifest),
                       "--out", str(dest), "--mode", mode, "--shards", str(len(cfg["gpus"])),
                       "--max-new-tokens", str(cfg["max_new_tokens"]), "--mc-choice-score"]
            if checkpoint:
                command += ["--checkpoint", str(checkpoint)]
            tag = directory.name + "-" + name
            run_jobs([(f"{tag}-shard-{i}", [*command, "--shard", str(i)], [gpu])
                      for i, gpu in enumerate(cfg["gpus"])], env, out / "logs")
            run_jobs([(f"{tag}-report", [*command, "--report-only"], [])], env, out / "logs")

        print("[2/5] Four-question real-model/checkpoint flow per dataset", flush=True)
        for dataset, manifest in manifests.items():
            rows = select_records(read_split(manifest, "val"), 4)
            tiny = out / f"flow-{dataset}.jsonl"
            atomic_text(tiny, "".join(json.dumps(r) + "\n" for r in rows))
            for mode in ("full", "grid"):
                evaluate(tiny, f"trained_{mode}", mode, checkpoints[mode], out / "flow" / dataset)

        print("[3/5] Original/trained x full/grid external QA; fixed 16-frame observations", flush=True)
        for dataset, manifest in manifests.items():
            for name in ARMS:
                mode = name.split("_")[1]
                checkpoint = checkpoints[mode] if name.startswith("trained") else None
                evaluate(manifest, name, mode, checkpoint, out / "eval" / dataset)
        contrasts = [("trained_grid", "original_grid"), ("trained_full", "original_full"),
                     ("trained_grid", "trained_full"), ("original_grid", "original_full")]
        result = external_report(manifests, out / "eval", ARMS, contrasts)
        gate = continuation_gate({d: r["paired"]["trained_grid_minus_original_grid"]
                                  for d, r in result.items()}, cfg["gate"])
        report = {"completed": True, "scope": "external development; one seed; fixed observations; "
                  "not an official OVO streaming score, not an architecture claim",
                  "primary_scoring": "highest first-answer-token logit among option letters; "
                                     "strict free generation and format validity also reported",
                  "results": result, "next_training_gate": gate}
        write_json(out / "transfer-report.json", report)
        print(json.dumps(gate), flush=True)
        if not gate["passed"]:
            write_json(out / "queue-status.json", {"transfer_complete": True, "training_started": False,
                       "reason": "predeclared external transfer/no-harm criteria not met", "gate": gate})
            print(f"TRANSFER COMPLETE; NEXT TRAINING HELD: {out / 'transfer-report.json'}", flush=True)
            return
        print("[4/5] Core-model continuation: plain grid vs ordinary causal temporal adapter (T0)", flush=True)
        train_manifest = Path(old["data_out"]) / "manifest.jsonl"
        next_checkpoints = {}
        for name, temporal in (("continued_grid", "none"), ("temporal_grid", "causal")):
            def train(flow):
                dest = out / ("flow-train" if flow else "train") / name
                command = training_command(cfg, train_manifest, dest, "grid", flow=flow)
                command += ["--init-checkpoint", str(checkpoints["grid"]), "--temporal", temporal]
                run_jobs([(f"{'flow-' if flow else ''}train-{name}", command, cfg["gpus"])], env, out / "logs")
                updates = cfg["flow_updates"] if flow else cfg["updates"]
                path = dest / f"checkpoint-{updates:06d}"
                complete = checked_checkpoint(path, json.loads((dest / "run.json").read_text())["id"])
                if complete["update"] != updates:
                    raise ValueError("continuation checkpoint is not final")
                return path
            short = train(True)
            evaluate(out / "flow-nextqa.jsonl", name, "grid", short, out / "continuation-flow")
            next_checkpoints[name] = train(False)
        print("[5/5] Internal/external reevaluation of both fixed final checkpoints", flush=True)
        for dataset, manifest in {"internal": train_manifest, **manifests}.items():
            for name, checkpoint in next_checkpoints.items():
                evaluate(manifest, name, "grid", checkpoint, out / "next-eval" / dataset)
        next_report = external_report(manifests, out / "next-eval", tuple(next_checkpoints),
                                      [("temporal_grid", "continued_grid")])
        # Also compare BOTH continuation arms against the untouched pre-continuation grid model.
        for dataset, manifest in manifests.items():
            rows = read_split(manifest, "val")
            reference = collect(rows, out / "eval" / dataset / "trained_grid")
            for name in next_checkpoints:
                candidate = collect(rows, out / "next-eval" / dataset / name)
                audit_observations(rows, {"reference": reference, "candidate": candidate})
                next_report[dataset]["paired"][name + "_minus_trained_grid"] = comparison(rows, candidate, reference, dataset)
        internal_rows = read_split(train_manifest, "val")
        internal = {n: collect(internal_rows, out / "next-eval/internal" / n) for n in next_checkpoints}
        audit_observations(internal_rows, internal)
        write_json(out / "next-stage-report.json", {"completed": True, "results": next_report,
                   "internal": {n: stats(internal_rows, p) for n, p in internal.items()},
                   "scope": "T0 is an ordinary temporal baseline, not T1; no automatic scale-up or best-checkpoint selection"})
        write_json(out / "queue-status.json", {"transfer_complete": True, "training_started": True,
                                               "next_stage_complete": True})
        print(f"VERIFIED TRANSFER AND CORE TEMPORAL PILOT: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    run(json.loads(Path(args.config).read_text()), args.out, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
