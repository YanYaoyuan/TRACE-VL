"""Train a causal patch compressor against a matched 128-token grid baseline."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import select_records
from .data import read_split, sha256
from .experiment import run_jobs, shared_args, training_command
from .mixed_data import frozen_external
from .mixed_experiment import audit_training, matched_report
from .prepare import digest, write_json
from .transfer import continuation_gate

ARMS = {"plain": "none", "causal_pool": "causal-pool"}
CONTRAST = "causal_pool_minus_plain"


def run(previous, local_data, out, updates, smoke, *, feature_cache=None, cache_check=False):
    receipt = json.loads((local_data / "complete.json").read_text())
    manifest = local_data / "manifest.jsonl"
    previous = Path(previous)
    original = previous / "mixture/manifest.jsonl"
    if sha256(manifest) != receipt["local_sha256"] or sha256(original) != receipt["original_sha256"]:
        raise ValueError("verified local training manifest or its source changed")
    cfg = json.loads((previous / "experiment-config.json").read_text())
    external, audit = frozen_external(cfg["external_data"], verify_media=not smoke)
    receipt_match = json.loads((previous / "mixture/inputs.json").read_text())["external"]
    if receipt_match != audit:
        raise ValueError("external development manifests changed since the 40K overlap audit")
    cfg.update(pool_grid=[4, 4], updates=updates)
    budget = cfg["steps"] * 16
    if budget != 128:
        raise ValueError("the comparison requires 128 visual tokens from eight steps")
    out.mkdir(parents=True, exist_ok=True)
    identity = {"config": cfg, "manifest_sha256": receipt["local_sha256"], "external": audit,
                "smoke": smoke, "cache_check": cache_check, "feature_cache": feature_cache,
                "revision": os.environ.get("TRACE_REV", "local"),
                "arms": ARMS, "implementation": sha256(__file__)}
    identity["id"] = digest(identity)
    with (out / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_run_manifest(out / "experiment.json", identity)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        gpu_check = ("import torch; "
                     f"assert torch.cuda.device_count()=={len(cfg['gpus'])}; "
                     "free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(torch.cuda.device_count())]; "
                     f"print(free); assert min(free)>={cfg['minimum_free_gib']}")
        logs = out / "logs"
        run_jobs([("gpu-check", [sys.executable, "-c", gpu_check], cfg["gpus"])], env, logs)
        if smoke and not cache_check:
            for dataset, path in external.items():
                subset = select_records(read_split(path, "val"), 4)
                for video in {row["video"] for row in subset}:
                    if sha256(video) != audit["videos"][video]:
                        raise ValueError("smoke evaluation video differs from the frozen benchmark")
                external[dataset] = out / f"smoke-{dataset}.jsonl"
                atomic_text(external[dataset], "".join(json.dumps(row) + "\n" for row in subset))
        for name, arm in ARMS.items():
            folder = out / "train" / name
            cmd = training_command(cfg, manifest, folder, "grid", flow=smoke)
            cmd += ["--temporal", arm]
            if feature_cache:
                cmd += ["--feature-cache", feature_cache]
            if smoke:
                cmd += ["--examples", "256"]
            if cfg["preservation_weight"]:
                cmd += ["--preservation-teacher", "--preservation-weight", str(cfg["preservation_weight"])]
            run_jobs([(f"train-{name}", cmd, cfg["gpus"])], env, logs)
        checkpoint_step = cfg["flow_updates"] if smoke else cfg["updates"]
        training = audit_training(out / "train", checkpoint_step, arms=ARMS)
        if cache_check:
            last = (out / "train/causal_pool/metrics.jsonl").read_text().splitlines()[-1]
            metrics = json.loads(last)
            hits = metrics["vision_cache_hits_rank0"]
            writes = metrics["vision_cache_writes_rank0"]
            if not hits or writes:
                raise ValueError("second arm did not reuse the first arm's cached observations")
            write_json(out / "report.json", {"completed": True, "cache_check": True,
                       "visual_tokens": budget, "training": training,
                       "vision_cache_hits_rank0": hits, "vision_cache_writes_rank0": writes})
            print(f"VERIFIED SHARED CACHE TRAINING: {out / 'report.json'}", flush=True)
            return
        for dataset, path in external.items():
            for name in ARMS:
                checkpoint = out / "train" / name / f"checkpoint-{checkpoint_step:06d}"
                directory = out / "eval" / dataset / name
                cmd = [sys.executable, "-m", "trace_vl.core_vlm.evaluate", *shared_args(cfg, path),
                       "--mode", "grid", "--out", str(directory), "--checkpoint", str(checkpoint),
                       "--shards", str(len(cfg["gpus"])), "--mc-choice-score",
                       "--max-new-tokens", str(cfg["max_new_tokens"])]
                tag = f"{dataset}-{name}"
                run_jobs([(f"{tag}-{i}", [*cmd, "--shard", str(i)], [gpu])
                          for i, gpu in enumerate(cfg["gpus"])], env, logs)
                run_jobs([(f"{tag}-report", [*cmd, "--report-only"], [])], env, logs)
        results = matched_report(external, out / "eval", tuple(ARMS), [("causal_pool", "plain")])
        decision = None if smoke else continuation_gate(
            {dataset: info["paired"][CONTRAST] for dataset, info in results.items()},
            {"gain_pp": 2., "noninferiority_pp": 2.})
        report = {"completed": True, "smoke": smoke, "visual_tokens": budget,
                  "training": training, "results": results, "decision": decision}
        write_json(out / "report.json", report)
        print(f"VERIFIED CAUSAL POOL {'SMOKE' if smoke else 'COMPARISON'}: {out / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--local-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--feature-cache")
    parser.add_argument("--cache-check", action="store_true")
    args = parser.parse_args()
    if args.cache_check and not args.feature_cache:
        parser.error("--cache-check needs --feature-cache")
    run(args.previous, args.local_data, args.out, args.updates, args.smoke or args.cache_check,
        feature_cache=args.feature_cache, cache_check=args.cache_check)


if __name__ == "__main__":
    main()
