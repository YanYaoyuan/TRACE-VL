"""Sequential core video-QA baselines; shared data, frozen protocol, resumable jobs.

This establishes original/full-trained/grid-trained references. It does not
implement T1 or claim an architectural improvement from an engineering pilot.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from trace_vl.v3.backbone import frame_size
from trace_vl.v3.run_io import ensure_run_manifest

from .control import score_answer, select_records
from .data import read_split, sha256
from .evaluate import summarize
from .prepare import digest, prepare, write_json
from .train import checked_checkpoint

ARMS = ("original_full", "original_grid", "trained_full", "trained_grid")


def paired(rows, left, right, seed=7, samples=4000):
    """Question-weighted accuracy difference, source-video cluster bootstrap."""
    groups = {}
    for row in rows:
        key = row.get("source_group", row["video"])
        score = int(score_answer(left[row["id"]]["answer"], row)["correct"])
        score -= int(score_answer(right[row["id"]]["answer"], row)["correct"])
        group = groups.setdefault(key, [0, 0])
        group[0] += score
        group[1] += 1
    if not groups:
        return None
    values = np.array(list(groups.values()))
    interval = None
    if len(values) > 1:
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(len(values), size=(samples, len(values)))].sum(axis=1)
        interval = np.quantile(100 * draws[:, 0] / draws[:, 1], [.025, .975]).tolist()
    return {"n": len(rows), "source_videos": len(groups),
            "delta_pp": 100 * values[:, 0].sum() / values[:, 1].sum(), "ci95_pp": interval,
            "bootstrap_samples": samples, "seed": seed, "unit": "source-video cluster"}


def report_experiment(manifest, out, limit):
    out = Path(out)
    rows = select_records(read_split(Path(manifest), "val"), limit)
    scores, maps = {}, {}
    for arm in ARMS:
        directory = out / "eval" / arm
        run = json.loads((directory / "run.json").read_text())
        predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        scores[arm] = summarize(rows, predictions, run["id"])
        maps[arm] = {r["id"]: r for r in predictions}
    for row in rows:
        # The pixel hash and source identity must match in all four arms, not
        # merely the question IDs. Compression changes token counts by design.
        obs = [maps[arm][row["id"]]["observation"] for arm in ARMS]
        for key in ("video_sha256", "frame_indices", "decoded_rgb_sha256", "input_ids_sha256",
                    "timestamp_policy", "timestamp_text_seconds"):
            if len({json.dumps(x[key], sort_keys=True) for x in obs}) != 1:
                raise ValueError(f"unmatched observation {key}: {row['id']}")
    comparisons = {}
    for left, right in (("trained_full", "original_full"), ("trained_grid", "original_grid"),
                        ("trained_grid", "trained_full"), ("original_grid", "original_full")):
        comparisons[f"{left}_minus_{right}"] = {
            metric: paired([r for r in rows if ("options" in r) == mc], maps[left], maps[right])
            for metric, mc in (("strict_mc_accuracy", True), ("normalized_text_exact_match", False))}
    report = {"completed": True, "scores": scores, "paired": comparisons,
              "scope": "LLaVA held-out development QA; exploratory comparisons, one training seed. "
                       "Free-text exact match is not semantic accuracy; not an official benchmark score. "
                       "No architectural improvement is established by this baseline queue."}
    write_json(out / "summary.json", report)
    return report


def shared_args(cfg, manifest):
    return ["--model", cfg["model"], "--manifest", str(manifest), "--steps", str(cfg["steps"]),
            "--pixels", str(cfg["pixels"]), "--pool-grid", *map(str, cfg["pool_grid"]),
            "--timestamp-policy", cfg["timestamp_policy"]]


def training_command(cfg, manifest, out, mode, *, flow=False):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={len(cfg['gpus'])}", "-m", "trace_vl.core_vlm.train",
            *shared_args(cfg, manifest), "--out", str(out), "--mode", mode,
            "--updates", str(cfg["flow_updates"] if flow else cfg["updates"]),
            "--accumulation", str(1 if flow else cfg["accumulation"]),
            "--rank", str(cfg["rank"]), "--lr", str(cfg["lr"]), "--seed", str(cfg["seed"]),
            "--save-every", str(cfg["save_every"]), "--cache-mb", str(cfg["cache_mb"]), "--native-check"]


def run_jobs(jobs, env, log_root):
    """Run independent shards concurrently; fail on any shard, keep all logs."""
    log_root.mkdir(parents=True, exist_ok=True)
    active, handles = [], []
    begin = last = time.monotonic()
    try:
        for name, cmd, devices in jobs:
            log = (log_root / f"{name}.log").open("ab", buffering=0)
            handles.append(log)
            child_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices))}
            proc = subprocess.Popen(cmd, env=child_env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active.append((name, proc))
            print(f"START {name} PID={proc.pid} GPUs={devices}", flush=True)
        while active:
            remaining = []
            for name, proc in active:
                code = proc.poll()
                if code is None:
                    remaining.append((name, proc))
                elif code:
                    raise RuntimeError(f"{name} exited {code}; inspect {log_root / (name + '.log')}")
                else:
                    print(f"DONE {name}", flush=True)
            active = remaining
            if time.monotonic() - last >= 30:
                print(f"RUNNING {[n for n, _ in active]} elapsed={time.monotonic() - begin:.0f}s", flush=True)
                last = time.monotonic()
            if active:
                time.sleep(.2)
    finally:
        # Only process groups started by this invocation; never user GPU jobs.
        for _, proc in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for _, proc in active:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for handle in handles:
            handle.close()


def validate_config(cfg):
    for key in ("steps", "pixels", "updates", "flow_updates", "accumulation", "rank", "save_every",
                "max_new_tokens", "flow_questions"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"positive integer required: {key}")
    if (not cfg["gpus"] or any(type(x) is not int or x < 0 for x in cfg["gpus"])
            or len(set(cfg["gpus"])) != len(cfg["gpus"])):
        raise ValueError("distinct GPU indices required")
    if (len(cfg["pool_grid"]) != 2 or any(type(x) is not int or x < 1 for x in cfg["pool_grid"])
            or cfg["eval_limit"] < 0 or cfg["cache_mb"] < 0
            or not np.isfinite(cfg["lr"]) or cfg["lr"] <= 0
            or not np.isfinite(cfg["minimum_free_gib"]) or cfg["minimum_free_gib"] < 0
            or cfg["timestamp_policy"] not in {"strict", "native-index-fps"}):
        raise ValueError("invalid experiment configuration")


def run(cfg, out, *, prepare_only=False):
    validate_config(cfg)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".experiment.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "WANDB_MODE": "offline", "TRACE_WANDB": "0", "PYTHONUNBUFFERED": "1",
               "OMP_NUM_THREADS": str(cfg.get("cpu_threads", 4)), "MKL_NUM_THREADS": str(cfg.get("cpu_threads", 4)),
               "TOKENIZERS_PARALLELISM": "false", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        print("[1/5] Prepare source-scoped video QA data", flush=True)
        data_report = prepare(cfg["data_config"], cfg["data_out"])
        manifest = Path(data_report["manifest"])
        completion = json.loads((manifest.parent / "complete.json").read_text())
        if sha256(manifest) != completion["manifest_sha256"]:
            raise ValueError("prepared manifest identity differs")
        verification = json.loads((manifest.parent / "verification.json").read_text())["videos"]
        for p, video in verification.items():
            h, w = frame_size(video["height"], video["width"], cfg["pixels"], 32)
            if video["frames"] < 2 * cfg["steps"] or h // 32 < cfg["pool_grid"][0] or w // 32 < cfg["pool_grid"][1]:
                raise ValueError(f"video cannot support declared frames/pooling grid: {p}")
        code = {str(p.relative_to(Path(__file__).parents[1])): sha256(p)
                for root in (Path(__file__).parent, Path(__file__).parents[1] / "v3")
                for p in sorted(root.glob("*.py"))}
        identity = {"protocol": "core-qa-baselines-v1", "config": cfg, "dataset": completion, "code": code}
        identity["id"] = digest(identity)
        ensure_run_manifest(out / "run.json", identity)
        if prepare_only:
            print("PREPARED: no GPU work requested", flush=True)
            return data_report
        # A subprocess observes exactly the device mapping used by torchrun.
        gpu_check = ("import torch; "
                     f"assert torch.cuda.device_count()=={len(cfg['gpus'])}; "
                     "free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(torch.cuda.device_count())]; "
                     f"print(free); assert min(free)>={cfg['minimum_free_gib']}, 'insufficient free GPU memory'")
        run_jobs([("gpu-check", [sys.executable, "-c", gpu_check], cfg["gpus"])], env, out / "logs")
        print("[2/5] Native-input parity, two short DDP QA runs, trained-checkpoint evaluation", flush=True)
        control = [sys.executable, "-m", "trace_vl.core_vlm.control", *shared_args(cfg, manifest),
                   "--out", str(out / "native-control"), "--limit", str(cfg["flow_questions"]),
                   "--max-new-tokens", str(cfg["max_new_tokens"])]
        run_jobs([("native-control", control, cfg["gpus"][:1])], env, out / "logs")

        def train(mode, flow):
            folder = out / ("flow-train" if flow else "train") / mode
            job = (f"{'flow-' if flow else ''}train-{mode}",
                   training_command(cfg, manifest, folder, mode, flow=flow), cfg["gpus"])
            run_jobs([job], env, out / "logs")
            updates = cfg["flow_updates"] if flow else cfg["updates"]
            checkpoint = folder / f"checkpoint-{updates:06d}"
            run_id = json.loads((folder / "run.json").read_text())["id"]
            if checked_checkpoint(checkpoint, run_id)["update"] != updates:
                raise ValueError("training did not reach the fixed final checkpoint")
            return checkpoint

        def evaluate(name, mode, checkpoint=None, flow=False):
            folder = out / ("flow-eval" if flow else "eval") / name
            base = [sys.executable, "-m", "trace_vl.core_vlm.evaluate", *shared_args(cfg, manifest),
                    "--out", str(folder), "--mode", mode, "--shards", str(len(cfg["gpus"])),
                    "--limit", str(cfg["flow_questions"] if flow else cfg["eval_limit"]),
                    "--max-new-tokens", str(cfg["max_new_tokens"])]
            if checkpoint:
                base += ["--checkpoint", str(checkpoint)]
            prefix = "flow-" if flow else ""
            jobs = [(f"{prefix}{name}-shard-{i}", [*base, "--shard", str(i)], [gpu])
                    for i, gpu in enumerate(cfg["gpus"])]
            run_jobs(jobs, env, out / "logs")
            run_jobs([(f"{prefix}{name}-report", [*base, "--report-only"], [])], env, out / "logs")

        for mode in ("full", "grid"):
            evaluate(mode, mode, train(mode, True), flow=True)
        print("[3/5] Original weights, full/grid controls on identical held-out questions", flush=True)
        for mode in ("full", "grid"):
            evaluate(f"original_{mode}", mode)
        print("[4/5] Full/grid QA LoRA from the same original weights; fixed final checkpoints", flush=True)
        for mode in ("full", "grid"):
            evaluate(f"trained_{mode}", mode, train(mode, False))
        print("[5/5] Matched-observation audit and paired development report", flush=True)
        report = report_experiment(manifest, out, cfg["eval_limit"])
        print(f"VERIFIED core QA baseline queue: {out / 'summary.json'}", flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.out, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
