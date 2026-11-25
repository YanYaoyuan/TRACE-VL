"""Data -> real gradient/reload smoke -> complete cache -> matched two-arm pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.experiment import run_jobs
from trace_vl.core_vlm.prepare import write_json

from .prepare import rows


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRACE_WANDB": "0",
        "TRACE_SWANLAB": "0",
    }
    logs = args.out / "logs"
    run_jobs(
        [
            (
                "data",
                [sys.executable, "-m", "trace_vl.longstream_training.prepare", "--out", str(args.data)],
                [],
            )
        ],
        env,
        logs,
    )
    cache = args.out / "features"
    cc = [
        sys.executable,
        "-m",
        "trace_vl.longstream_training.cache",
        "--data",
        str(args.data),
        "--out",
        str(cache),
        "--model",
        str(args.model),
    ]
    run_jobs([(f"cache-smoke-{i}", [*cc, "--shard", str(i), "--smoke"], [i]) for i in range(2)], env, logs)
    tc = [
        sys.executable,
        "-m",
        "trace_vl.longstream_training.train",
        "--data",
        str(args.data),
        "--cache",
        str(cache),
        "--model",
        str(args.model),
        "--adapter",
        str(args.adapter),
        "--updates",
        str(args.updates),
    ]
    arms = ("reservoir", "learned")

    def commands(extra):
        return [
            (arm, [*tc, "--arm", arm, "--out", str(args.out / arm), *extra], [i])
            for i, arm in enumerate(arms)
        ]

    run_jobs(commands(["--stop-after", "2"]), env, logs / "smoke")
    run_jobs(commands(["--stop-after", "2"]), env, logs / "reload")
    metrics = {
        a: [json.loads(s) for s in (args.out / a / "metrics.jsonl").read_text().splitlines()] for a in arms
    }
    for a, m in metrics.items():
        if len(m) != 2 or any(r["lm_gradient"] <= 0 for r in m):
            raise ValueError("real LM smoke incomplete")
        if a == "learned" and any(r["selector_gradient"] <= 0 for r in m):
            raise ValueError("real selector smoke failed")
        if not json.loads((args.out / a / "restore-audit.json").read_text())["completed"]:
            raise ValueError("reload failed")
    if [r["question_ids"] for r in metrics["reservoir"]] != [r["question_ids"] for r in metrics["learned"]]:
        raise ValueError("matched sample prefix changed")
    write_json(
        args.out / "smoke-ok.json",
        {
            "completed": True,
            "updates_per_arm": 2,
            "same_full200_update_recipe_prefix": True,
            "real_answer_gradients_and_reload": True,
            "accuracy_gate": False,
        },
    )
    run_jobs([(f"cache-full-{i}", [*cc, "--shard", str(i)], [i]) for i in range(2)], env, logs)
    for row in rows(args.data / "train.jsonl") + rows(args.data / "dev.jsonl"):
        from trace_vl.core_vlm.prepare import digest

        path = cache / f"{digest(row['id'])}.pt"
        proof = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != proof["sha256"]:
            raise ValueError("full feature cache integrity failed")
    run_jobs(commands([]), env, logs / "full")
    metrics = {
        a: [json.loads(s) for s in (args.out / a / "metrics.jsonl").read_text().splitlines()] for a in arms
    }
    if any(len(m) != args.updates for m in metrics.values()):
        raise ValueError("training incomplete")
    if [r["question_ids"] for r in metrics["reservoir"]] != [r["question_ids"] for r in metrics["learned"]]:
        raise ValueError("training sample orders differ")
    predictions = {a: json.loads((args.out / a / "dev-predictions.json").read_text()) for a in arms}
    expected = {r["id"] for r in rows(args.data / "dev.jsonl")}
    for a, ps in predictions.items():
        if len(ps) != len(expected) or {p["id"] for p in ps} != expected:
            raise ValueError("dev coverage changed")
        if any(p["visual_tokens"] > 128 for p in ps):
            raise ValueError("dev read budget exceeded")
    right = {p["id"]: p for p in predictions["learned"]}
    d = np.array(
        [
            int(right[p["id"]]["choice"] == p["gold"]) - int(p["choice"] == p["gold"])
            for p in predictions["reservoir"]
        ]
    )
    bootstrap = np.random.default_rng(7).choice(d, (4000, len(d)), replace=True).mean(1)
    write_json(
        args.out / "report.json",
        {
            "completed": True,
            "updates_per_arm": args.updates,
            "source_videos": len(d),
            "questions": len(d),
            "predictions": 2 * len(d),
            "correct": {a: sum(p["choice"] == p["gold"] for p in ps) for a, ps in predictions.items()},
            "learned_minus_trained_reservoir_pp": float(100 * d.mean()),
            "ci95_pp": (100 * np.quantile(bootstrap, [0.025, 0.975])).tolist(),
            "scope": "Single-seed pilot on previously used ActivityNet development pool; not official or untouched final evaluation.",
            "automatic_scale_up": False,
            "novelty_claim": False,
        },
    )


def main():
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "out", "model", "adapter"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--updates", type=int, default=200)
    run(p.parse_args())


if __name__ == "__main__":
    main()
