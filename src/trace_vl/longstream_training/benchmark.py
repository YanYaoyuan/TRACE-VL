"""Same-GPU, same-video cache throughput and exact-feature regression."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.prepare import digest, write_json

from .prepare import rows


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / "data"
    data.mkdir()
    selected = {}
    for split, count in (("train", 4), ("dev", 2)):
        selected[split] = rows(args.data / f"{split}.jsonl")[:count]
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected[split]))
    # Both implementations consume shard0: exactly three of these six sources.
    audit = {
        "completed": True,
        "manifest_sha256": {s: sha256(data / f"{s}.jsonl") for s in selected},
        "scope": "Performance diagnostic only; fixed first sources, no accuracy or outcome selection.",
    }
    audit["input_id"] = digest(audit)
    write_json(data / "data-audit.json", audit)
    sources = {"baseline": args.baseline_source, "prefetch": Path(__file__).resolve().parents[3]}
    measurements = {}
    for name, source in sources.items():
        out = args.out / name
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(source / "src"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        stopped, samples = threading.Event(), []

        def sample():
            while not stopped.is_set():
                value = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={args.gpu}",
                        "--query-gpu=utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                samples.append([float(v.strip()) for v in value.strip().split(",")])
                stopped.wait(1)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            with (args.out / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "trace_vl.longstream_training.cache",
                        "--data",
                        str(data),
                        "--out",
                        str(out),
                        "--model",
                        str(args.model),
                        "--shard",
                        "0",
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        finally:
            stopped.set()
            thread.join()
        measurements[name] = {
            "wall_seconds": time.monotonic() - started,
            "gpu_samples": samples,
            "source": str(source),
            "gpu": args.gpu,
        }
        print(json.dumps({"arm": name, "wall_seconds": measurements[name]["wall_seconds"]}), flush=True)
    ids = (selected["train"] + selected["dev"])[::2]
    for row in ids:
        values = [
            torch.load(args.out / name / f"{digest(row['id'])}.pt", weights_only=True) for name in sources
        ]
        for v in values:
            v.pop("input_id")
        tensors = [v.pop("features") for v in values]
        if not torch.equal(*tensors) or values[0] != values[1]:
            raise ValueError("prefetched features or causal/native metadata differ from baseline")
    write_json(
        args.out / "report.json",
        {
            "completed": True,
            "sources": len(ids),
            "gpu": args.gpu,
            "exact_features_and_all_causal_metadata": True,
            "measurements": measurements,
            "wall_speedup": measurements["baseline"]["wall_seconds"]
            / measurements["prefetch"]["wall_seconds"],
            "scope": "One baseline then prefetch pass; includes model load/hash/checks, short-run estimate, not a training-speed claim.",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "data", "model", "baseline-source"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gpu", type=int, default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
