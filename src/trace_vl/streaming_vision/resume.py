"""Resume evaluation against an existing, verified streaming vision checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import transformers

from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.prepare import digest
from trace_vl.v3.run_io import ensure_run_manifest

from .data import read_rows
from .experiment import evaluate, load_prepared, report
from .media import DECODE_POLICY


def verify(root, revision):
    prepared = load_prepared(root)
    inputs = json.loads((root / "inputs.json").read_text())
    input_id = inputs.pop("id")
    if digest(inputs) != input_id:
        raise ValueError("training input identity has changed")
    receipt = json.loads((root / "training-complete.json").read_text())
    if receipt != {"input_id": input_id, "weights_sha256": sha256(root / "correction.pt")}:
        raise ValueError("trained correction receipt does not match")
    source = root / "source"
    base_revision = (source / ".revision").read_text().strip()
    current = Path(__file__).resolve().parents[1]
    changes = {}
    allowed = {"core_vlm/data.py", "streaming_vision/experiment.py"}
    for name, expected in inputs["code"].items():
        if sha256(source / "src/trace_vl" / name) != expected:
            raise ValueError(f"training source changed: {name}")
        actual = sha256(current / name)
        if actual != expected:
            if name not in allowed:
                raise ValueError(f"evaluation would change an existing model or data component: {name}")
            changes[name] = actual
    if str(torch.__version__) != inputs["torch"] or transformers.__version__ != inputs["transformers"]:
        raise ValueError("use the Torch and Transformers environment from the original training run")
    if checkpoint_identity(inputs["config"]["model"]) != inputs["model"]:
        raise ValueError("base Qwen weights or tokenizer changed")
    if inputs["config"]["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is required by the original evaluation")
    saved = root / "predictions"
    expected_ids = {
        row["id"] for name in prepared["manifests"] if name != "train"
        for row in read_rows(root / f"{name}.jsonl")
    }
    for file in saved.glob("*.json"):
        data = json.loads(file.read_text())
        if (data["run"] != input_id or data["correction"] != receipt["weights_sha256"]
                or data["id"] not in expected_ids or file.stem != digest(data["id"])):
            raise ValueError(f"prediction does not belong to the trained run: {file}")
    fingerprint = {"training_revision": base_revision, "evaluation_revision": revision,
                   "input_id": input_id, "weights_sha256": receipt["weights_sha256"], "decode_policy": DECODE_POLICY,
                   "modified_code": changes,
                   "added_code": {name: sha256(current / "streaming_vision" / name)
                                  for name in ("media.py", "resume.py")}}
    fingerprint["id"] = digest(fingerprint)
    ensure_run_manifest(root / "eval-recovery.json", fingerprint)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("verify", "eval", "report"))
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=4)
    args = p.parse_args()
    root = args.run.resolve()
    if args.stage == "verify":
        verify(root, args.revision)
        saved = len(list((root / "predictions").glob("*.json")))
        print(f"REUSED training checkpoint and {saved} complete question records", flush=True)
        return
    repair = json.loads((root / "eval-recovery.json").read_text())
    if repair["evaluation_revision"] != args.revision:
        raise ValueError("evaluation recovery revision differs")
    config = json.loads((root / "inputs.json").read_text())["config"]
    settings = argparse.Namespace(**{**config, "shard": args.shard, "gpus": args.shards,
                                     "workers": int(os.environ.get("TRACE_EVAL_DECODE_WORKERS", 2))})
    if settings.workers < 1 or not 0 <= settings.shard < settings.gpus:
        raise ValueError("invalid shard or worker count")
    if args.stage == "eval":
        device = torch.device("cuda:0" if settings.device == "cuda" else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        evaluate(settings, root, device)
    else:
        report(settings, root)


if __name__ == "__main__":
    main()
