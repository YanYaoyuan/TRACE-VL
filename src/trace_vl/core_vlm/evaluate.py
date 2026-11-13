"""Sharded held-out video QA evaluation for original and trained core models.

Native observation/position protocol, fresh state for every question. No final
test-set entry point yet: the development evaluator must not open frozen final
benchmarks as a side effect of a training queue.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import torch

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import checkpoint_identity, score_answer, select_records
from .data import decode_observation, read_split, sha256
from .native import NativeVideoQALoss, encode_native, prepare_native
from .temporal import attach_temporal, load_temporal
from .train import checked_checkpoint


def summarize(rows, predictions, run_id):
    expected = {r["id"]: r for r in rows}
    if len(expected) != len(rows) or len(predictions) != len(rows):
        raise ValueError("evaluation coverage differs")
    if {r["id"] for r in predictions} != expected.keys():
        raise ValueError("evaluation IDs differ")
    by_metric = {}
    for result in predictions:
        if result["run"] != run_id:
            raise ValueError("prediction run identity differs")
        score = score_answer(result["answer"], expected[result["id"]])
        # Recompute from generated text and the frozen manifest, never trust a
        # saved correct flag. These scores use strict answer parsing.
        item = by_metric.setdefault(score["metric"], {"n": 0, "correct": 0})
        item["n"] += 1
        item["correct"] += score["correct"]
    for item in by_metric.values():
        item["score"] = item["correct"] / item["n"]
    return {"completed": True, "run": run_id, "n": len(rows),
            "videos": len({r["video"] for r in rows}), "scores": by_metric,
            "note": "Development evaluation; exact match is not semantic QA accuracy. "
                    "No-cache greedy generation is not deployment throughput."}


@torch.no_grad()
def run(args):
    import peft
    import transformers
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.max_new_tokens < 1:
        raise ValueError("invalid shard/generation configuration")
    rows = select_records(read_split(Path(args.manifest), args.split), args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    original = checkpoint_identity(args.model)
    adapter_identity = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        training = json.loads((checkpoint.parent / "run.json").read_text())
        if training["model"] != original:
            raise ValueError("evaluation base weights differ from training")
        adapter_identity = checked_checkpoint(checkpoint, training["id"])
        if training["config"]["mode"] != args.mode or training["config"]["pool_grid"] != args.pool_grid:
            raise ValueError("training/evaluation compression differs; declare a separate ablation protocol")
    configuration = {k: v for k, v in vars(args).items() if k not in {"out", "shard", "report_only", "device"}}
    identity = {"protocol": "core-native-qa-eval-v1", "config": configuration,
                "torch": str(torch.__version__), "transformers": transformers.__version__, "peft": peft.__version__,
                "manifest_sha256": sha256(args.manifest), "ids": [r["id"] for r in rows],
                "model": original, "adapter": adapter_identity,
                "videos": {p: sha256(p) for p in sorted({r["video"] for r in rows})},
                "code": {name: sha256(Path(__file__).parent / name) for name in
                         ("evaluate.py", "native.py", "qa.py", "data.py", "control.py", "train.py", "temporal.py",
                          "../v3/backbone.py", "../v3/positions.py", "../v3/run_io.py")}}
    identity["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ensure_run_manifest(out / "run.json", identity)
    units = out / "questions"
    units.mkdir(exist_ok=True)

    def question_path(row):
        return units / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")

    if args.report_only:
        missing = [r["id"] for r in rows if not question_path(r).is_file()]
        if missing:
            raise ValueError(f"incomplete evaluation: {len(missing)} questions missing; examples {missing[:3]}")
        predictions = [json.loads(question_path(r).read_text()) for r in rows]
        report = summarize(rows, predictions, identity["id"])
        atomic_text(out / "predictions.jsonl", "".join(json.dumps(r) + "\n" for r in predictions))
        atomic_text(out / "report.json", json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    with (out / f".shard-{args.shard}.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = rows[args.shard::args.shards]
        pending = []
        for row in selected:
            path = question_path(row)
            if path.is_file():
                saved = json.loads(path.read_text())
                if saved["run"] != identity["id"] or saved["id"] != row["id"]:
                    raise ValueError("cached prediction identity differs")
            else:
                pending.append(row)
        if pending:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(7)
            bb = Backbone(BackboneConfig(name=args.model, dtype="bfloat16", local_files_only=True),
                          StreamConfig(pixels_per_frame=args.pixels), device=args.device)
            if args.checkpoint:
                config = training["config"]
                bb.add_lora(config["rank"], 2 * config["rank"], dropout=0., adapter=str(checkpoint / "adapter"))
            bb.model.eval()
            bb.model.requires_grad_(False)
            net = NativeVideoQALoss(bb, mode=args.mode, pool_grid=tuple(args.pool_grid),
                                   answer_end_token_id=bb.tokenizer.eos_token_id)
            temporal = training["config"].get("temporal", "none") if args.checkpoint else "none"
            attach_temporal(net, temporal)
            if temporal != "none":
                load_temporal(net, checkpoint)
            net.eval()
            for index, row in enumerate(pending):
                images, times, record, audit = decode_observation(row, steps=args.steps, pixels=args.pixels)
                layout = prepare_native(bb.processor, images, times, record, audit,
                                        allow_native_timestamp_approximation=args.timestamp_policy == "native-index-fps")
                steps = encode_native(bb, layout)
                record = {**record, "native_layout": layout}
                constrained = getattr(args, "mc_choice_score", False)
                output = net.generate(steps, record, max_new_tokens=args.max_new_tokens,
                                      return_choice=constrained)
                ids, choice = output if constrained else (output, None)
                text = bb.tokenizer.decode(ids, skip_special_tokens=True)
                prediction = {"run": identity["id"], "id": row["id"], "video": row["video"],
                              "answer": text, "token_ids": ids, "observation": layout.audit,
                              "choice": choice,
                              "visual_tokens": net.prefix(steps, record)[4]["visual_tokens_total"],
                              **score_answer(text, record)}
                atomic_text(question_path(row), json.dumps(prediction, indent=2) + "\n")
                print(json.dumps({"shard": args.shard, "new_completed": index + 1,
                                  "new_total": len(pending), "id": row["id"]}), flush=True)
        atomic_text(out / f"shard-{args.shard}.json", json.dumps({"run": identity["id"],
                    "completed": True, "ids": [r["id"] for r in selected]}) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--mode", choices=("full", "grid"), default="grid")
    p.add_argument("--pool-grid", nargs=2, type=int, default=[4, 5])
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--pixels", type=int, default=256 * 256)
    p.add_argument("--timestamp-policy", choices=("strict", "native-index-fps"), default="strict")
    p.add_argument("--split", choices=("val", "dev"), default="val")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--mc-choice-score", action="store_true", help="also score first-token option logits; no gold input")
    run(p.parse_args())


if __name__ == "__main__":
    main()
