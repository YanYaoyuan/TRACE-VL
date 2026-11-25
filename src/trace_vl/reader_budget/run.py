"""Evaluate nested reads of a frozen retained bank; never train or decode video."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.experiment import run_jobs
from trace_vl.core_vlm.native import NativeVideoQALoss
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.train import checked_checkpoint
from trace_vl.longstream_training.prepare import rows
from trace_vl.longstream_training.train import sample
from trace_vl.reader_controls import run as controls
from trace_vl.v3.run_io import ensure_run_manifest

from . import analysis, policies

REFERENCE = Path("/data/yyy/tracevl-runs/reader-controls-76b1162-001")
REFERENCE_REPORT_SHA = "3cd3e190fb702de29b449f9e3975ac5adf0be7fca10e28a62112d7d7911e0ecb"
REFERENCE_INPUT = "e9051aa794358fb06cd70cc5937a36e3d12385f91db8e445dfb37b0d42ac48aa"
REPRODUCTION_TOLERANCE = .25


def code_identity():
    root = Path(__file__).parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def versions():
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__}


def load_inputs(args):
    value = controls.load_json(args.out / "inputs.json")
    if value["id"] != digest({k: v for k, v in value.items() if k != "id"}):
        raise ValueError("budget input identity corrupted")
    if value["code"] != code_identity() or value["versions"] != versions():
        raise ValueError("code or versions changed")
    return value


def reference(args):
    if sha256(args.reference / "report.json") != REFERENCE_REPORT_SHA:
        raise ValueError("complete control report changed")
    report = controls.load_json(args.reference / "report.json")
    ref = controls.load_json(args.reference / "inputs.json")
    if not report["completed"] or report["inputs"] != ref["id"] or ref["id"] != REFERENCE_INPUT:
        raise ValueError("wrong completed controls")
    if digest({k: v for k, v in ref.items() if k != "id"}) != ref["id"]:
        raise ValueError("reference manifest corrupted")
    if checkpoint_identity(args.model) != ref["model"]:
        raise ValueError("base model changed")
    if sha256(args.adapter / "adapter_model.safetensors") != ref["adapter_sha256"]:
        raise ValueError("starting adapter changed")
    blank = controls.load_json(args.reference / "blank.json")
    if blank["inputs"] != ref["id"] or sha256(args.reference / "blank.pt") != blank["sha256"]:
        raise ValueError("reference blank changed")
    if any(v != 0 for v in blank["max_difference_between_temporal_steps"].values()):
        raise ValueError("reusing blank step zero requires exact temporal equality")
    return ref, blank


def freeze(args):
    ref, blank = reference(args)
    if sha256(args.data / "dev.jsonl") != ref["dev_sha256"]:
        raise ValueError("development manifest changed")
    dev = rows(args.data / "dev.jsonl")
    if len(dev) != 115 or len({r["source_group"] for r in dev}) != 115:
        raise ValueError("all 115 sources required")
    questions = []
    for row in dev:
        entry, feature = controls.load_entry(args.pilot / "features", row, ref["cache"])
        if feature != ref["feature_sha256"][row["id"]]:
            raise ValueError("reference cached features changed")
        reads = policies.build(entry["times"], row["query_time"], row["source_group"])
        path = args.reference / "scores" / "start" / f"{digest(row['id'])}.json"
        saved = controls.load_json(path)
        if (saved["inputs"] != ref["id"] or saved["id"] != row["id"]
                or saved["source_group"] != row["source_group"] or saved["gold"] != row["answer"]):
            raise ValueError("reference score identity changed")
        if row["task"] not in ("before", "after") or len(row["options"]) != 2:
            raise ValueError("unexpected task")
        questions.append({"id": row["id"], "source_group": row["source_group"], "gold": row["answer"],
                          "task": row["task"], "query_time": row["query_time"],
                          "bank_times": [[float(t) for t in pair] for pair in entry["times"]],
                          "reads": reads, "reference_score_sha256": sha256(path)})
    value = {"protocol": "nested-reader-budget-v1", "reference_input": ref["id"],
             "reference_report_sha256": REFERENCE_REPORT_SHA, "dev_sha256": ref["dev_sha256"],
             "model": ref["model"], "adapter_sha256": ref["adapter_sha256"], "cache": ref["cache"],
             "feature_sha256": ref["feature_sha256"], "blank_sha256": blank["sha256"],
             "budgets": list(policies.BUDGETS), "recent": 4, "history": 32,
             "read_seeds": list(policies.SEEDS), "primary": list(analysis.PRIMARY),
             "contrasts": analysis.CONTRASTS, "bootstrap": {"draws": 10000, "seed": 7},
             "reproduction_tolerance": REPRODUCTION_TOLERANCE, "code": code_identity(), "versions": versions(),
             "protocol_sha256": sha256(Path(__file__).parents[3] / "docs/reader_budget/protocol.md"),
             "questions": questions, "training": False}
    value["id"] = digest(value)
    ensure_run_manifest(args.out / "inputs.json", value)


def synchronize(bb):
    if torch.device(bb.device).type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def score_question(bb, net, entry, row, reads, blank):
    synchronize(bb)
    begin = time.perf_counter()
    logits = {"text_only": controls.text_only(bb, row)}
    synchronize(bb)
    text_seconds = time.perf_counter() - begin
    tokens, timings = {"text_only": 0}, {}
    layouts = OrderedDict()
    for name, indices in reads.items():
        positions = None
        for prefix, values in (("", None), ("blank:", blank)):
            synchronize(bb)
            begin = time.perf_counter()
            steps, record, chosen = sample(bb, controls.fixed(indices, values), entry, row, layouts)
            embeds, pos, _, _, info = net.prefix(steps, record)
            synchronize(bb)
            packing = time.perf_counter() - begin
            if chosen != indices or info["visual_tokens_total"] != 16 * len(indices):
                raise ValueError("packing changed frozen indices or budget")
            if positions is not None and not torch.equal(pos, positions):
                raise ValueError("blank changed positions")
            positions = pos
            begin = time.perf_counter()
            hidden = bb.lm(embeds, pos, cache=None, use_cache=False)
            logits[prefix + name] = controls.option_logits(bb, hidden, row)
            synchronize(bb)
            timings[prefix + name] = {"packing_seconds": packing, "lm_seconds": time.perf_counter() - begin,
                                      "input_tokens": int(embeds.shape[1])}
            tokens[prefix + name] = info["visual_tokens_total"]
    return logits, tokens, timings, text_seconds


def validate_record(record, question, inputs):
    names = {"text_only", *question["reads"], *("blank:" + n for n in question["reads"])}
    if (not record["completed"] or record["inputs"] != inputs["id"] or record["id"] != question["id"]
            or record["source_group"] != question["source_group"] or record["gold"] != question["gold"]
            or record["task"] != question["task"] or record["weights"] != "start"
            or set(record["logits"]) != names or set(record["visual_tokens"]) != names):
        raise ValueError("score identity or policy coverage changed")
    if any(record["visual_tokens"][n] != policies.expected_tokens(n) for n in names):
        raise ValueError("stored score budget changed")
    if any(len(logits) != 2 or not all(math.isfinite(v) for v in logits)
           for logits in record["logits"].values()):
        raise ValueError("invalid option logits")
    if not math.isfinite(record["anchor_max_logit_gap"]) or record["anchor_max_logit_gap"] > REPRODUCTION_TOLERANCE:
        raise ValueError("reference anchor mismatch")


def worker(args):
    inputs = load_inputs(args)
    ref, _ = reference(args)
    if sha256(args.data / "dev.jsonl") != inputs["dev_sha256"]:
        raise ValueError("development manifest changed")
    previous = controls.load_json(args.adapter.parent.parent / "run.json")
    checked_checkpoint(args.adapter.parent, previous["id"])
    bb = controls.backbone(args)
    bb.add_lora(8, 16, adapter=str(args.adapter))
    bb.model.eval().requires_grad_(False)
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4), answer_end_token_id=bb.tokenizer.eos_token_id)
    net.eval()
    blanks = torch.load(args.reference / "blank.pt", map_location="cpu", weights_only=True)
    by_id = {q["id"]: q for q in inputs["questions"]}
    queue = rows(args.data / "dev.jsonl")[args.shard::2]
    if args.smoke:
        queue = queue[:1]
    for row in queue:
        q = by_id[row["id"]]
        path = args.out / "scores" / f"{digest(row['id'])}.json"
        if path.exists():
            validate_record(controls.load_json(path), q, inputs)
            continue
        entry, feature = controls.load_entry(args.pilot / "features", row, inputs["cache"])
        if feature != inputs["feature_sha256"][row["id"]]:
            raise ValueError("cached feature changed")
        source = args.reference / "scores" / "start" / path.name
        if sha256(source) != q["reference_score_sha256"]:
            raise ValueError("reference logits changed")
        saved = controls.load_json(source)
        h, w = entry["hw"]
        torch.cuda.reset_peak_memory_stats()
        logits, tokens, timing, text_seconds = score_question(
            bb, net, entry, row, q["reads"], blanks[f"{h}x{w}"]
        )
        pairs = (("all36", "all_bank"), ("blank:all36", "blank:all_bank"), ("text_only", "text_only"))
        gaps = {a: max(abs(x - y) for x, y in zip(logits[a], saved["logits"][b], strict=True))
                for a, b in pairs}
        if max(gaps.values()) > REPRODUCTION_TOLERANCE:
            raise ValueError(f"reference logits not reproduced: {gaps}")
        value = {"completed": True, "inputs": inputs["id"], "reference_input": ref["id"], "weights": "start",
                 "id": row["id"], "source_group": row["source_group"], "gold": row["answer"], "task": row["task"],
                 "logits": logits, "visual_tokens": tokens, "timings": timing,
                 "text_only_total_seconds": text_seconds, "anchor_max_logit_gap": max(gaps.values()),
                 "anchor_gaps": gaps, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        validate_record(value, q, inputs)
        write_json(path, value)
        print(json.dumps({"scored": row["id"], "anchor_max_gap": max(gaps.values())}), flush=True)


def records(args, inputs):
    found = []
    for q in inputs["questions"]:
        path = args.out / "scores" / f"{digest(q['id'])}.json"
        if path.exists():
            value = controls.load_json(path)
            validate_record(value, q, inputs)
            found.append(value)
    return found


def report(args):
    inputs = load_inputs(args)
    scored = records(args, inputs)
    result = analysis.analyze(scored, inputs["questions"])
    timing = {}
    for name in scored[0]["timings"]:
        values = [r["timings"][name] for r in scored]
        timing[name] = {k: sum(v[k] for v in values) / len(values)
                        for k in ("packing_seconds", "lm_seconds", "input_tokens")}
    write_json(args.out / "report.json", {"completed": True, "inputs": inputs["id"], "analysis": result,
               "max_anchor_gap": max(r["anchor_max_logit_gap"] for r in scored),
               "peak_allocated_gib": max(r["peak_allocated_gib"] for r in scored), "mean_timings": timing,
               "timing_scope": "synchronized per-question preparation and LM time; descriptive, no deployment claim",
               "automatic_training": False})


def run(args):
    gpus = [int(g) for g in args.gpus.split(",")]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("exactly two distinct GPUs required")
    args.out.mkdir(parents=True, exist_ok=True)
    freeze(args)
    common = []
    for name in ("out", "reference", "pilot", "data", "model", "adapter"):
        common += [f"--{name}", str(getattr(args, name))]
    base = [sys.executable, "-m", "trace_vl.reader_budget.run", "worker", *common]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "TRACE_WANDB": "0", "TRACE_SWANLAB": "0"}
    def jobs(stage, extra):
        return [(f"{stage}-{i}", [*base, "--shard", str(i), *extra], [g]) for i, g in enumerate(gpus)]
    run_jobs(jobs("smoke", ["--smoke"]), env, args.out / "logs")
    inputs = load_inputs(args)
    smoke = records(args, inputs)
    smoke_ids = {q["id"] for q in inputs["questions"][:2]}
    smoke = [r for r in smoke if r["id"] in smoke_ids]
    if len(smoke) != 2:
        raise ValueError("smoke incomplete")
    write_json(args.out / "smoke-ok.json", {"completed": True, "sources": 2,
               "inputs": inputs["id"], "max_anchor_gap": max(r["anchor_max_logit_gap"] for r in smoke),
               "peak_allocated_gib": max(r["peak_allocated_gib"] for r in smoke),
               "mean_lm_seconds_per_source": sum(sum(t["lm_seconds"] for t in r["timings"].values())
                                                for r in smoke) / 2,
               "outcome_metrics_exposed": False})
    if args.smoke_only:
        return
    run_jobs(jobs("full", []), env, args.out / "logs")
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "freeze", "worker", "report"))
    p.add_argument("--out", type=Path, required=True)
    for name, default in (("reference", REFERENCE), ("pilot", controls.PILOT), ("data", controls.DATA),
                          ("model", controls.MODEL), ("adapter", controls.ADAPTER)):
        p.add_argument(f"--{name}", type=Path, default=default)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--shard", type=int, choices=(0, 1), default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    {"run": run, "freeze": freeze, "worker": worker, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
