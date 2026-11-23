"""Evaluate two spatial sampling rates in completed historical video blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trace_vl.core_vlm.data import decode_frame_indices, sha256
from trace_vl.core_vlm.experiment import paired
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VideoQALoss, grid_pool
from trace_vl.streaming_vision.data import read_rows, write_rows
from trace_vl.streaming_vision.media import VideoProbeCache
from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from .experiment import encode_steps, load, option_heads, query_hidden
from .paired_probe import pooled_history, preview

ARMS = ("uniform", "fixed", "adaptive")
HISTORY_TOKENS = 128
SCOPE = "Previously used development pairs, given annotated evidence, one frozen Qwen checkpoint."


def replay(cache, pair, receipt, pixels):
    audits = receipt["sampling"]
    indices = sorted(i for clip in audits for i in clip["indices"])
    if len(indices) != len(set(indices)):
        raise ValueError("reference windows have overlapping frame indices")
    video, metadata, checksum = cache.source(pair["video"], stage=True)
    if any(clip["video"] != checksum for clip in audits):
        raise ValueError("source video differs from reference")
    stream = metadata["streams"][0]
    height, width = frame_size(stream["height"], stream["width"], pixels, 32)
    frames = decode_frame_indices(video, indices, height, width, cache.retry_timeout, stop_after_selection=True)
    offsets = {index: i for i, index in enumerate(indices)}
    clips = [frames[[offsets[i] for i in clip["indices"]]].reshape(-1, 2, height, width, 3) for clip in audits]
    if any(hashlib.sha256(clip.tobytes()).hexdigest() != audit["rgb"]
           for clip, audit in zip(clips, audits, strict=True)):
        raise ValueError(f"frames differ from the verified reference: {pair['id']}")
    return clips


def two_rates(steps, anchors):
    """Save eight tokens at two steps and 24 at each of the two anchors."""
    pooled = []
    for i, step in enumerate(steps):
        height, width = step.grid
        grid = (4, 6) if i in anchors else (2, 4)
        if height > width:
            grid = grid[::-1]
        tokens = grid_pool(step.tokens, step.grid, grid)[0]
        pooled.append(replace(step, tokens=tokens, grid=grid))
    return pooled


def changing_steps(steps):
    """Rank visual changes measured at each step against the preceding step."""
    scores = [float("-inf")]
    for previous, current in zip(steps, steps[1:]):
        change = 1 - F.cosine_similarity(previous.tokens.float(), current.tokens.float(), dim=-1)
        scores.append(change.topk(max(1, change.numel() // 4)).values.mean().item())
    return sorted(sorted(range(len(steps)), key=scores.__getitem__, reverse=True)[:2])


@torch.no_grad()
def evaluate(args, bb, qa, pair, frames, receipt):
    total_steps = sum(len(clip) for clip in frames)
    times = np.arange(2 * total_steps, dtype=float).reshape(-1, 2) / 2
    encoded = list(encode_steps(args, bb, np.concatenate(frames), times))
    clips = np.split(np.arange(len(encoded)), np.cumsum([len(clip) for clip in frames])[:-1])
    if len(clips) != 4 or any(len(clip) != 4 for clip in clips[:3]):
        raise ValueError("128-token protocol expects three four-step historical blocks")
    options = pair["labels"][:2][::-1] if pair["options_reversed"] else pair["labels"][:2]
    record = {"question": "Which of these two events happened first in the video?",
              "options": options, "query_time": float(times[-1, -1])}
    original = pair["order"]
    reverse = [1 - i if i < 2 else i for i in original]
    predictions = []
    for order, ref in zip((original, reverse), receipt["predictions"], strict=True):
        history = {arm: [] for arm in ARMS}
        cursor = 0
        adaptive_anchors = []
        for clip_index in (*order, 3):
            steps = []
            for index in clips[clip_index]:
                ts = tuple(float(t) for t in times[cursor])
                steps.append(replace(encoded[index], end=ts[-1], frame_times=ts))
                cursor += 1
            if clip_index == 3:
                recent = steps
            if clip_index >= 2:
                continue
            anchors = changing_steps(steps)
            adaptive_anchors.append(anchors)
            history["uniform"].extend(pooled_history(steps, 4))
            history["fixed"].extend(two_rates(steps, (1, 3)))
            history["adaptive"].extend(two_rates(steps, anchors))
        answer = next(i for i in order if i < 2) ^ int(pair["options_reversed"])
        record["answer"] = answer
        if answer != ref["answer"]:
            raise ValueError("answer drift from the reference population")
        heads = option_heads(bb, record)
        choices = {}
        for arm, steps in history.items():
            if sum(len(step.tokens) for step in steps) != HISTORY_TOKENS:
                raise ValueError(f"history budget mismatch in {arm}")
            choices[arm] = int((heads @ query_hidden(bb, qa, record, steps + recent)).argmax())
        if choices["uniform"] != ref["choice"]["oracle_grid4"]:
            raise ValueError(f"uniform 128-token baseline failed parity for {pair['id']}")
        predictions.append({"answer": answer, "choice": choices, "adaptive_anchors": adaptive_anchors})
    return predictions


def run_shard(args, plan, receipts, out):
    rows = plan["rows"][args.shard::args.shards]
    pending = [r for r in rows if not (out / "rows" / f"{r['id']}.json").is_file()]
    if not pending:
        return
    backbone = Backbone(BackboneConfig(name=args.model, dtype=args.dtype, local_files_only=True),
                        StreamConfig(pixels_per_frame=args.pixels), device=torch.device(args.device))
    qa = VideoQALoss(backbone, mode="full").eval()
    cache = VideoProbeCache(args.scratch, stage_media=True)
    (out / "rows").mkdir(exist_ok=True)
    for i, row in enumerate(pending, 1):
        receipt = receipts[row["id"]]
        frames = replay(cache, row, receipt, args.pixels)
        predictions = evaluate(args, backbone, qa, row, frames, receipt)
        preview(out / "previews" / f"{row['id']}.jpg", row, frames)
        write_json(out / "rows" / f"{row['id']}.json",
                   {"id": row["id"], "source_group": row["source_group"], "plan": plan["id"],
                    "predictions": predictions})
        if i == 1 or i % 5 == 0 or i == len(pending):
            print(f"DUAL RATE shard={args.shard} {i}/{len(pending)}", flush=True)


def report(plan, out):
    rows = [json.loads((out / "rows" / f"{row['id']}.json").read_text()) for row in plan["rows"]]
    refs, results, correctness = [], {}, {arm: {} for arm in ARMS}
    for source, row in zip(plan["rows"], rows, strict=True):
        if row["plan"] != plan["id"] or row["id"] != source["id"]:
            raise ValueError("row does not match active plan")
        refs.append({"id": row["id"], "video": source["source_group"], "answer": 0,
                     "options": ["both correct", "at most one correct"]})
        for arm in ARMS:
            correct = all(p["choice"][arm] == p["answer"] for p in row["predictions"])
            correctness[arm][row["id"]] = {"answer": "A" if correct else "B"}
    for arm in ARMS:
        count = sum(p["answer"] == "A" for p in correctness[arm].values())
        results[arm] = {"both_correct": count, "pair_accuracy": count / len(rows)}
    comparisons = {(a, b): paired(refs, correctness[a], correctness[b])
                   for a, b in (("fixed", "uniform"), ("adaptive", "uniform"), ("adaptive", "fixed"))}
    meets_threshold = {}
    for arm in ("fixed", "adaptive"):
        value = comparisons[arm, "uniform"]
        meets_threshold[arm] = bool(value["delta_pp"] >= 2 and value["ci95_pp"]
                                    and value["ci95_pp"][0] > 0)
    result = {"plan": plan["id"], "scope": SCOPE, "n": len(rows), "history_tokens": HISTORY_TOKENS,
              "verified_reference_baseline": True, "arms": results,
              "paired": {f"{a}_minus_{b}": value for (a, b), value in comparisons.items()},
              "independent_validation_warranted": meets_threshold}
    write_json(out / "report.json", result)
    write_rows(out / "predictions.jsonl", rows)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print(f"VERIFIED DUAL RATE: {out / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "eval", "report"))
    for name in ("base", "reference", "out", "scratch"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = load(args.base, args)
    ref_plan_path = args.reference / "plan.json"
    ref_prediction_path = args.reference / "predictions.jsonl"
    reference = json.loads(ref_plan_path.read_text())
    receipts = {row["id"]: row for row in read_rows(ref_prediction_path)}
    if reference["source_run"] != run or "history_budget_protocol" not in reference:
        raise ValueError("expected a fixed-budget report from the same source run")
    if not reference["rows"] or set(receipts) != {r["id"] for r in reference["rows"]}:
        raise ValueError("reference coverage is incomplete")
    plan = {"source_run": run, "reference_plan_sha256": sha256(ref_plan_path),
            "reference_predictions_sha256": sha256(ref_prediction_path), "rows": reference["rows"],
            "protocol": "two-rates-128-v1", "revision": os.environ.get("TRACE_ROUTE_REV", "local")}
    if any("sampling" not in r for r in receipts.values()):
        raise ValueError("reference contains excluded media")
    plan["id"] = digest(plan)
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_run_manifest(args.out / "plan.json", plan)
    if args.stage == "plan":
        print(f"PLAN {len(plan['rows'])} verified pairs, {HISTORY_TOKENS} history tokens", flush=True)
    elif args.stage == "eval":
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard")
        run_shard(args, plan, receipts, args.out)
    else:
        report(plan, args.out)


if __name__ == "__main__":
    main()
