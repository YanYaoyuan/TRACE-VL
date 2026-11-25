"""Readout and routing controls on paired history orders with identical recent frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.experiment import paired
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VideoQALoss, grid_pool
from trace_vl.streaming_vision.data import read_rows, write_rows
from trace_vl.streaming_vision.media import VideoProbeCache
from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from .data import source_id, tasks
from .experiment import encode_steps, load, option_heads, query_hidden
from .model import IntervalReader, VisualHistory

ARMS = ("recent", "dense_oracle", "oracle", "position", "cosine", "learned")
EXTRA_ARMS = ("oracle_grid4", "oracle_grid6", "oracle_motion2")


def center_window(event, seconds):
    start, end, _ = event
    if end - start < seconds:
        return None
    center = (start + end) / 2
    return [center - seconds / 2, center + seconds / 2]


def make_plan(args, base, out, run):
    inputs = json.loads((base / "inputs.json").read_text())
    if sha256(args.annotation) != inputs["activitynet_sha256"]:
        raise ValueError("ActivityNet annotations differ from the frozen inputs")
    with zipfile.ZipFile(args.annotation) as z:
        captions = {source_id(key): value for key, value in json.loads(z.read("train.json")).items()}
    candidates, rejected = [], Counter()
    for row in read_rows(base / "dev.jsonl"):
        source = row["source_group"]
        matches = tasks(captions[source], source, "dev", args.seed, args.recent, args.lookback,
                        args.block, include_events=True)
        left, anchor, right = next(x["events"] for x in matches if x["id"] == row["id"])
        events = [center_window(event, args.block) for event in (left, right, anchor)]
        if any(span is None for span in events):
            rejected["short_annotated_event_or_distractor"] += 1
            continue
        recent = [right[1], right[1] + args.recent]
        if recent[1] > row["media_duration"]:
            rejected["short_common_tail"] += 1
            continue
        rng = np.random.default_rng(int(digest([args.seed, row["id"], "paired-history-v1"])[:12], 16))
        candidates.append({"id": row["id"], "source_group": source, "video": row["video"],
                           "labels": [left[2], right[2], anchor[2]], "intervals": events + [recent],
                           "order": rng.permutation(3).tolist(), "options_reversed": bool(rng.integers(2))})
    if not candidates:
        raise ValueError("no paired histories fit the annotated event and recent windows")
    plan = {"source_run": run, "dev_sha256": sha256(base / "dev.jsonl"), "rows": candidates,
            "rejected": dict(rejected), "reader_sha256": sha256(base / "learned.pt"),
            "revision": os.environ.get("TRACE_ROUTE_REV", "local"),
            "scope": "Caption-derived counterfactual development pairs; sampled label visibility requires review."}
    if args.budget_sweep:
        plan["history_budget_protocol"] = "oracle-grid-and-motion-v1"
    plan["id"] = digest(plan)
    ensure_run_manifest(out / "plan.json", plan)
    print(f"PLAN {len(candidates)} pairs; exclusions {dict(rejected)}", flush=True)


def sample(args, cache, pair):
    images, audits = [], []
    for i, (start, end) in enumerate(pair["intervals"]):
        end = math.nextafter(end, start)
        row = {"video": pair["video"], "query_time": end, "observation": {"start": start, "end": end}}
        frames, times, _, audit = cache.decode(row, steps=args.block if i < 3 else args.recent,
                                              pixels=args.pixels, drop_bad_pts=True)
        images.append(frames)
        audits.append({"indices": audit["frame_indices"], "times": times.tolist(),
                       "rgb": audit["decoded_rgb_sha256"], "video": audit["video_sha256"]})
    if len({a["video"] for a in audits}) != 1:
        raise ValueError("source changed during pair construction")
    if len({a["rgb"] for a in audits[:3]}) != 3:
        raise ValueError("duplicate historical clips")
    all_indices = [index for audit in audits for index in audit["indices"]]
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("shared source frames across clip intervals")
    return images, audits


def preview(path, pair, frames):
    canvas = Image.new("RGB", (8 * 144, 6 * 110), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f"{pair['id']} | A, B, distractor, identical recent window", fill="black")
    for i, array in enumerate(frames):
        row = (i + 1) * 105
        label = pair["labels"][i] if i < 3 else "Identical recent window"
        draw.text((5, row), label[:180].encode("ascii", "replace").decode(), fill="black")
        flat = array.reshape(-1, *array.shape[2:])
        for j, index in enumerate(np.linspace(0, len(flat) - 1, 8).round().astype(int)):
            thumb = Image.fromarray(flat[index])
            thumb.thumbnail((140, 84))
            canvas.paste(thumb, (j * 144, row + 16))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=82)


def pooled_history(steps, side):
    for step in steps:
        height, width = step.grid
        grid = min(side, height), min(side, width)
        tokens = grid_pool(step.tokens, step.grid, grid)[0]
        yield replace(step, tokens=tokens, grid=grid)


def motion_history(steps):
    """Favor changing patches inside each 2x2 region and keep its position."""
    previous = None
    for step in steps:
        height, width = step.grid
        features = step.tokens.float().reshape(height, width, -1)
        scores = (features.new_zeros((height, width)) if previous is None
                  else 1 - F.cosine_similarity(features, previous, dim=-1))
        rows, cols = min(2, height), min(2, width)
        parts = []
        for row in range(rows):
            for col in range(cols):
                y = slice(row * height // rows, (row + 1) * height // rows)
                x = slice(col * width // cols, (col + 1) * width // cols)
                cells = features[y, x].reshape(-1, features.shape[-1])
                weights = scores[y, x].flatten().mul(10).softmax(0)
                parts.append(weights @ cells)
        pooled = torch.stack(parts).reshape(rows, cols, -1).expand(2, 2, -1)
        yield replace(step, tokens=pooled.reshape(4, -1).to(step.tokens.dtype), grid=(2, 2))
        previous = features


@torch.no_grad()
def evaluate(args, bb, qa, reader, pair, frames):
    n_steps = [len(clip) for clip in frames]
    times = np.arange(2 * sum(n_steps), dtype=float).reshape(-1, 2) / 2
    encoded = list(encode_steps(args, bb, np.concatenate(frames), times))
    clips = np.split(np.arange(len(encoded)), np.cumsum(n_steps)[:-1])
    order = pair["order"]
    reversed_order = [1 - x if x < 2 else x for x in order]
    options = pair["labels"][:2][::-1] if pair["options_reversed"] else pair["labels"][:2]
    row = {"question": "Which of these two events happened first in the video?",
           "options": options, "query_time": float(times[-1, -1])}
    predictions = []
    for permutation in (order, reversed_order):
        bank = VisualHistory(args.recent, args.block, args.capacity)
        dense = []
        cursor = 0
        for clip in (*permutation, 3):
            steps = []
            for j in clips[clip]:
                ts = tuple(float(t) for t in times[cursor])
                step = replace(encoded[j], end=ts[-1], frame_times=ts)
                steps.append(step)
                bank.append(step)
                cursor += 1
            dense.append(steps)
        recent = list(bank.recent)
        if len(bank.slots) != 3 or len(recent) != args.recent:
            raise ValueError("pair must contain three complete history blocks and the fixed recent window")
        target = [i for i, event in enumerate(permutation) if event < 2]
        first_event = permutation[target[0]]
        row["answer"] = int(first_event) ^ int(pair["options_reversed"])
        heads = option_heads(bb, row)
        query = query_hidden(bb, qa, row, recent)
        slots, spans = bank.tensors()
        slots = slots.float()
        bounds = ((spans - row["query_time"]) / args.lookback).to(bb.device)
        cosine = F.normalize(slots.mean(1), dim=-1) @ F.normalize(query, dim=-1)
        _, learned, _ = reader(query, slots, bounds, heads=heads)
        routes = {"oracle": target, "position": [0, 1],
                  "cosine": sorted(cosine.topk(2).indices.tolist()),
                  "learned": sorted(learned.topk(2).indices.tolist())}
        choices = {"recent": int((heads @ query).argmax())}
        dense_history = [step for i in target for step in dense[i]]
        choices["dense_oracle"] = int((heads @ query_hidden(bb, qa, row, dense_history + recent)).argmax())
        count = {"recent": 0, "dense_oracle": sum(len(s.tokens) for s in dense_history)}
        if args.budget_sweep:
            alternatives = {
                "oracle_grid4": list(pooled_history(dense_history, 4)),
                "oracle_grid6": list(pooled_history(dense_history, 6)),
                "oracle_motion2": [s for i in target for s in motion_history(dense[i])],
            }
            for arm, history in alternatives.items():
                count[arm] = sum(len(s.tokens) for s in history)
                choices[arm] = int((heads @ query_hidden(bb, qa, row, history + recent)).argmax())
            if count["oracle_motion2"] != 2 * args.block * 4:
                raise ValueError("motion pooling changed the 2x2 history budget")
        cached = {}
        for arm, indices in routes.items():
            key = tuple(indices)
            if key not in cached:
                selected = [step for i in indices for step in bank.slots[i]]
                if sum(len(step.tokens) for step in selected) != 2 * args.block * 4:
                    raise ValueError("history token budget differs across paired routes")
                cached[key] = int((heads @ query_hidden(bb, qa, row, selected + recent)).argmax())
            choices[arm] = cached[key]
            count[arm] = 2 * args.block * 4
        prediction = {"answer": row["answer"], "choice": choices, "route": routes}
        if args.budget_sweep:
            prediction["history_tokens"] = count
        predictions.append(prediction)
    if predictions[0]["answer"] == predictions[1]["answer"]:
        raise ValueError("paired answers did not flip")
    if predictions[0]["choice"]["recent"] != predictions[1]["choice"]["recent"]:
        raise ValueError("shared recent baseline changed across paired inputs")
    return predictions


def run_shard(args, base, out, plan):
    selected = plan["rows"][args.shard::args.shards]
    selected = [row for row in selected if not (out / "rows" / f"{row['id']}.json").exists()]
    if not selected:
        return
    device = torch.device(args.device)
    bb = Backbone(BackboneConfig(name=args.model, dtype=args.dtype, local_files_only=True),
                  StreamConfig(pixels_per_frame=args.pixels), device=device)
    qa = VideoQALoss(bb, mode="full").eval()
    weights = torch.load(base / "learned.pt", map_location="cpu", weights_only=True)
    if weights["run"] != plan["source_run"] or weights["mode"] != "learned":
        raise ValueError("reader checkpoint belongs to a different experiment")
    reader = IntervalReader(weights["hidden"], args.width, "learned").to(device).eval()
    reader.load_state_dict(weights["state"])
    cache = VideoProbeCache(args.scratch, stage_media=True)
    (out / "rows").mkdir(exist_ok=True)
    for i, pair in enumerate(selected, 1):
        path = out / "rows" / f"{pair['id']}.json"
        result = {"id": pair["id"], "source_group": pair["source_group"], "plan": plan["id"]}
        try:
            frames, audits = sample(args, cache, pair)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            result["excluded"] = str(error)
        else:
            result["predictions"] = evaluate(args, bb, qa, reader, pair, frames)
            result["sampling"] = audits
            result["recent_rgb_sha256"] = hashlib.sha256(frames[-1].tobytes()).hexdigest()
            preview(out / "previews" / f"{pair['id']}.jpg", pair, frames)
        write_json(path, result)
        if i == 1 or i % 5 == 0 or i == len(selected):
            print(f"PAIRS shard={args.shard} {i}/{len(selected)}", flush=True)


def report(args, out, plan):
    all_rows = [json.loads((out / "rows" / f"{r['id']}.json").read_text()) for r in plan["rows"]]
    if any((r["id"], r["plan"]) != (s["id"], plan["id"])
           for r, s in zip(all_rows, plan["rows"], strict=True)):
        raise ValueError("saved pairs have a different identity")
    rows = [r for r in all_rows if "predictions" in r]
    results, maps = {}, {}
    for arm in ARMS + (EXTRA_ARMS if args.budget_sweep else ()):
        outcomes = [[p["choice"][arm] == p["answer"] for p in row["predictions"]] for row in rows]
        count = sum(all(pair) for pair in outcomes)
        results[arm] = {"both_correct": count, "pair_accuracy": count / len(rows) if rows else None,
                        "correct_questions": sum(sum(pair) for pair in outcomes)}
        if arm in ("position", "cosine", "learned"):
            results[arm]["both_blocks_found"] = sum(
                p["route"][arm] == p["route"]["oracle"] for row in rows for p in row["predictions"]
            )
        if args.budget_sweep and rows:
            counts = [p["history_tokens"][arm] for r in rows for p in r["predictions"]]
            results[arm]["history_tokens"] = {"mean": sum(counts) / len(counts),
                                               "min": min(counts), "max": max(counts)}
        maps[arm] = {r["id"]: {"answer": "A" if all(pair) else "B"}
                     for r, pair in zip(rows, outcomes, strict=True)}
    refs = [{"id": r["id"], "video": r["source_group"], "source_group": r["source_group"],
             "options": ["both correct", "at most one correct"], "answer": 0} for r in rows]
    compare = (("dense_oracle", "recent"), ("oracle", "dense_oracle"),
               ("learned", "position"), ("learned", "cosine"), ("learned", "oracle"))
    if args.budget_sweep:
        compare += (("oracle_grid4", "oracle"), ("oracle_grid6", "oracle_grid4"),
                    ("oracle_grid6", "dense_oracle"), ("oracle_motion2", "oracle"))
    budget_key = "matched_history_tokens" if args.budget_sweep else "history_tokens_per_budgeted_arm"
    result = {"plan": plan["id"], "scope": plan["scope"], "eligible_pairs": len(rows),
              "excluded": {r["id"]: r["excluded"] for r in all_rows if "excluded" in r},
              budget_key: 2 * args.block * 4, "arms": results,
              "paired": {f"{a}_minus_{b}": paired(refs, maps[a], maps[b]) for a, b in compare}}
    write_json(out / "report.json", result)
    write_rows(out / "predictions.jsonl", all_rows)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not rows:
        raise ValueError("all pairs were excluded during sampling")
    print(f"VERIFIED PAIRED HISTORY: {out / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "eval", "report"))
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-sweep", action="store_true")
    args = parser.parse_args()
    base, out = Path(args.base), Path(args.out)
    run = load(base, args)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        make_plan(args, base, out, run)
        return
    plan = json.loads((out / "plan.json").read_text())
    if plan["source_run"] != run:
        raise ValueError("pair plan belongs to a different source run")
    if ("history_budget_protocol" in plan) != args.budget_sweep:
        raise ValueError("pair plan belongs to a different budget protocol")
    if args.stage == "eval":
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard")
        run_shard(args, base, out, plan)
    else:
        report(args, out, plan)


if __name__ == "__main__":
    main()
