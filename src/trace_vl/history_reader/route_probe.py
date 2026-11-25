"""Compare the history routes under a shared frozen Qwen answer path."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.experiment import paired
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VideoQALoss
from trace_vl.streaming_vision.data import read_rows, write_rows
from trace_vl.streaming_vision.experiment import decoded
from trace_vl.streaming_vision.media import VideoProbeCache
from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from .experiment import encode_clip, load, option_heads, query_hidden
from .model import IntervalReader, evidence_indices

WINDOWS = ("original", "delayed")
ARMS = ("recent", "position", "cosine", "learned", "oracle", "distractor", "learned_residual", "oracle_residual")


def query_delay(row, args):
    """Choose a later query on real footage, keeping the annotated event visible in history."""
    initial = row["query_time"]
    first = math.ceil(initial + 16)
    last = math.floor(min(initial + 48, row["media_duration"] - 1,
                          row["evidence"][1] + args.lookback - args.block - 1))
    if first > last:
        return None
    return first + int(digest([args.seed, row["id"], "route-probe-v1"])[:12], 16) % (last - first + 1)


def setup(args, base, out, base_run):
    plan = []
    for row in read_rows(base / "dev.jsonl"):
        if (delay := query_delay(row, args)) is not None:
            plan.append({"id": row["id"], "original": row["query_time"], "delayed": delay,
                         "task": row["task"], "source_group": row["source_group"]})
    if not plan:
        raise ValueError("no sources can support the planned query offsets")
    identity = {"source_run": base_run, "plan": plan, "dev_manifest_sha256": sha256(base / "dev.jsonl"),
                "reader_sha256": sha256(base / "learned.pt"), "revision": os.environ.get("TRACE_ROUTE_REV", "local"),
                "protocol": "activitynet-paired-query-delay-v1",
                "notes": "Real footage, seeded delay 16–48 s when duration permits. Same video and questions."}
    identity["id"] = digest(identity)
    ensure_run_manifest(out / "plan.json", identity)
    print("PLAN", len(plan), {task: sum(x["task"] == task for x in plan) for task in ("before", "after")},
          flush=True)


@torch.no_grad()
def evaluate(args, bb, qa, reader, observation):
    images, times, row, audit = observation
    state = encode_clip(args, bb, images, times)
    recent = list(state.recent)
    targets = None
    if row["evidence"][1] < recent[0].frame_times[0]:
        targets = evidence_indices(state.slots, row["evidence"], row["id"])
    result = {"id": row["id"], "variant": row["route_variant"], "query_time": row["query_time"],
              "frame_times": times.tolist(), "frame_indices": audit["frame_indices"],
              "video_sha256": audit["video_sha256"]}
    if targets is None:
        return {**result, "eligible": False}
    gold, distractor = targets
    heads = option_heads(bb, row)
    query = query_hidden(bb, qa, row, recent)
    memory, spans = state.tensors()
    memory = memory.float()
    bounds = ((spans - row["query_time"]) / args.lookback).to(bb.device)
    cosine = (F.normalize(memory.mean(1), dim=-1) @ F.normalize(query, dim=-1)).argmax()
    residual, _, learned = reader(query, memory, bounds, heads=heads)
    forced, _, _ = reader(query, memory[gold:gold+1], bounds[gold:gold+1], heads=heads)
    choices = {"recent": int((heads @ query).argmax()), "learned_residual": int(residual.argmax()),
               "oracle_residual": int(forced.argmax())}
    routes = {"position": 0 if row["task"] == "before" else len(spans)-1,
              "cosine": int(cosine), "learned": int(learned), "oracle": gold, "distractor": distractor}
    by_block = {}
    for arm, index in routes.items():
        if index not in by_block:
            previous = list(state.slots[index])
            if sum(len(step.tokens) for step in previous) != args.block * 4:
                raise ValueError("history read budget differs across routes")
            by_block[index] = int((heads @ query_hidden(bb, qa, row, previous + recent)).argmax())
        choices[arm] = by_block[index]
    hits = [index for index, (start, end) in enumerate(spans.tolist())
            if min(end, row["evidence"][1]) > max(start, row["evidence"][0])]
    return {**result, "eligible": True, "choice": choices, "routes": routes,
            "overlap_slots": hits, "slots": len(spans), "history_tokens_read": args.block * 4}


def run_shard(args, base, out, plan):
    rows = {row["id"]: row for row in read_rows(base / "dev.jsonl")}
    shard = plan["plan"][args.shard::args.shards]
    pending = []
    for sample in shard:
        for variant in WINDOWS:
            query = sample[variant]
            row = {**rows[sample["id"]], "query_time": query, "route_variant": variant}
            row["observation"] = {**row["observation"], "end": query}
            path = out / "rows" / variant / f"{row['id']}.json"
            if not path.exists():
                pending.append((path, row))
    if not pending:
        return
    device = torch.device(args.device)
    bb = Backbone(BackboneConfig(name=args.model, dtype=args.dtype, local_files_only=True),
                  StreamConfig(pixels_per_frame=args.pixels), device=device)
    qa = VideoQALoss(bb, mode="full").eval()
    weights = torch.load(base / "learned.pt", map_location="cpu", weights_only=True)
    if weights["run"] != plan["source_run"] or weights["mode"] != "learned":
        raise ValueError("checkpoint does not belong to the frozen experiment")
    reader = IntervalReader(weights["hidden"], args.width, "learned").to(device).eval()
    reader.load_state_dict(weights["state"])
    cache = VideoProbeCache(args.scratch, stage_media=True)
    args.hz = 2.
    for i, ((path, _), observation) in enumerate(zip(
            pending, decoded(args, [row for _, row in pending], cache, drop_bad_pts=True), strict=True), 1):
        result = evaluate(args, bb, qa, reader, observation)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {**result, "plan": plan["id"]})
        if i == 1 or i % 5 == 0 or i == len(pending):
            print(f"ROUTES shard={args.shard} {i}/{len(pending)}", flush=True)


def report(base, out, plan):
    rows = {row["id"]: row for row in read_rows(base / "dev.jsonl")}
    populations = {variant: {} for variant in WINDOWS}
    all_answers, excluded = [], []
    for sample in plan["plan"]:
        row_id = sample["id"]
        variants = {}
        for variant in WINDOWS:
            path = out / "rows" / variant / f"{row_id}.json"
            item = json.loads(path.read_text())
            if (item["id"], item["variant"], item["plan"], item["query_time"]) != (
                    row_id, variant, plan["id"], sample[variant]):
                raise ValueError(f"response identity differs: {path}")
            if any(t > sample[variant] for step in item["frame_times"] for t in step):
                raise ValueError(f"future frame: {path}")
            variants[variant] = item
            all_answers.append({**item, "answer": rows[row_id]["answer"], "task": rows[row_id]["task"],
                                "source_group": sample["source_group"]})
        if not all(item["eligible"] for item in variants.values()):
            excluded.append(row_id)
            continue
        if variants["original"]["video_sha256"] != variants["delayed"]["video_sha256"]:
            raise ValueError(f"media changed between variants: {row_id}")
        for variant, item in variants.items():
            populations[variant][row_id] = item
    comparisons = (("oracle", "distractor"), ("oracle", "recent"), ("learned", "position"),
                   ("learned", "cosine"), ("learned", "recent"), ("learned", "learned_residual"),
                   ("oracle", "oracle_residual"))
    output = {"plan": plan["id"], "planned": len(plan["plan"]), "excluded": excluded, "windows": {}}
    for variant, predictions in populations.items():
        groups = {"all": list(predictions)}
        groups.update({task: [i for i in predictions if rows[i]["task"] == task] for task in ("before", "after")})
        entry = {}
        for name, ids in groups.items():
            answers = [rows[i] for i in ids]
            maps = {a: {i: {"answer": chr(65 + predictions[i]["choice"][a])} for i in ids} for a in ARMS}
            accuracy = {a: sum(predictions[i]["choice"][a] == rows[i]["answer"] for i in ids)
                        for a in ARMS}
            hits = {a: sum(predictions[i]["routes"][a] in predictions[i]["overlap_slots"] for i in ids)
                    for a in ("position", "cosine", "learned")}
            entry[name] = {"n": len(ids), "correct": accuracy, "overlap_hits": hits,
                           "paired": {f"{a}_minus_{b}": paired(answers, maps[a], maps[b])
                                      for a, b in comparisons}}
        output["windows"][variant] = entry
    write_rows(out / "predictions.jsonl", all_answers)
    write_json(out / "report.json", output)
    for variant, groups in output["windows"].items():
        main = groups["all"]
        print(variant, "n", main["n"], "correct", main["correct"], "hits", main["overlap_hits"], flush=True)
        for key, value in main["paired"].items():
            if value:
                print(key, f"{value['delta_pp']:+.2f} pp", value["ci95_pp"], flush=True)
    print(f"VERIFIED HISTORY ROUTES: {out / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "eval", "report"))
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    base, out = Path(args.base), Path(args.out)
    base_run = load(base, args)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        setup(args, base, out, base_run)
        return
    plan = json.loads((out / "plan.json").read_text())
    if plan["source_run"] != base_run:
        raise ValueError("route plan uses another source run")
    if args.stage == "eval":
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard")
        run_shard(args, base, out, plan)
    else:
        report(base, out, plan)


if __name__ == "__main__":
    main()
