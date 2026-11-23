"""One queue for aligned data, evidence ceiling, and matched history readers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.experiment import paired
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VideoQALoss, VisualStep, question_and_answer
from trace_vl.streaming_vision.data import read_rows
from trace_vl.streaming_vision.experiment import atomic_checkpoint, decoded, patch_inputs
from trace_vl.streaming_vision.media import VideoProbeCache
from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .data import prepare_activitynet
from .model import IntervalReader, VisualHistory, evidence_indices

ARMS = ("cosine", "learned")
POPULATIONS = ("train", "dev", "ovo")
ORACLE = ("recent", "text_only", "oracle", "distractor")


def prepare(args, root):
    settings = {k: v for k, v in vars(args).items()
                if k not in {"stage", "out", "split", "shard", "shards", "arm", "workers", "scratch", "device"}}
    identity = {"settings": settings, "activitynet_sha256": sha256(args.annotation),
                "ovo_sha256": sha256(args.ovo) if args.ovo else None,
                "model": checkpoint_identity(args.model), "revision": os.environ.get("TRACE_REV", "local")}
    identity["id"] = digest(identity)
    ensure_run_manifest(root / "inputs.json", identity)
    if (root / "prepared.json").exists():
        load(root, args)
        return
    count = prepare_activitynet(args, root)
    write_json(root / "prepared.json", {"run": identity["id"], "counts": count,
               "manifests": {split: sha256(root / f"{split}.jsonl") for split in POPULATIONS}})


def load(root, args):
    manifest = json.loads((root / "inputs.json").read_text())
    ready = json.loads((root / "prepared.json").read_text())
    if manifest["id"] != ready["run"]:
        raise ValueError("data belongs to another experiment")
    for split, checksum in ready["manifests"].items():
        if sha256(root / f"{split}.jsonl") != checksum:
            raise ValueError(f"frozen {split} manifest changed")
    for key, value in manifest["settings"].items():
        setattr(args, key, value)
    return manifest["id"]


def option_heads(bb, row):
    ids = [bb.ids(chr(65 + i)) for i in range(len(row["options"]))]
    if any(len(tokens) != 1 for tokens in ids) or bb.lm_head.bias is not None:
        raise ValueError("the Qwen choice scorer requires a biasless head and single-token options")
    return bb.lm_head.weight[[tokens[0] for tokens in ids]].float()


def encode_clip(args, bb, images, times):
    pixels, _, h, w = patch_inputs(bb, images)
    shape = (h // bb.visual.spatial_merge_size, w // bb.visual.spatial_merge_size)
    bank = VisualHistory(args.recent, args.block, args.capacity)
    for begin in range(0, len(pixels), args.vision_batch):
        chunk = pixels[begin:begin + args.vision_batch]
        grid = torch.tensor([[len(chunk), h, w]], device=bb.device)
        xn = bb.encode(torch.cat(chunk), grid).split(shape[0] * shape[1])
        for feat, pair in zip(xn, times[begin:begin + len(chunk)], strict=True):
            bank.append(VisualStep(feat.clone(), shape, float(pair[-1]), tuple(float(t) for t in pair)))
    return bank


def query_hidden(bb, qa, row, steps):
    if steps:
        embeds, positions, *_ = qa.prefix(steps, row)
    else:
        question, _ = question_and_answer(row)
        ids = bb.ids(bb.head_text + question + bb.tail_text)
        embeds = bb.embed_ids(ids)
        positions = torch.arange(len(ids), device=bb.device).expand(3, -1)
    return bb.lm(embeds, positions, cache=None, use_cache=False)[0, -1].float()


@torch.no_grad()
def clip_features(args, bb, qa, observation, *, controls):
    images, times, row, audit = observation
    state = encode_clip(args, bb, images, times)
    recent = list(state.recent)
    heads = option_heads(bb, row)
    query = query_hidden(bb, qa, row, recent)
    slots, spans = state.tensors()
    targets = None
    predictions = {"recent": int((heads @ query).argmax())}
    if row.get("evidence") and row["evidence"][1] < recent[0].frame_times[0]:
        targets = evidence_indices(state.slots, row["evidence"], row["id"])
    if controls and targets is not None:
        target, other = targets
        for arm, pick in (("oracle", target), ("distractor", other)):
            previous = list(state.slots[pick])
            hidden = query_hidden(bb, qa, row, previous + recent)
            predictions[arm] = int((heads @ hidden).argmax())
        predictions["text_only"] = int((heads @ query_hidden(bb, qa, row, [])).argmax())
    return {
        "id": row["id"], "query": query.cpu(), "heads": heads.cpu(), "memory": slots.cpu(),
        "spans": spans, "relative_times": (spans - row["query_time"]) / args.lookback,
        "gold_slot": targets[0] if targets is not None else -1, "baseline": predictions,
        "control_slot": targets[1] if targets is not None else -1,
        "sampled_times": times.tolist(), "video_sha256": audit["video_sha256"],
        "frame_indices": audit["frame_indices"], "timestamp_filter": audit.get("timestamp_filter"),
        "recent_tokens": sum(len(step.tokens) for step in recent),
        "history_tokens": int(slots.shape[0] * slots.shape[1]),
    }


def feature_path(root, split, row):
    return root / "features" / split / f"{digest(row['id'])}.pt"


def cache_features(args, root, run):
    rows = read_rows(root / f"{args.split}.jsonl")[args.shard::args.shards]
    pending = [(row, feature_path(root, args.split, row)) for row in rows]
    pending = [(row, path) for row, path in pending if not path.exists()]
    if not pending:
        return
    device = torch.device(args.device)
    bb = Backbone(BackboneConfig(name=args.model, dtype=args.dtype, local_files_only=True),
                  StreamConfig(pixels_per_frame=args.pixels), device=device)
    qa = VideoQALoss(bb, mode="full").eval()
    cache = VideoProbeCache(args.scratch or root / "media-cache",
                            stage_media=os.environ.get("TRACE_HISTORY_STAGE_MEDIA") == "1")
    for index, ((row, path), observation) in enumerate(zip(
        pending, decoded(args, [r for r, _ in pending], cache, drop_bad_pts=True), strict=True,
    ), 1):
        features = clip_features(args, bb, qa, observation, controls=args.split == "dev")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_checkpoint(path, {**features, "run": run})
        if features["timestamp_filter"]["dropped_frames"]:
            print(f"PTS FILTER {row['id']} {features['timestamp_filter']}", flush=True)
        if index == 1 or index % 10 == 0 or index == len(pending):
            print(f"CACHE {args.split} {args.shard} {index}/{len(pending)} {row['id']}", flush=True)


def feature_rows(root, split, run):
    rows = read_rows(root / f"{split}.jsonl")
    result = []
    for row in rows:
        path = feature_path(root, split, row)
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["run"] != run or item["id"] != row["id"]:
            raise ValueError(f"stale features: {path}")
        if any(t > row["query_time"] for pair in item["sampled_times"] for t in pair):
            raise ValueError(f"future frames found: {row['id']}")
        result.append((row, item))
    return result


def choices(items, answer, arms):
    return {arm: {row["id"]: {"answer": chr(65 + answer(item, arm))} for row, item in items} for arm in arms}


def timestamp_audit(items):
    return {row["id"]: audit for row, item in items
            if (audit := item.get("timestamp_filter")) and audit["dropped_frames"]}


def oracle_report(root, run):
    candidates = feature_rows(root, "dev", run)
    items = [(row, item) for row, item in candidates if item["gold_slot"] >= 0]
    rows = [r for r, _ in items]
    results = choices(items, lambda item, arm: item["baseline"][arm], ORACLE)
    stats = {arm: {"n": len(items), "correct": sum(
        arm_results[row["id"]]["answer"] == chr(65 + row["answer"]) for row, _ in items)}
        for arm, arm_results in results.items()}
    pairs = {arm: paired(rows, results["oracle"], results[arm]) for arm in ("distractor", "recent", "text_only")}
    passed = bool(len(rows) >= 24 and all(
        d and d["ci95_pp"] and d["ci95_pp"][0] > 0 and d["delta_pp"] >= 5 for d in pairs.values()
    ))
    data = {"run": run, "eligible": len(rows), "prepared_dev": len(candidates), "scores": stats,
            "code_revision": os.environ.get("TRACE_HISTORY_CODE_REV"),
            "timestamp_filter_applied": timestamp_audit(candidates),
            "excluded_ids": [r["id"] for r, d in candidates if d["gold_slot"] < 0],
            "paired": pairs, "passed": passed,
            "rule": "At least 24 eligible videos; oracle gains >=5 pp over all controls, 95% lower bound >0"}
    write_json(root / "oracle.json", data)
    atomic_text(root / "oracle-predictions.jsonl", "".join(
        json.dumps({"id": row["id"], "source_group": row["source_group"], "answer": row["answer"],
                    "choices": item["baseline"], "evidence": row["evidence"],
                    "oracle_span": item["spans"][item["gold_slot"]].tolist(),
                    "distractor_span": item["spans"][item["control_slot"]].tolist(),
                    "sampled_times": item["sampled_times"], "frame_indices": item.get("frame_indices"),
                    "video_sha256": item["video_sha256"]}) + "\n"
        for row, item in items
    ))
    print("ORACLE", json.dumps(data), flush=True)
    return passed


def reader_forward(model, item, device):
    query = item["query"].to(device)
    memory = item["memory"].to(device, torch.float32)
    bounds = item["relative_times"].to(device)
    heads = item["heads"].to(device)
    return model(query, memory, bounds, heads=heads)


def fit(args, root, run):
    rows = [(r, f) for r, f in feature_rows(root, "train", run) if f["gold_slot"] >= 0]
    if not rows:
        raise ValueError("no aligned train questions with an older evidence interval")
    checkpoint = root / f"{args.arm}.pt"
    expected = digest([run, args.arm, [r["id"] for r, _ in rows]])
    if checkpoint.exists():
        info = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if info["identity"] != expected:
            raise ValueError("saved reader uses different training inputs")
        return
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    hidden = rows[0][1]["query"].numel()
    model = IntervalReader(hidden, args.width, args.arm).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    logs = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(rows))
        correct, total_ce, interval_ce = 0, 0., 0.
        for offset in range(0, len(order), args.batch):
            optim.zero_grad(set_to_none=True)
            batch = order[offset:offset + args.batch]
            for index in batch:
                row, item = rows[index]
                logits, scores, _ = reader_forward(model, item, device)
                ce = F.cross_entropy(logits[None], torch.tensor([row["answer"]], device=device))
                locator = (F.cross_entropy(scores[None], torch.tensor([item["gold_slot"]], device=device))
                           if args.arm == "learned" else ce.new_zeros(()))
                ((ce + args.span_weight * locator) / len(batch)).backward()
                correct += int(logits.argmax() == row["answer"])
                total_ce += float(ce.detach())
                interval_ce += float(locator.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optim.step()
        log = {"epoch": epoch + 1, "examples": len(rows), "accuracy": correct / len(rows),
               "choice_ce": total_ce / len(rows), "interval_ce": interval_ce / len(rows)}
        logs.append(log)
        write_json(root / f"train-{args.arm}.json", logs)
        print("TRAIN", args.arm, json.dumps(log), flush=True)
    atomic_checkpoint(checkpoint, {"identity": expected, "run": run, "mode": args.arm, "hidden": hidden,
                                   "state": model.state_dict()})


@torch.no_grad()
def final_report(args, root, run):
    device = torch.device(args.device)
    models = {}
    for arm in ARMS:
        weights = torch.load(root / f"{arm}.pt", map_location="cpu", weights_only=True)
        if weights["run"] != run or weights["mode"] != arm:
            raise ValueError("reader checkpoint identity differs")
        model = IntervalReader(weights["hidden"], args.width, arm).to(device)
        model.load_state_dict(weights["state"])
        models[arm] = model.eval()
    results = {}
    for split in ("dev", "ovo"):
        features = feature_rows(root, split, run)
        items = [(row, item) for row, item in features if split == "ovo" or item["gold_slot"] >= 0]
        if not items:
            continue
        rows, predictions = [], []
        for row, item in items:
            arms = {"recent": item["baseline"]["recent"]}
            spans = {}
            for arm, model in models.items():
                logits, _, index = reader_forward(model, item, device)
                arms[arm] = int(logits.argmax())
                spans[arm] = item["spans"][int(index)].tolist() if index is not None else None
            rows.append(row)
            predictions.append({"id": row["id"], "task": row["task"], "answer": row["answer"],
                                "source_group": row.get("source_group", row["video"]),
                                "choice": arms, "spans": spans,
                                "gold_slot": item["gold_slot"], "slots": len(item["spans"]),
                                "recent_tokens": item["recent_tokens"], "history_tokens": item["history_tokens"],
                                "history_tokens_read": (args.block * 4 if len(item["spans"]) else 0),
                                "video_sha256": item["video_sha256"]})
        groups = {"all": rows}
        for task in sorted({r["task"] for r in rows}):
            groups[task] = [r for r in rows if r["task"] == task]
        if split == "ovo":
            groups["backward"] = [r for r in rows if r["task"] in {"EPM", "ASI", "HLD"}]
            groups["realtime"] = [r for r in rows if r["task"] in {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}]
        lookup = {p["id"]: p for p in predictions}
        maps = {arm: {p["id"]: {"answer": chr(65 + p["choice"][arm])} for p in predictions}
                for arm in ("recent", *ARMS)}
        scores = {group: {arm: sum(lookup[r["id"]]["choice"][arm] == r["answer"] for r in subset) / len(subset)
                          for arm in maps} for group, subset in groups.items() if subset}
        comparison = {group: paired(subset, maps["learned"], maps["cosine"])
                      for group, subset in groups.items() if subset}
        recent_comparison = {group: paired(subset, maps["learned"], maps["recent"])
                             for group, subset in groups.items() if subset}
        hits = {}
        for arm in ARMS:
            labelled = [(r, p) for r, p in zip(rows, predictions, strict=True) if p["gold_slot"] >= 0]
            if labelled:
                count = 0
                for row, sample in labelled:
                    proposed = sample["spans"][arm]
                    interval = row["evidence"]
                    count += int(proposed is not None and min(proposed[1], interval[1]) >
                                 max(proposed[0], interval[0]))
                hits[arm] = {"n": len(labelled), "overlap_hits": count, "rate": count / len(labelled)}
        results[split] = {"n": len(rows), "accuracy": scores, "learned_minus_cosine": comparison,
                          "learned_minus_recent": recent_comparison,
                          "timestamp_filter_applied": timestamp_audit(features),
                          "interval_hit": hits, "empty_history": sum(p["slots"] == 0 for p in predictions)}
        atomic_text(root / f"{split}-predictions.jsonl",
                    "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in predictions))
    main = results["dev"]["learned_minus_cosine"]["all"]
    ovo = results.get("ovo", {}).get("learned_minus_cosine", {})
    guard = {key: ovo[key]["ci95_pp"] is not None and ovo[key]["ci95_pp"][0] > -2
             for key in ("realtime", "OCR") if key in ovo}
    passed = bool(main["delta_pp"] >= 2 and main["ci95_pp"] and main["ci95_pp"][0] > 0
                  and len(guard) == 2 and all(guard.values()))
    result = {"run": run, "datasets": results, "passed": passed, "ovo_no_harm": guard,
              "code_revision": os.environ.get("TRACE_HISTORY_CODE_REV"),
              "trainable_parameters": {arm: sum(p.numel() for p in model.parameters())
                                       for arm, model in models.items()},
              "visual_token_budget": {"history_capacity": args.capacity * args.block * 4,
                                      "max_read_per_question": args.block * 4},
              "decision": "ActivityNet old-evidence +2 pp, paired CI lower >0; OVO realtime and OCR CI lower >−2 pp",
              "scope": "Frozen native Qwen vision and text, single decoder-output cross-attention residual. "
                       "ActivityNet caption-derived controlled QA; OVO development MC only."}
    write_json(root / "report.json", result)
    lines = ["# TRACE-VL evidence interval reader", "", result["scope"], ""]
    for name, entry in results.items():
        lines += [f"## {name} ({entry['n']} questions)", ""]
        for group in ("all", "backward", "realtime", "OCR"):
            if group in entry["accuracy"]:
                scores = entry["accuracy"][group]
                contrast = entry["learned_minus_cosine"][group]
                lines.append(f"- {group} {scores}. Learned − cosine {contrast['delta_pp']:+.2f} pp "
                             f"(95% CI {contrast['ci95_pp']}).")
        lines += [f"- Labelled interval overlap {entry['interval_hit']}.", ""]
    lines.extend([f"Decision passed: {passed}.", result["decision"]])
    atomic_text(root / "REPORT.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "cache", "oracle", "fit", "report"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--model")
    parser.add_argument("--annotation")
    parser.add_argument("--media", nargs="+", default=[])
    parser.add_argument("--ovo")
    parser.add_argument("--ovo-media")
    parser.add_argument("--allow-annotation-fixture", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    parser.add_argument("--scratch")
    parser.add_argument("--split", choices=POPULATIONS, default="dev")
    parser.add_argument("--arm", choices=ARMS, default="learned")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--dev-limit", type=int, default=192)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--archive-workers", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--recent", type=int, default=8)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=22)
    parser.add_argument("--vision-batch", type=int, default=8)
    parser.add_argument("--pixels", type=int, default=65536)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--span-weight", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        if not args.annotation or not args.model or not args.media:
            parser.error("prepare requires --annotation, --model and --media")
        if (min(args.recent, args.block, args.capacity, args.vision_batch, args.width, args.epochs, args.batch,
                args.workers, args.archive_workers) < 1
                or min(args.train_limit, args.dev_limit) < 0
                or not 0 < args.lr < float("inf") or not 0 <= args.span_weight < float("inf")
                or args.lookback < args.recent + 2 * args.block):
            parser.error("invalid experiment settings; lookback must include the recent window and two old blocks")
        prepare(args, root)
        return
    run = load(root, args)
    args.hz = 2.
    if not 0 <= args.shard < args.shards or args.workers < 1:
        parser.error("invalid shard or worker count")
    if args.stage == "cache":
        cache_features(args, root, run)
    elif args.stage == "oracle":
        raise SystemExit(0 if oracle_report(root, run) else 3)
    elif args.stage == "fit":
        fit(args, root, run)
    else:
        final_report(args, root, run)


if __name__ == "__main__":
    main()
