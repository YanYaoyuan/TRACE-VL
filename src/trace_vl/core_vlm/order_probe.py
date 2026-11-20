"""A paired video-order diagnostic with a frozen Plain checkpoint."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from dataclasses import replace
from html import escape
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.positions import PositionCounter
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import checkpoint_identity
from .data import decode_observation, sha256
from .native import NativeVideoQALoss, encode_native, prepare_native
from .prepare import aliases, digest, write_json
from .qa import question_and_answer
from .temporal import attach_temporal
from .train import checked_checkpoint, tensor_digest

FRAMES_PER_ACTION = 8
PIXELS = 65_536
GRID = (4, 5)
FPS = 4
ARMS = ("plain", "single_frame", "text_only", "adapter")


def load_pairs(path):
    rows = []
    seen, groups, seen_content = set(), {}, set()
    source_hashes = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pair_id, group, split = row["id"], row["source_group"], row["split"]
        if (not isinstance(pair_id, str) or not pair_id.strip() or pair_id in seen
                or not isinstance(group, str) or not group.strip() or split not in {"train", "val"}):
            raise ValueError("unique pair ID, source group and explicit train/val split required")
        seen.add(pair_id)
        source = (path.parent / row["video"]).resolve()
        if source not in source_hashes:
            source_hashes[source] = sha256(source)
        fingerprint = source_hashes[source]
        for alias in {group, fingerprint, *aliases(source)}:
            previous = groups.setdefault(alias, (group, split))
            if previous != (group, split):
                raise ValueError("shared sources must use the same group and split")
        if row["ask"] not in {"first", "last"} or len(row["actions"]) != 2 or len(row["spans"]) != 2:
            raise ValueError("two action labels, two spans and first/last question type required")
        labels = [" ".join(label.split()) for label in row["actions"]]
        if not all(labels) or labels[0].casefold() == labels[1].casefold():
            raise ValueError("action labels must differ")
        (a, b), (c, d) = row["spans"]
        if not all(type(x) in {float, int} and math.isfinite(x) for x in (a, b, c, d)) or not 0 <= a < b < c < d:
            raise ValueError("non-overlapping chronological action spans required")
        content = (fingerprint, a, b, c, d)
        if content in seen_content:
            raise ValueError("same physical evidence spans appear more than once")
        seen_content.add(content)
        rows.append({**row, "actions": labels, "video": str(source), "video_sha256": fingerprint})
    if {r["split"] for r in rows} != {"train", "val"}:
        raise ValueError("train and held-out source groups are required")
    return sorted(rows, key=lambda r: r["id"])


def sample_pair(pair):
    halves, original = [], []
    for start, end in pair["spans"]:
        images, times, _, audit = decode_observation(
            {"id": pair["id"], "video": pair["video"], "query_time": end,
             "observation": {"start": start, "end": end}}, steps=FRAMES_PER_ACTION // 2, pixels=PIXELS)
        if audit["video_sha256"] != pair["video_sha256"]:
            raise ValueError(f"source changed: {pair['id']}")
        if len(set(map(int, audit["frame_indices"]))) != FRAMES_PER_ACTION:
            raise ValueError("each event must contribute eight distinct source frames")
        halves.append(images)
        original.append({"indices": audit["frame_indices"], "pts": times.reshape(-1).tolist(),
                         "rgb_sha256": audit["decoded_rgb_sha256"]})
    if original[0]["rgb_sha256"] == original[1]["rgb_sha256"]:
        raise ValueError(f"event observations are identical: {pair['id']}")
    # Review survives relocation of the exact same source video to another host.
    receipt = {"pair": {k: v for k, v in pair.items() if k != "video"}, "sampling": original}
    receipt["review_id"] = digest(receipt)
    return halves, receipt


def preview(path, out):
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(path)
    receipts, index = [], []
    for pair in pairs:
        halves, receipt = sample_pair(pair)
        canvas = Image.new("RGB", (128 * 8, 118 * 4 + 42), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"{pair['id']} | expected AB then BA | source timestamps (sec)", fill="black")
        draw.text((4, 22), f"A = {pair['actions'][0]}   B = {pair['actions'][1]}", fill="black")
        for i, k in enumerate((0, 1, 1, 0)):
            y = 42 + i * 118
            timestamps = receipt["sampling"][k]["pts"]
            for j, frame in enumerate(halves[k].reshape(-1, *halves[k].shape[2:])):
                thumb = Image.fromarray(frame)
                thumb.thumbnail((124, 92))
                canvas.paste(thumb, (128 * j, y))
                draw.text((128 * j + 2, y + 94), f"{'AB'[k]} {timestamps[j]:.2f}s", fill="black")
        canvas.save(out / (receipt["review_id"] + ".png"))
        receipts.append({"id": pair["id"], "split": pair["split"],
                         "review_id": receipt["review_id"], "approved": False, "reviewer": "",
                         "source_observation": receipt["sampling"]})
        title = escape(f"{pair['id']} · {pair['split']}")
        labels = escape(" → ".join(pair["actions"]))
        image = receipt["review_id"] + ".png"
        index.append(f'<article><h2>{title}</h2><p>{labels}</p><a href="{image}">'
                     f'<img loading="lazy" src="{image}" alt="{title}"></a></article>')
    report = out / "review.jsonl"
    if report.exists():
        old_ids = [json.loads(line)["review_id"] for line in report.read_text().splitlines() if line.strip()]
        if old_ids != [r["review_id"] for r in receipts]:
            raise ValueError("review inputs changed; use a new preview directory")
    else:
        atomic_text(report, "".join(json.dumps(r) + "\n" for r in receipts))
    atomic_text(out / "index.html", """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRACE-VL 顺序对照样本复核</title>
<style>
body{margin:2rem auto;max-width:1060px;padding:0 1rem;font:16px system-ui;color:#172536;background:#f7f8fa}
article{margin:1.5rem 0;padding:1rem;background:white;border:1px solid #d5dce3;border-radius:8px}
h2{font-size:1rem}img{display:block;width:100%;height:auto}
</style>
<h1>顺序对照样本复核</h1>
<p>每组第一、二行是 A→B，第三、四行是 B→A。核对两个动作在实际抽帧中是否都清楚可见。</p>
""" + "\n".join(index))
    print(f"PREVIEW: {len(receipts)} pairs, exact sampled frames available in {out}", flush=True)


def packed_view(pair, halves, swapped, *, single=False):
    # Same raw frames and timestamp schedule in both views. Only event block
    # order changes. Motion within each short span is kept intact.
    sequence = np.concatenate(halves[::-1] if swapped else halves)
    if single:
        flat = sequence.reshape(-1, *sequence.shape[2:])
        slot = int(digest([pair["id"], "single-frame"])[:8], 16) % len(flat)
        sequence = np.broadcast_to(flat[slot], flat.shape).copy().reshape(sequence.shape)
    indices = np.arange(sequence.shape[0] * sequence.shape[1])
    times = indices.reshape(-1, 2) / FPS
    actions = pair["actions"]
    options = actions[::(-1 if int(digest(pair["id"])[-1], 16) % 2 else 1)]
    gold_action = actions[int(swapped) ^ int(pair["ask"] == "last")]
    row = {"question": f"Which of these actions happened {pair['ask']}?",
           "options": options, "answer": options.index(gold_action), "query_time": float(times[-1, -1])}
    audit = {"source_fps": FPS, "source_frame_count": len(indices), "frame_indices": indices.tolist(),
             "decoded_rgb_sha256": hashlib.sha256(sequence.tobytes()).hexdigest()}
    return sequence, times, row, audit


def encode_views(bb, pair, halves, *, single=False):
    output = []
    for swapped in (False, True):
        images, times, row, audit = packed_view(pair, halves, swapped, single=single)
        layout = prepare_native(bb.processor, images, times, row, audit)
        steps = [replace(step, tokens=step.tokens.cpu()) for step in encode_native(bb, layout)]
        layout.inputs.pop("pixel_values_videos")
        output.append((steps, {**row, "native_layout": layout}))
    return output


def fit(net, samples, updates, seed, lr):
    net.requires_grad_(False)
    net.eval()
    net.temporal.requires_grad_(True)
    net.temporal.train()
    optimizer = torch.optim.AdamW(net.temporal.parameters(), lr=lr, weight_decay=0.)
    rng = np.random.default_rng(seed)
    for step in range(updates):
        if step % len(samples) == 0:
            order = rng.permutation(len(samples))
        visuals, rows = zip(*samples[int(order[step % len(samples)])], strict=True)
        optimizer.zero_grad(set_to_none=True)
        # Frozen LM remains in the autograd graph from answer CE to visual residual.
        loss, _ = net(list(visuals), list(rows))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.temporal.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss) or norm <= 0:
            raise ValueError("answer CE did not produce finite nonzero adapter gradients")
        if step > 0 and not net.temporal.down.weight.grad.count_nonzero():
            raise ValueError("gradient did not reach the adapter's internal projection")
        optimizer.step()
        yield {"update": step + 1, "answer_ce": float(loss.detach()), "adapter_grad_norm": float(norm)}
    net.eval()


@torch.no_grad()
def choice(net, view, *, text_only=False):
    steps, row = view
    bb = net.bb
    if text_only:
        question, _ = question_and_answer(row)
        prompt = bb.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": question}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = bb.ids(prompt)
        embeds = bb.embed_ids(ids)
        positions = PositionCounter().text(embeds.shape[1], bb.device)
    else:
        embeds, positions, _, _, _ = net.prefix(steps, row)
    ids = [bb.ids(label) for label in ("A", "B")]
    if any(len(v) != 1 for v in ids):
        raise ValueError("A/B must tokenize to single letters")
    hidden = bb.lm(embeds, positions, cache=None, use_cache=False)
    logits = bb.logits(hidden[:, -1:]).float()
    if not logits.isfinite().all():
        raise ValueError("nonfinite next-token logits")
    return int(logits[0, 0, [x[0] for x in ids]].argmax())


def summarize(predictions):
    source_groups = sorted({r["source_group"] for r in predictions})
    report = {"pairs": len(predictions), "source_groups": len(source_groups), "arms": {}, "paired": {}}
    outcomes = {}
    for name in ARMS:
        correctness = np.array([[p == g for p, g in zip(r[name], r["gold"], strict=True)]
                                for r in predictions])
        outcomes[name] = correctness.all(axis=1)
        report["arms"][name] = {"question_accuracy": float(correctness.mean()),
                                "pair_accuracy": float(outcomes[name].mean()),
                                "prediction_flip_rate": float(np.mean([r[name][0] != r[name][1] for r in predictions]))}
    mask = np.array([[r["source_group"] == key for r in predictions] for key in source_groups])
    n = mask.sum(axis=1)
    draws = np.random.default_rng(7).integers(len(n), size=(4000, len(n)))
    for arm in ARMS[:-1]:
        delta = outcomes["adapter"].astype(np.int64) - outcomes[arm]
        sums = mask @ delta
        ci = np.quantile(100 * sums[draws].sum(axis=1) / n[draws].sum(axis=1), [.025, .975])
        report["paired"][f"adapter_minus_{arm}"] = {
            "delta_pp": float(100 * delta.mean()), "ci95_pp": ci.tolist() if len(n) > 1 else None,
            "unit": "source-group cluster", "bootstrap_samples": 4000}
    return report


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(args, out)


def _run(args, out):
    pairs = load_pairs(args.pairs)
    approvals = {}
    for line in args.review.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] in approvals:
                raise ValueError("duplicate pair review")
            approvals[row["id"]] = row
    prepared = []
    for pair in pairs:
        halves, receipt = sample_pair(pair)
        review = approvals.get(pair["id"], {})
        reviewer = review.get("reviewer")
        if (review.get("review_id") != receipt["review_id"] or review.get("approved") is not True
                or not isinstance(reviewer, str) or not reviewer.strip()):
            raise ValueError(f"verify both action spans on the sampled contact sheet: {pair['id']}")
        prepared.append((pair, halves, receipt["review_id"]))

    checkpoint = args.plain.resolve()
    parent = json.loads((checkpoint.parent / "run.json").read_text())
    complete = checked_checkpoint(checkpoint, parent["id"])
    cfg = parent["config"]
    if cfg.get("temporal") != "none" or cfg["mode"] != "grid":
        raise ValueError("a verified plain grid checkpoint is required")
    if cfg["updates"] != complete["update"]:
        raise ValueError("use the fixed final Plain checkpoint")
    if (cfg["steps"], cfg["pixels"], tuple(cfg["pool_grid"])) != (8, PIXELS, GRID):
        raise ValueError("Plain checkpoint's observation protocol differs")
    if checkpoint_identity(args.model) != parent["model"]:
        raise ValueError("base weights differ from the verified Plain checkpoint")
    previous_sources = {key for path in parent["videos"] for key in aliases(path)}
    previous_media = set(parent["videos"].values())
    overlaps = [p["id"] for p in pairs if p["split"] == "val" and (
        p["video_sha256"] in previous_media
        or bool(({p["source_group"]} | aliases(p["video"])) & previous_sources))]
    if overlaps:
        raise ValueError(f"held-out pairs reuse Plain training sources: {overlaps}")
    identity = {"checkpoint": complete, "source_pair_count": len(pairs),
                "review_ids": [receipt for _, _, receipt in prepared],
                "review_sha256": sha256(args.review), "updates": args.updates,
                "seed": args.seed, "lr": args.lr,
                "code": {p: sha256(Path(__file__).with_name(p)) for p in
                         ("order_probe.py", "data.py", "qa.py", "temporal.py", "native.py")}}
    identity["id"] = digest(identity)
    ensure_run_manifest(out / "run.json", identity)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    bb = Backbone(BackboneConfig(name=args.model, dtype="bfloat16", local_files_only=True),
                  StreamConfig(pixels_per_frame=PIXELS), device=args.device)
    bb.add_lora(cfg["rank"], 2 * cfg["rank"], dropout=0.0, adapter=str(checkpoint / "adapter"))
    frozen = bb.model.eval().requires_grad_(False)
    lora = {name: value for name, value in frozen.named_parameters() if "lora_" in name}
    before = tensor_digest(lora)
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=GRID, answer_end_token_id=bb.tokenizer.eos_token_id)
    train, heldout = [], []
    for pair, halves, _ in prepared:
        views = encode_views(bb, pair, halves)
        if pair["split"] == "train":
            train.append(views)
            continue
        single = encode_views(bb, pair, halves, single=True)
        pred = {"id": pair["id"], "source_group": pair["source_group"],
                "gold": [row["answer"] for _, row in views],
                "plain": [choice(net, v) for v in views],
                "single_frame": [choice(net, v) for v in single],
                "text_only": [choice(net, v, text_only=True) for v in views]}
        heldout.append((views, pred))
    attach_temporal(net, "causal")
    metrics = list(fit(net, train, args.updates, args.seed, args.lr))
    if not lora or before != tensor_digest(lora) or any(p.grad is not None for p in frozen.parameters()):
        raise ValueError("frozen LM/LoRA audit failed")
    predictions = [{**pred, "adapter": [choice(net, v) for v in views]} for views, pred in heldout]
    report = {**summarize(predictions), "frozen_plain": True, "training_pairs": len(train),
              "run_id": identity["id"], "scope": "order diagnostic on paired constructed views"}
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in net.temporal.state_dict().items()},
              str(out / "temporal.safetensors"))
    for name, rows in (("metrics.jsonl", metrics), ("predictions.jsonl", predictions)):
        atomic_text(out / name, "".join(json.dumps(r) + "\n" for r in rows))
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest="stage", required=True)
    inspect = stages.add_parser("preview", help="publish exact sampled frames for human review")
    train = stages.add_parser("run", help="compare frozen Plain with a fitted two-layer adapter")
    for cmd in (inspect, train):
        cmd.add_argument("--pairs", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
    train.add_argument("--review", type=Path, required=True)
    train.add_argument("--plain", type=Path, required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--updates", type=int, default=100)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    if args.stage == "preview":
        preview(args.pairs, args.out)
    else:
        if args.updates < 2 or not math.isfinite(args.lr) or args.lr <= 0:
            p.error("at least two updates and a positive finite learning rate are required")
        run(args)


if __name__ == "__main__":
    main()
