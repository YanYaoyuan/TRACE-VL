"""Matched A/B state-direction experiment with a verified, frozen Plain model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import checkpoint_identity
from .data import decode_observation, sha256
from .native import NativeVideoQALoss, encode_native, prepare_native
from .order_probe import GRID, PIXELS, load_pairs, packed_view, sample_pair
from .prepare import aliases, digest, write_json
from .state_attention import (
    READERS,
    OnlineBoxMatcher,
    StateAttention,
    StatePairBuffer,
    answer_logits,
    region_feature,
)
from .train import checked_checkpoint, tensor_digest

ARMS = ("plain", *READERS)
BOX = (0., 0., 1., 1.)
STEPS = 8


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def approved_order_samples(path, review):
    approvals = {row["id"]: row for row in read_jsonl(review)}
    for pair in load_pairs(path):
        halves, receipt = sample_pair(pair)
        approval = approvals.get(pair["id"], {})
        if (approval.get("review_id") != receipt["review_id"]
                or approval.get("approved") is not True or not approval.get("reviewer", "").strip()):
            raise ValueError(f"unreviewed sampled frames: {pair['id']}")
        views = []
        for swapped in (False, True):
            images, times, row, audit = packed_view(pair, halves, swapped)
            # The bundled action review supplies no boxes. This is a full-frame
            # engineering control with two annotated spans, not a tracked camera result.
            boxes = [{} for _ in range(STEPS)]
            boxes[1] = boxes[5] = {"full_frame": BOX}
            views.append((images, times, row, audit, boxes, STEPS))
        yield pair, views


def natural_manifest(path):
    """Natural opposite directions, with per-step tracks from an external annotator/tracker."""
    groups, ids, repeats = {}, set(), set()
    hashes = {}
    out = []
    for row in read_jsonl(path):
        key, group, split = row["id"], row["source_group"], row["split"]
        if not all(isinstance(s, str) and s.strip() for s in (key, group)) or key in ids:
            raise ValueError("unique nonempty pair and source-group IDs required")
        if split not in ("train", "val") or len(row["views"]) != 2:
            raise ValueError("two natural views and an explicit train/val split required")
        if any(type(v["answer"]) is not int for v in row["views"]) or sorted(
                v["answer"] for v in row["views"]) != [0, 1]:
            raise ValueError("paired views must have opposite binary answers")
        options = row["options"]
        if (len(options) != 2 or any(not isinstance(o, str) or not o.strip() for o in options)
                or options[0].strip().casefold() == options[1].strip().casefold()
                or not isinstance(row["question"], str) or not row["question"].strip()):
            raise ValueError("nonempty fixed question and two distinct options required")
        ids.add(key)
        for view in row["views"]:
            media = (path.parent / view["video"]).resolve()
            if media not in hashes:
                hashes[media] = sha256(media)
            video_hash = hashes[media]
            view.update(video=str(media), video_sha256=video_hash)
            for alias in {group, video_hash, *aliases(media)}:
                if groups.setdefault(alias, (group, split)) != (group, split):
                    raise ValueError(f"source reused across groups or splits: {alias}")
            start, end = view["span"]
            query = view["query_time"]
            if not all(type(x) in (int, float) and math.isfinite(x) for x in (start, end, query)):
                raise ValueError("finite span and query timestamps required")
            if not 0 <= start < end <= query:
                raise ValueError("natural observations must end at or before the query")
            observed = video_hash, start, end
            if observed in repeats:
                raise ValueError("duplicate natural evidence interval")
            repeats.add(observed)
            if ("tracks" in view) == ("boxes" in view):
                raise ValueError("supply either tracked boxes or untracked box proposals")
            tracks = view.get("tracks", view.get("boxes"))
            if len(tracks) != STEPS:
                raise ValueError(f"expected one track map for each of the {STEPS} sampled steps")
            visible = view.get("visible_steps", STEPS)
            if type(visible) is not int or not 1 <= visible <= STEPS:
                raise ValueError("visible_steps must be between 1 and 8")
            for observation in tracks:
                if not isinstance(observation, dict if "tracks" in view else list):
                    raise ValueError("expected track-to-box maps or lists of boxes")
                items = observation.items() if isinstance(observation, dict) else enumerate(observation, 1)
                if len(observation) > 32:
                    raise ValueError("at most 32 tracked/proposed boxes per step")
                for track, (x1, y1, x2, y2) in items:
                    if not track or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                        raise ValueError("invalid track ID or normalized xyxy box")
        out.append(row)
    if {r["split"] for r in out} != {"train", "val"}:
        raise ValueError("natural manifest requires source-disjoint train and validation pairs")
    return sorted(out, key=lambda row: row["id"])


def natural_samples(rows):
    for pair in rows:
        views = []
        for view in pair["views"]:
            start, end = view["span"]
            record = {"id": pair["id"], "video": view["video"], "query_time": view["query_time"],
                      "observation": {"start": start, "end": end}, "question": pair["question"],
                      "options": pair["options"], "answer": view["answer"]}
            images, times, record, audit = decode_observation(record, steps=STEPS, pixels=PIXELS)
            if audit["video_sha256"] != view["video_sha256"]:
                raise ValueError(f"video changed during preparation: {pair['id']}")
            views.append((images, times, record, audit, view.get("tracks", view.get("boxes")),
                          view.get("visible_steps", STEPS)))
        yield pair, views


def verify_plain(model, checkpoint):
    parent = json.loads((checkpoint.parent / "run.json").read_text())
    complete = checked_checkpoint(checkpoint, parent["id"])
    cfg = parent["config"]
    if (cfg["mode"], cfg["temporal"], cfg["steps"], cfg["pixels"], tuple(cfg["pool_grid"])) != (
            "grid", "none", STEPS, PIXELS, GRID):
        raise ValueError("expected a verified Plain grid checkpoint on native 16-frame inputs")
    if complete["update"] != cfg["updates"] or checkpoint_identity(model) != parent["model"]:
        raise ValueError("final checkpoint and its exact Qwen base are required")
    return parent, complete


def source_overlap(pairs, parent):
    source_ids = {alias for path in parent["videos"] for alias in aliases(path)}
    source_hashes = set(parent["videos"].values())
    for pair in pairs:
        if pair["split"] != "val":
            continue
        for view in pair.get("views", (pair,)):
            if (view["video_sha256"] in source_hashes
                    or ({pair["source_group"]} | aliases(view["video"])) & source_ids):
                raise ValueError(f"validation pair overlaps Plain training: {pair['id']}")


def load_backbone(model, checkpoint, device, rank):
    dtype = "float32" if torch.device(device).type == "cpu" else "bfloat16"
    bb = Backbone(BackboneConfig(name=model, dtype=dtype, local_files_only=True),
                  StreamConfig(pixels_per_frame=PIXELS), device=device)
    bb.add_lora(rank, 2 * rank, dropout=0., adapter=str(checkpoint / "adapter"))
    bb.model.eval().requires_grad_(False)
    return bb


@torch.no_grad()
def encode_view(bb, net, view, capacity, natural):
    images, times, row, audit, boxes, visible = view
    layout = prepare_native(bb.processor, images, times, row, audit,
                            allow_native_timestamp_approximation=natural)
    steps = encode_native(bb, layout)
    buffer = StatePairBuffer(capacity)
    matcher = OnlineBoxMatcher()
    for index, (step, tracks) in enumerate(zip(steps, boxes, strict=True)):
        if isinstance(tracks, dict):
            observed = {name: region_feature(step.tokens, step.grid, box) for name, box in tracks.items()}
        else:
            observed = matcher.update(index, [region_feature(step.tokens, step.grid, box) for box in tracks])
        buffer.observe(step.end, observed)
    pairs = buffer.read(row["query_time"])
    if not pairs:
        raise ValueError("each view requires at least two observations of one track")
    history_audit = layout.audit
    if visible < len(steps):
        recent = images[-visible:]
        recent_audit = {**audit, "frame_indices": audit["frame_indices"][-2 * visible:],
                        "decoded_rgb_sha256": hashlib.sha256(recent.tobytes()).hexdigest()}
        layout = prepare_native(bb.processor, recent, times[-visible:], row, recent_audit,
                                allow_native_timestamp_approximation=natural)
    embeds, positions, _, _, metadata = net.prefix(steps[-visible:], {**row, "native_layout": layout})
    options = [bb.ids(label) for label in ("A", "B")]
    if any(len(ids) != 1 for ids in options):
        raise ValueError("the frozen tokenizer must encode A/B as single tokens")
    tensor = torch.stack([torch.stack((p.before, p.after)) for p in pairs]).cpu()
    return {"embeds": embeds[0].cpu(), "positions": positions.cpu(), "pairs": tensor,
            "answer": row["answer"], "choices": [ids[0] for ids in options],
            "evidence": [[p.track, p.t_before, p.t_after] for p in pairs],
            "observation": {**layout.audit, **metadata}, "history_sampling": history_audit}


def verify_features(out, run):
    receipt = json.loads((out / "features.json").read_text())
    if receipt["run"] != run["id"] or receipt["sha256"] != sha256(out / "features.pt"):
        raise ValueError("shared feature cache does not match the experiment")
    return receipt


def prepare(args):
    import transformers

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    parent, complete = verify_plain(args.model, args.plain)
    natural = natural_manifest(args.natural) if args.natural else []
    bundled = load_pairs(args.pairs)
    source_overlap([*bundled, *natural], parent)
    if natural:
        # The bundled diagnostic may use the same underlying source only as a
        # separate diagnostic. In particular, do not learn from its validation videos.
        sources = {alias for r in bundled if r["split"] == "val"
                   for alias in {r["source_group"], r["video_sha256"], *aliases(r["video"])}}
        for pair in natural:
            if pair["split"] == "train" and any(
                    {v["video_sha256"], *aliases(v["video"]), pair["source_group"]} & sources
                    for v in pair["views"]):
                raise ValueError("natural training reuses bundled validation media")
    source = Path(__file__).parent
    run = {"protocol": "paired-state-attention-v1", "checkpoint": complete,
           "model": str(Path(args.model).resolve()), "plain": str(args.plain.resolve()),
           "rank": parent["config"]["rank"], "updates": args.updates, "lr": args.lr,
           "seed": args.seed, "width": args.width, "capacity": args.capacity, "batch": args.batch,
           "torch": str(torch.__version__), "transformers": transformers.__version__,
           "suites": ["constructed", "natural"] if natural else ["constructed"],
           "inputs": {"paired_order": sha256(args.pairs), "review": sha256(args.review),
                      "natural": sha256(args.natural) if natural else None,
                      "videos": sorted({r["video_sha256"] for r in bundled} |
                                       {v["video_sha256"] for r in natural for v in r["views"]})},
           "code": {str(p.relative_to(source.parent)): sha256(p)
                    for p in sorted([*source.glob("*.py"), (source.parent / "v3/backbone.py")])}}
    run["id"] = digest(run)
    ensure_run_manifest(out / "run.json", run)
    if (out / "features.json").exists():
        verify_features(out, run)
        print("REUSE shared features", flush=True)
        return
    bb = load_backbone(run["model"], args.plain, args.device, run["rank"])
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=GRID)
    suites = {"constructed": approved_order_samples(args.pairs, args.review)}
    if natural:
        suites["natural"] = natural_samples(natural)
    features = {}
    for suite, samples in suites.items():
        features[suite] = []
        for row, views in samples:
            encoded = [encode_view(bb, net, view, args.capacity, suite == "natural") for view in views]
            if sorted(v["answer"] for v in encoded) != [0, 1]:
                raise ValueError("paired views have the same gold option")
            features[suite].append({"id": row["id"], "source_group": row["source_group"],
                                    "split": row["split"], "views": encoded})
            print(f"FEATURES {suite} {row['id']}", flush=True)
    part = out / "features.tmp"
    torch.save(features, part)
    part.replace(out / "features.pt")
    receipt = {"run": run["id"], "sha256": sha256(out / "features.pt"),
               "suites": {k: {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
                          for k, rows in features.items()}}
    write_json(out / "features.json", receipt)


def batch_logits(bb, views, reader, layer, *, swap=False, empty=False, full_vocab=False):
    length = max(len(v["embeds"]) for v in views)
    pair_count = max(len(v["pairs"]) for v in views)
    device = bb.device
    embeds = torch.stack([F.pad(v["embeds"], (0, 0, 0, length - len(v["embeds"])))
                          for v in views]).to(device)
    positions = torch.stack([F.pad(v["positions"], (0, length - len(v["embeds"])))
                             for v in views], dim=1).to(device)
    sizes = torch.tensor([len(v["embeds"]) for v in views], device=device)
    token_mask = torch.arange(length, device=device)[None] < sizes[:, None]
    pairs = torch.stack([F.pad(v["pairs"], (0, 0, 0, 0, 0, pair_count - len(v["pairs"])))
                         for v in views]).to(device)
    if swap:
        pairs = pairs.flip(-2)
    pair_sizes = torch.tensor([len(v["pairs"]) for v in views], device=device)
    valid = torch.arange(pair_count, device=device)[None] < pair_sizes[:, None]
    if empty:
        valid.zero_()
    logits = answer_logits(bb, embeds, positions, token_mask, sizes - 1, reader, pairs, valid,
                           layer=layer)
    if full_vocab:
        return logits
    return logits.gather(-1, torch.tensor([v["choices"] for v in views], device=device))


def verify_reader(bb, reader, layer, views):
    with torch.no_grad():
        base = batch_logits(bb, views, None, layer, full_vocab=True)
        zero = batch_logits(bb, views, reader, layer, full_vocab=True)
        if not torch.equal(base, zero):
            raise RuntimeError("zero-gate next-token logits differ from frozen Plain")
        width = bb.hidden
        sample = torch.arange(6 * width, dtype=torch.float32, device=bb.device).reshape(1, 3, 2, width)
        sample = (sample * 0.01).sin()
        q = sample[:, 0, 0]
        mask = torch.tensor([[True, False, True]], device=bb.device)
        delta, weights = reader.read(q, sample, mask)
        reverse, other_weights = reader.read(q, sample.flip(-2), mask)
        same_weights = torch.equal(weights, other_weights)
        reversal_error = float((delta + reverse).abs().max())
        if reader.reader == "antisymmetric" and (not same_weights or reversal_error > 1e-6):
            raise RuntimeError("fixed-query reversal invariance failed")
        return {"zero_gate_equal": True, "fixed_query_weights_equal": same_weights,
                "fixed_query_reversal_max_abs_error": reversal_error}


def fit(bb, reader, layer, rows, run):
    ordered = [row["views"] for row in rows if row["split"] == "train"]
    if not ordered:
        raise ValueError("training split is empty")
    sampler = np.random.default_rng(run["seed"])
    optimizer = torch.optim.AdamW(reader.parameters(), lr=run["lr"], weight_decay=0.)
    for step in range(run["updates"]):
        indices = sampler.choice(len(ordered), size=min(len(ordered), run["batch"] // 2), replace=False)
        batch = [view for i in indices for view in ordered[int(i)]]
        target = torch.tensor([v["answer"] for v in batch], device=bb.device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(batch_logits(bb, batch, reader, layer), target)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise RuntimeError("nonfinite training")
        if step == 1 and (reader.state.weight.grad is None or not reader.state.weight.grad.count_nonzero()):
            raise RuntimeError("training signal did not reach the state projection at update 2")
        optimizer.step()
        metric = {"update": step + 1, "loss": float(loss.detach()), "adapter_grad_norm": float(grad),
                  "gate": float(reader.gate.detach().tanh()), "examples": len(batch)}
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == run["updates"]:
            print(json.dumps(metric), flush=True)
        yield metric


@torch.no_grad()
def evaluate(bb, reader, layer, rows, batch_size):
    result = []
    heldout = [r for r in rows if r["split"] == "val"]
    for start in range(0, len(heldout), max(1, batch_size // 2)):
        subset = heldout[start:start + max(1, batch_size // 2)]
        views = [v for r in subset for v in r["views"]]
        variants = {"normal": batch_logits(bb, views, reader, layer).cpu()}
        if reader is not None:
            variants["swapped_evidence"] = batch_logits(bb, views, reader, layer, swap=True).cpu()
            empty = batch_logits(bb, views, reader, layer, empty=True, full_vocab=True)
            original = batch_logits(bb, views, None, layer, full_vocab=True)
            if not torch.equal(empty, original):
                raise RuntimeError("empty evidence changed the frozen model")
        for i, row in enumerate(subset):
            outputs = {name: logits[2*i:2*i+2] for name, logits in variants.items()}
            result.append({"id": row["id"], "source_group": row["source_group"],
                           "gold": [v["answer"] for v in row["views"]],
                           "evidence": [v["evidence"] for v in row["views"]],
                           "observation": [v["observation"] for v in row["views"]],
                           "history_sampling": [v["history_sampling"] for v in row["views"]],
                           "visual_tokens": [v["observation"]["visual_tokens_total"] for v in row["views"]],
                           "predictions": {name: x.argmax(-1).tolist() for name, x in outputs.items()},
                           "option_logits": {name: x.tolist() for name, x in outputs.items()}})
    return result


def completed(directory, run, features):
    path = directory / "complete.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    expected = {"predictions.jsonl", "audit.json"}
    if directory.name != "plain":
        expected |= {"metrics.jsonl", "reader.safetensors"}
    if (record["run"] != run["id"] or record["features"] != features["sha256"]
            or set(record["files"]) != expected
            or any(sha256(directory / name) != value for name, value in record["files"].items())):
        raise ValueError(f"experiment arm integrity check failed: {directory}")
    return True


def run_arm(args):
    run = json.loads((args.out / "run.json").read_text())
    receipt = verify_features(args.out, run)
    with (args.out / "features.pt").open("rb") as handle:
        features = torch.load(handle, map_location="cpu", weights_only=True)
    pending = []
    for suite, rows in features.items():
        directory = args.out / suite / args.arm
        if not completed(directory, run, receipt):
            pending.append((suite, rows, directory))
        else:
            print(f"REUSE {suite}/{args.arm}", flush=True)
    if not pending:
        return
    bb = load_backbone(run["model"], Path(run["plain"]), args.device, run["rank"])
    layer = (len(bb.language_model.layers) - 1) // 2
    lora = {n: p for n, p in bb.model.named_parameters() if "lora_" in n}
    before = tensor_digest(lora)
    if not lora:
        raise ValueError("frozen Plain LoRA is missing")
    for suite, rows, directory in pending:
        directory.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(run["seed"])
        reader = None
        audit = {"decoder_layer": layer, "frozen_lora_sha256": before}
        files = ["predictions.jsonl", "audit.json"]
        if args.arm != "plain":
            reader = StateAttention(bb.hidden, run["width"], args.arm).to(bb.device)
            audit.update(verify_reader(bb, reader, layer, rows[0]["views"]))
            initial = tensor_digest(dict(reader.named_parameters()))
            metrics = list(fit(bb, reader, layer, rows, run))
            audit.update(trainable_parameters=sum(p.numel() for p in reader.parameters()),
                         initialization_sha256=initial)
            if tensor_digest(dict(reader.named_parameters())) == initial:
                raise RuntimeError("reader weights did not update")
            write_jsonl(directory / "metrics.jsonl", metrics)
            save_file({n: p.detach().cpu().contiguous() for n, p in reader.state_dict().items()},
                      str(directory / "reader.safetensors"))
            files += ["metrics.jsonl", "reader.safetensors"]
        predictions = evaluate(bb, reader, layer, rows, run["batch"])
        if before != tensor_digest(lora) or any(p.requires_grad or p.grad is not None for p in bb.model.parameters()):
            raise RuntimeError("frozen backbone audit failed")
        write_json(directory / "audit.json", audit)
        write_jsonl(directory / "predictions.jsonl", predictions)
        write_json(directory / "complete.json", {"run": run["id"], "features": receipt["sha256"],
                    "files": {name: sha256(directory / name) for name in files}})
        print(f"DONE {suite}/{args.arm}", flush=True)


def paired_delta(outcome, baseline, groups):
    groups = np.asarray(groups)
    memberships = np.stack([groups == g for g in sorted(set(groups))])
    delta = outcome.astype(int) - baseline.astype(int)
    counts = memberships.sum(axis=1)
    wins = memberships @ delta
    ci = None
    if len(counts) > 1:
        draw = np.random.default_rng(7).integers(len(counts), size=(4000, len(counts)))
        ci = np.quantile(100 * wins[draw].sum(1) / counts[draw].sum(1), [.025, .975]).tolist()
    return {"delta_pp": float(100 * delta.mean()), "ci95_pp": ci}


def summarize(out):
    run = json.loads((out / "run.json").read_text())
    receipt = verify_features(out, run)
    results = {}
    lines = ["# 有向状态注意力对照", "",
             "| 样本 | 模型 | 单题正确率 | 一对都答对 |",
             "|---|---|---:|---:|"]
    for suite in run["suites"]:
        scores, outcomes, expected = {}, {}, None
        init = set()
        for arm in ARMS:
            directory = out / suite / arm
            if not completed(directory, run, receipt):
                raise ValueError(f"incomplete arm: {suite}/{arm}")
            rows = read_jsonl(directory / "predictions.jsonl")
            provenance = [(r["id"], r["source_group"], r["gold"], r["evidence"], r["observation"],
                           r["history_sampling"], r["visual_tokens"]) for r in rows]
            if len({r["id"] for r in rows}) != receipt["suites"][suite]["val"] or len(rows) != len({
                    r["id"] for r in rows}):
                raise ValueError(f"incomplete validation coverage: {suite}/{arm}")
            if expected is not None and provenance != expected:
                raise ValueError(f"mismatched evaluation sample, clock or evidence: {suite}/{arm}")
            expected = provenance
            labels = np.asarray([r["gold"] for r in rows])
            pred = np.asarray([r["predictions"]["normal"] for r in rows])
            correct = pred == labels
            outcomes[arm] = correct.all(axis=1)
            scores[arm] = {"question_accuracy": float(correct.mean()),
                           "pair_accuracy": float(outcomes[arm].mean())}
            if arm != "plain":
                audit = json.loads((directory / "audit.json").read_text())
                init.add((audit["initialization_sha256"], audit["trainable_parameters"]))
                reversed_pred = np.asarray([r["predictions"]["swapped_evidence"] for r in rows])
                scores[arm]["swap_prediction_flip_rate"] = float((pred != reversed_pred).mean())
            lines.append(f"| {suite} | {arm} | {correct.mean():.2%} | {outcomes[arm].mean():.2%} |")
        if len(init) != 1:
            raise ValueError(f"reader parameter counts or initialization differ: {suite}")
        groups = [row[1] for row in expected]
        comparisons = {arm: paired_delta(outcomes["antisymmetric"], outcomes[arm], groups)
                       for arm in ARMS[:-1]}
        results[suite] = {"scores": scores, "paired_antisymmetric_minus": comparisons,
                          "validation_pairs": len(groups), "source_groups": len(set(groups)),
                          "reader_trainable_parameters": next(iter(init))[1]}
    lines += ["", "| 样本 | Antisymmetric 相对 | 配对差值（百分点） | 95% 区间 |",
              "|---|---|---:|---:|"]
    for suite, result in results.items():
        for baseline, contrast in result["paired_antisymmetric_minus"].items():
            interval = contrast["ci95_pp"]
            bounds = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "—"
            lines.append(f"| {suite} | {baseline} | {contrast['delta_pp']:+.2f} | {bounds} |")
    lines += ["", "constructed 使用人工重排和全帧证据，作为工程验证。",
              "natural 使用原始播放顺序和提供的轨迹。配对区间按来源组 bootstrap。",
              "所有结果使用固定选项的一次前向判分。"]
    natural = results.get("natural")
    if natural:
        intervals = natural["paired_antisymmetric_minus"].values()
        supported = all(item["ci95_pp"] and item["ci95_pp"][0] > 0 for item in intervals)
        verdict = "gain_detected_on_supplied_natural_set" if supported else "gain_not_established"
    else:
        verdict = "engineering_check_only"
    lines += ["", f"Decision: {verdict}"]
    write_json(out / "report.json", {"completed": True, "run": run["id"], "results": results,
                                    "natural_evaluated": bool(natural), "decision": verdict})
    atomic_text(out / "REPORT.md", "\n".join(lines) + "\n")
    print("VERIFIED STATE ATTENTION:", out / "report.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    arm = sub.add_parser("arm")
    report = sub.add_parser("report")
    for cmd in (prep, arm, report):
        cmd.add_argument("--out", required=True, type=Path)
    for cmd in (prep, arm):
        cmd.add_argument("--device", default="cuda:0")
    prep.add_argument("--pairs", type=Path, required=True)
    prep.add_argument("--review", type=Path, required=True)
    prep.add_argument("--natural", type=Path)
    prep.add_argument("--plain", type=Path, required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--updates", type=int, default=100)
    prep.add_argument("--seed", type=int, default=17)
    prep.add_argument("--width", type=int, default=256)
    prep.add_argument("--capacity", type=int, default=8)
    prep.add_argument("--batch", type=int, default=4)
    prep.add_argument("--lr", type=float, default=1e-4)
    arm.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        if min(args.updates, args.width, args.batch) < 2 or args.capacity < 1 or not (
                math.isfinite(args.lr) and args.lr > 0):
            parser.error("invalid training or storage budget")
        prepare(args)
    elif args.stage == "arm":
        run_arm(args)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
