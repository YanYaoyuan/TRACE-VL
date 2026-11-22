"""Train and compare the native dense, selective and corrected vision paths."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel

from trace_vl.core_vlm.control import checkpoint_identity, select_records
from trace_vl.core_vlm.data import decode_observation, read_split, sha256
from trace_vl.core_vlm.experiment import paired
from trace_vl.core_vlm.native import NativeVideoQALoss, prepare_native
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VisualStep
from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .data import read_rows, streaming_population, train_population, validate_population, write_rows
from .encoder import ContextCorrection, SpatialRefresh
from .media import DECODE_POLICY, VideoProbeCache

ORIGINAL_DECODE_POLICY = "per-question-pts-probe-v1"


def frozen_inputs(args, root):
    manifests = {"train": Path(args.train_manifest), "ovo": Path(args.ovo_manifest)}
    hashes = {name: sha256(path) for name, path in manifests.items()}
    if args.streaming_root:
        candidates = sorted(Path(args.streaming_root).rglob("Real_Time_Visual_Understanding.csv"))
        if len(candidates) != 1:
            raise ValueError(f"StreamingBench official real-time CSV unavailable or ambiguous: {candidates}")
        hashes["streamingbench"] = sha256(candidates[0])
    code = Path(__file__).resolve().parents[1]
    files = [*Path(__file__).resolve().parent.glob("*.py"),
             *(code / "core_vlm" / n for n in ("data.py", "native.py", "qa.py", "experiment.py")),
             code / "v3" / "backbone.py"]
    identity = {"config": {k: v for k, v in vars(args).items()
                           if k not in {"command", "shard", "out", "workers"}},
                "manifests": hashes, "model": checkpoint_identity(args.model),
                "transformers": transformers.__version__, "torch": str(torch.__version__),
                "code": {str(p.relative_to(code)): sha256(p) for p in files}}
    identity["id"] = digest(identity)
    ensure_run_manifest(root / "inputs.json", identity)
    return identity


def prepare(args, root):
    identity = frozen_inputs(args, root)
    stamp = root / "prepared.json"
    if stamp.is_file():
        return load_prepared(root)
    ovo = read_split(Path(args.ovo_manifest), "val")
    if args.ovo_limit:
        ovo = select_records(ovo, args.ovo_limit)
    external = {"ovo": ovo}
    audit = {}
    if args.streaming_root:
        external["streamingbench"], audit = streaming_population(
            args.streaming_root, root / "streaming-media", args.streaming_limit,
        )
    for rows in external.values():
        validate_population(rows)
    # Exclude known IDs from the full OVO development manifest even for a smoke subset.
    train = train_population(args.train_manifest, {**external, "ovo": read_split(Path(args.ovo_manifest), "val")},
                             args.train_clips, args.seed)
    populations = {"train": train, **external}
    paths = {}
    for name, rows in populations.items():
        path = root / f"{name}.jsonl"
        write_rows(path, rows)
        paths[name] = sha256(path)
    record = {"input_id": identity["id"], "manifests": paths,
              "counts": {name: len(rows) for name, rows in populations.items()}, "streamingbench": audit}
    write_json(stamp, record)
    print("PREPARED", json.dumps(record), flush=True)
    return record


def load_prepared(root):
    record = json.loads((root / "prepared.json").read_text())
    identity = json.loads((root / "inputs.json").read_text())
    if record["input_id"] != identity["id"]:
        raise ValueError("prepared population belongs to a different run")
    for name, expected in record["manifests"].items():
        if sha256(root / f"{name}.jsonl") != expected:
            raise ValueError(f"prepared population changed: {name}")
    return record


def decoded(args, rows, media_cache=None):
    """Bound the decode queue and overlap FFmpeg with GPU work."""
    def work(row):
        start = time.perf_counter()
        decoder = decode_observation if media_cache is None else media_cache.decode
        images, times, record, audit = decoder(row, steps=args.lookback, pixels=args.pixels, recent_hz=args.hz)
        return images, times, record, {**audit, "cpu_decode_ms": 1000 * (time.perf_counter() - start),
                                       "decode_policy": DECODE_POLICY if media_cache else ORIGINAL_DECODE_POLICY}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        queue = {}
        for i, row in enumerate(rows):
            upcoming = min(i + args.workers, len(rows))
            for j in range(i, upcoming):
                if j not in queue:
                    queue[j] = pool.submit(work, rows[j])
            try:
                yield queue.pop(i).result()
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                raise ValueError(f"media decode failed for {row['id']}: {row['video']}") from error


def encoder(args, bb, fraction):
    return SpatialRefresh(bb.visual, args.shallow, fraction, args.full_every, args.cut_score)


def backbone(args, device):
    return Backbone(BackboneConfig(name=args.model, dtype=args.dtype, local_files_only=True),
                    StreamConfig(pixels_per_frame=args.pixels), device=device)


def corrector(args, bb):
    return ContextCorrection(bb.visual.config.hidden_size, bb.hidden, args.rank).to(bb.device)


def patch_inputs(bb, images):
    packed = bb.processor.video_processor(
        videos=[images.reshape(-1, *images.shape[2:])], do_sample_frames=False,
        do_resize=False, return_tensors="pt",
    )
    t, h, w = packed["video_grid_thw"][0].tolist()
    if len(images) != t:
        raise ValueError("processor changed the temporal grid")
    pixels = packed["pixel_values_videos"].to(device=bb.device, dtype=bb.dtype).split(h * w)
    grid = torch.tensor([[1, h, w]], device=bb.device)
    return pixels, grid, h, w


def dense(bb, pixels, grid):
    return bb.visual(pixels, grid_thw=grid).pooler_output


@torch.no_grad()
def training_sample(args, bb, images, fraction):
    pixels, grid, h, w = patch_inputs(bb, images)
    state = encoder(args, bb, fraction)
    parts = []
    for step in pixels:
        result = state.update(step, h, w)
        if result.reason != "selective":
            continue
        stale = ~result.refreshed
        teacher = dense(bb, step, grid)[stale].float()
        old = result.features[stale].float()
        scene, old_scenes, tiles = state.correction_inputs(stale)
        parts.append((scene.expand_as(old_scenes), old_scenes, tiles, teacher - old,
                      teacher.square().mean(-1).clamp_min(1e-6)))
    return parts


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args, root, device, rank, world):
    prepared = load_prepared(root)
    done = root / "training-complete.json"
    weights = root / "correction.pt"
    if done.is_file():
        receipt = json.loads(done.read_text())
        if receipt != {"input_id": prepared["input_id"], "weights_sha256": sha256(weights)}:
            raise ValueError("trained weights or training inputs changed")
        print("REUSED verified correction checkpoint", flush=True)
        return
    rows = read_rows(root / "train.jsonl")
    if len(rows) % world:
        raise ValueError("number of clips must divide evenly across ranks")
    torch.manual_seed(args.seed)
    bb = backbone(args, device)
    correction = corrector(args, bb)
    model = (DistributedDataParallel(correction, device_ids=[device.index]) if world > 1 else correction)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    checkpoint = root / "training-state.pt"
    metrics_file = root / "training.jsonl"
    updates = len(rows) // world * args.epochs
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["input_id"] != prepared["input_id"]:
            raise ValueError("training state belongs to a different run")
        correction.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["update"]
    metrics = [m for m in read_rows(metrics_file) if m["update"] <= start] if metrics_file.exists() else []
    if rank == 0 and len(metrics) != start:
        raise ValueError("training log does not cover the saved checkpoint")
    if rank == 0:
        write_rows(metrics_file, metrics)
    # Round-robin across the fixed seeded population. No training or eval label
    # enters the encoder. Alternate refresh budgets during distillation.
    local = rows[rank::world] * args.epochs
    for i, (images, _times, _row, _audit) in enumerate(decoded(args, local[start:]), start + 1):
        fraction = args.fractions[(i - 1) % len(args.fractions)]
        samples = training_sample(args, bb, images, fraction)
        if samples:
            scene, old_scene, tiles, target, norm = (torch.cat(x) for x in zip(*samples, strict=True))
        else:
            # Participate in the same DDP collectives even for a cut-only clip.
            scene = old_scene = tiles = torch.zeros((1, bb.visual.config.hidden_size), device=device)
            target = torch.zeros((1, bb.hidden), device=device)
            norm = torch.ones(1, device=device)
        optimizer.zero_grad(set_to_none=True)
        update = model(scene, old_scene, tiles)
        error = (update - target).square().mean(-1) / norm
        baseline = target.square().mean(-1) / norm
        token_count = torch.tensor(len(target) if samples else 0, device=device, dtype=torch.float32)
        stats = torch.stack([error.detach().sum(), baseline.sum(), token_count])
        if world > 1:
            dist.all_reduce(stats)
        loss = error.sum() * world / stats[2].clamp_min(1)
        loss.backward()
        if not torch.isfinite(stats).all():
            raise ValueError("nonfinite visual distillation loss")
        if stats[2] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if rank == 0:
            mse, base, tokens = stats.tolist()
            metric = {"update": i, "normalized_mse": mse / max(tokens, 1),
                      "reuse_mse": base / max(tokens, 1), "stale_tiles": int(tokens),
                      "refresh_fraction": fraction}
            with metrics_file.open("a") as handle:
                handle.write(json.dumps(metric) + "\n")
            if i % 10 == 0 or i == 1 or i == updates:
                print(f"TRAIN {i}/{updates} {json.dumps(metric)}", flush=True)
            if i % args.save_every == 0 or i == updates:
                atomic_checkpoint(checkpoint, {"input_id": prepared["input_id"], "update": i,
                                               "model": correction.state_dict(), "optimizer": optimizer.state_dict()})
    if rank == 0:
        rows = read_rows(metrics_file)
        if sum(m["stale_tiles"] for m in rows) == 0 or torch.count_nonzero(correction.output.weight) == 0:
            raise ValueError("no learned correction; inspect refresh diagnostics")
        atomic_checkpoint(weights, correction.state_dict())
        write_json(done, {"input_id": prepared["input_id"], "weights_sha256": sha256(weights)})
    if world > 1:
        dist.barrier()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def visual_check(args, bb, pixels, grid, h, w):
    state = encoder(args, bb, 1.)
    # Check more than the initial refresh. The native full path is the reference.
    for step in pixels[:2]:
        reference = dense(bb, step, grid)
        candidate = state.update(step, h, w).features
        torch.testing.assert_close(reference, candidate, atol=0, rtol=0)


def arm_name(fraction, calibrated):
    return f"{'corrected' if calibrated else 'reuse'}_{fraction:g}"


@torch.no_grad()
def predict(args, bb, net, correction, observation, rotate, *, check=False):
    images, times, row, audit = observation
    start = time.perf_counter()
    pixels, grid, h, w = patch_inputs(bb, images)
    visible = min(args.context, len(pixels))
    audit_prompt = {**audit, "frame_indices": audit["frame_indices"][-2 * visible:],
                    "decoded_rgb_sha256": hashlib.sha256(images[-visible:].tobytes()).hexdigest(),
                    "frame_times": times[-visible:].tolist()}
    layout = prepare_native(bb.processor, images[-visible:], times[-visible:], row, audit_prompt,
                            allow_native_timestamp_approximation=True)
    sync(bb.device)
    setup_ms = (time.perf_counter() - start) * 1000
    row = {**row, "native_layout": layout}
    if check:
        visual_check(args, bb, pixels, grid, h, w)
        native_pixels = layout.inputs["pixel_values_videos"].to(bb.device, bb.dtype)
        torch.testing.assert_close(torch.cat(pixels[-visible:]), native_pixels, atol=0, rtol=0)
    mm_grid = (h // bb.visual.spatial_merge_size, w // bb.visual.spatial_merge_size)
    arms = [("dense", None, False)] + [
        (arm_name(fraction, calibrated), fraction, calibrated)
        for fraction in args.fractions for calibrated in (False, True)
    ]
    shift = rotate % len(arms)
    results, outputs = {}, {}
    for name, fraction, calibrated in arms[shift:] + arms[:shift]:
        state = encoder(args, bb, fraction) if fraction is not None else None
        tokens, durations, reasons, refreshed = [], [], [], []
        if bb.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(bb.device)
        sync(bb.device)
        for parts in pixels:
            start = time.perf_counter()
            if state is None:
                feature = dense(bb, parts, grid)
                refresh_count, reason = feature.shape[0], "dense"
            else:
                update = state.update(parts, h, w)
                feature = state.corrected(update, correction) if calibrated else update.features
                refresh_count, reason = int(update.refreshed.sum()), update.reason
            sync(bb.device)
            durations.append(1000 * (time.perf_counter() - start))
            tokens.append(feature)
            refreshed.append(refresh_count)
            reasons.append(reason)
        steps = [VisualStep(feature, mm_grid, float(pair[-1]), tuple(pair))
                 for feature, pair in zip(tokens[-visible:], layout.frame_times, strict=True)]
        started = time.perf_counter()
        _, choice = net.generate(steps, row, max_new_tokens=1, return_choice=True)
        sync(bb.device)
        answer_ms = (time.perf_counter() - started) * 1000
        results[name] = {"answer": chr(65 + choice), "choice": choice, "vision_ms": durations,
                         "answer_ms": answer_ms, "recent_loop_and_answer_ms": sum(durations) + answer_ms,
                         "replay_service_ms": audit["cpu_decode_ms"] + setup_ms + sum(durations) + answer_ms,
                         "refreshed_tiles": refreshed, "refresh_reasons": reasons, "tiles": len(tokens[0]),
                         "gpu_peak_allocated_mib": (torch.cuda.max_memory_allocated(bb.device) / 2**20
                                                    if bb.device.type == "cuda" else None)}
        outputs[name] = torch.stack(tokens).float()
    reference = outputs["dense"]
    normalizer = reference.square().mean(-1).clamp_min(1e-6)
    for name in results:
        results[name]["normalized_feature_mse"] = float(
            ((outputs[name] - reference).square().mean(-1) / normalizer).mean()
        )
    for fraction in args.fractions:
        a, b = (results[arm_name(fraction, mode)] for mode in (False, True))
        if (a["refresh_reasons"], a["refreshed_tiles"]) != (b["refresh_reasons"], b["refreshed_tiles"]):
            raise ValueError("refresh budgets differ between reuse and correction")
    return {"id": row["id"], "video": row["video"], "observation": layout.audit,
            "all_frame_times": times.tolist(), "rgb_sha256": audit["decoded_rgb_sha256"],
            "decode_ms": audit["cpu_decode_ms"], "processor_ms": setup_ms, "arms": results,
            "decode_policy": audit.get("decode_policy", ORIGINAL_DECODE_POLICY)}


def evaluate(args, root, device):
    prepared = load_prepared(root)
    path = root / "correction.pt"
    training = json.loads((root / "training-complete.json").read_text())
    if training != {"input_id": prepared["input_id"], "weights_sha256": sha256(path)}:
        raise ValueError("correction weights are incomplete or altered")
    directory = root / "predictions"
    directory.mkdir(exist_ok=True)
    all_rows = [row for name in prepared["manifests"] if name != "train" for row in read_rows(root / f"{name}.jsonl")]
    expected = [(row, directory / f"{digest(row['id'])}.json")
                for row in all_rows[args.shard::args.gpus]]
    pending = [(row, dest) for row, dest in expected if not dest.is_file()]
    if not pending:
        return
    bb = backbone(args, device)
    correction = corrector(args, bb)
    correction.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net = NativeVideoQALoss(bb, mode="full", answer_end_token_id=bb.tokenizer.eos_token_id).eval()
    cache = VideoProbeCache(
        Path(os.environ.get("TRACE_EVAL_SCRATCH", root / "tmp" / "media-cache")),
        fast_timeout=int(os.environ.get("TRACE_MEDIA_TIMEOUT", 180)),
        retry_timeout=int(os.environ.get("TRACE_MEDIA_RETRY_TIMEOUT", 900)),
    )
    observations = decoded(args, [row for row, _ in pending], cache)
    for i, ((row, dest), observation) in enumerate(zip(pending, observations, strict=True)):
        if i == 0:
            # Warm each process and validate both full encoder paths before timing.
            predict(args, bb, net, correction, observation, 0, check=True)
        prediction = predict(args, bb, net, correction, observation, int(digest(row["id"])[:8], 16))
        write_json(dest, {**prediction, "run": prepared["input_id"], "correction": training["weights_sha256"]})
        if i % 20 == 0 or i + 1 == len(pending):
            print(f"EVAL shard={args.shard} {i + 1}/{len(pending)}", flush=True)


def latency(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "p50": float(np.quantile(values, .50)), "p95": float(np.quantile(values, .95))}


def report(args, root):
    prepared = load_prepared(root)
    training = json.loads((root / "training-complete.json").read_text())
    results = {}
    for name in prepared["manifests"]:
        if name == "train":
            continue
        rows = read_rows(root / f"{name}.jsonl")
        saved = {r["id"]: json.loads((root / "predictions" / f"{digest(r['id'])}.json").read_text()) for r in rows}
        groups = {"all": rows}
        video_hashes, decode_groups = {}, {}
        for row in rows:
            pred = saved[row["id"]]
            if (pred["id"] != row["id"] or pred["run"] != prepared["input_id"]
                    or pred["correction"] != training["weights_sha256"]
                    or any(t > row["query_time"] for pair in pred["all_frame_times"] for t in pair)):
                raise ValueError("prediction identity/causality check failed")
            checksum = pred["observation"]["video_sha256"]
            if video_hashes.setdefault(row["video"], checksum) != checksum:
                raise ValueError(f"video bytes differ between predictions: {row['video']}")
            policy = pred.get("decode_policy", ORIGINAL_DECODE_POLICY)
            decode_groups.setdefault(policy, []).append(row["id"])
            groups.setdefault(row["task"], []).append(row)
        if name == "ovo":
            groups["realtime"] = [r for r in rows if r["task"] in {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}]
            groups["backward"] = [r for r in rows if r["task"] in {"EPM", "ASI", "HLD"}]
        names = saved[rows[0]["id"]]["arms"]
        arms = {arm: {qid: pred["arms"][arm] for qid, pred in saved.items()} for arm in names}
        scores = {}
        for arm, predictions in arms.items():
            vals = list(predictions.values())
            steady = [t for p in vals for t in p["vision_ms"][1:]]
            scores[arm] = {
                "accuracy": {group: {"n": len(subset), "score": sum(
                    predictions[r["id"]]["choice"] == r["answer"] for r in subset) / len(subset)}
                             for group, subset in groups.items() if subset},
                "visual_first_step_ms": latency([p["vision_ms"][0] for p in vals]),
                "visual_after_first_step_ms": latency(steady),
                "visual_all_steps_ms": latency([t for p in vals for t in p["vision_ms"]]),
                "lm_choice_ms": latency([p["answer_ms"] for p in vals]),
                "visual_loop_plus_lm_ms": latency([p["recent_loop_and_answer_ms"] for p in vals]),
                "replay_service_ms": latency([p["replay_service_ms"] for p in vals]),
                "replay_service_ms_by_decode_policy": {
                    policy: latency([predictions[qid]["replay_service_ms"] for qid in ids])
                    for policy, ids in decode_groups.items()
                },
                "refresh_events": dict(Counter(reason for p in vals for reason in p["refresh_reasons"])),
                "encoder_sampled_frames_per_second": (2000 * len(steady) / sum(steady) if steady else None),
                "refreshed_fraction": sum(sum(p["refreshed_tiles"]) for p in vals) /
                                      sum(len(p["vision_ms"]) * p["tiles"] for p in vals),
                "normalized_feature_mse": float(np.mean([p["normalized_feature_mse"] for p in vals])),
                "gpu_peak_allocated_mib": max((p["gpu_peak_allocated_mib"] for p in vals
                                                if p["gpu_peak_allocated_mib"] is not None), default=None),
            }
        comparisons = {}
        for fraction in args.fractions:
            reuse, correct = (arm_name(fraction, mode) for mode in (False, True))
            for left, right in ((correct, reuse), (correct, "dense"), (reuse, "dense")):
                comparisons[f"{left}_minus_{right}"] = {group: paired(subset, arms[left], arms[right])
                                                       for group, subset in groups.items() if subset}
        results[name] = {"scores": scores, "paired": comparisons,
                         "decode_policy_question_counts": {key: len(ids) for key, ids in decode_groups.items()}}
    config = json.loads((root / "inputs.json").read_text())["config"]
    output = {"completed": True, "device": args.device, "config": config, "data": prepared, "results": results,
              "scope": "One seed. Matched causal fixed-rate recent clips, reset per question; native Qwen "
                       "choice score. Encoder rates exclude decoding/preprocessing and are not deployment FPS. "
                       "Replay service time adds shared decode/preprocess cost to per-arm vision and one-token LM."}
    repair = root / "eval-recovery.json"
    if repair.is_file():
        output["evaluation_recovery"] = json.loads(repair.read_text())
    write_json(root / "report.json", output)
    lines = ["# TRACE-VL 流式视觉编码验证", "", output["scope"], ""]
    for name, result in results.items():
        lines += [f"## {name}", "", "| 模型 | 选择准确率 | 首步视觉 P95 ms | 后续视觉 P95 ms | "
                  "含解码的回放 P95 ms | 实际刷新比例 | 特征误差 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, s in result["scores"].items():
            lines.append(f"| {arm} | {s['accuracy']['all']['score']:.2%} | "
                         f"{s['visual_first_step_ms']['p95']:.2f} | "
                         f"{s['visual_after_first_step_ms'].get('p95', math.nan):.2f} | "
                         f"{s['replay_service_ms']['p95']:.2f} | {s['refreshed_fraction']:.2%} | "
                         f"{s['normalized_feature_mse']:.5f} |")
        lines.append("")
        policies = result["decode_policy_question_counts"]
        if len(policies) > 1:
            lines += [f"回放时延涵盖两种解码流程 {policies}。分流程统计见 report.json。", ""]
        for arm, values in result["paired"].items():
            value = values["all"]
            lines.append(f"- {arm} {value['delta_pp']:+.2f} pp，95% CI {value['ci95_pp']}")
        lines.append("")
    atomic_text(root / "REPORT.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "train", "eval", "report"))
    for key in ("out", "model", "train-manifest", "ovo-manifest"):
        p.add_argument(f"--{key}", required=True)
    p.add_argument("--streaming-root")
    p.add_argument("--fractions", nargs="+", type=float, default=[.25, .50])
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--train-clips", type=int, default=512)
    p.add_argument("--ovo-limit", type=int, default=0)
    p.add_argument("--streaming-limit", type=int, default=250)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--shallow", type=int, default=4)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--lookback", type=int, default=16)
    p.add_argument("--hz", type=float, default=2.)
    p.add_argument("--pixels", type=int, default=65536)
    p.add_argument("--full-every", type=int, default=12)
    p.add_argument("--cut-score", type=float, default=.12)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    if (min(args.gpus, args.workers, args.train_clips, args.epochs, args.context, args.rank, args.save_every) < 1
            or args.lookback < args.context or args.train_clips % args.gpus or not 0 <= args.shard < args.gpus
            or len(set(args.fractions)) != len(args.fractions)
            or any(not 0 < f < 1 for f in args.fractions)):
        p.error("invalid refresh fractions, population, shard count or window")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(args, root)
        return
    if args.command == "report":
        report(args, root)
        return
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    device = torch.device(args.device, int(os.environ.get("LOCAL_RANK", "0"))) if args.device == "cuda" else (
        torch.device("cpu")
    )
    if args.command == "train" and world != args.gpus:
        p.error("training workers must match --gpus")
    with contextlib.ExitStack() as cleanup:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if world > 1:
            dist.init_process_group("nccl")
            cleanup.callback(dist.destroy_process_group)
        if args.command == "train":
            train(args, root, device, rank, world)
        else:
            evaluate(args, root, device)


if __name__ == "__main__":
    main()
