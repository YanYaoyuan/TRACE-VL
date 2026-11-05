"""Replay sampled visible prefixes chronologically; no target annotations enter model calls."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .config import ModelConfig
from .data.schema import write_jsonl
from .evaluation.metrics import score_predictions
from .models.model import TraceVL
from .runtime import (
    autocast_for,
    device_for,
    environment,
    file_hash,
    make_dataset,
    restore_model,
    seed_everything,
    write_json,
)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(config, checkpoint, output, split="test", allow_random=False):
    settings = config.get("evaluation", {})
    seed_everything(config.get("seed", 7), settings.get("cpu_threads", 2))
    device = device_for(settings.get("device", "auto"))
    if checkpoint:
        model = restore_model(config, checkpoint, device)
    elif allow_random:
        model = TraceVL(ModelConfig.from_dict(config["model"])).to(device).eval()
    else:
        raise ValueError("evaluation requires a checkpoint; --allow-random is only for software checks")
    dataset = make_dataset(config, split)
    threshold, cooldown = settings.get("alert_threshold", 0.5), settings.get("cooldown_seconds", 1.0)
    if not 0 < threshold < 1 or cooldown < 0:
        raise ValueError("invalid fixed alert threshold or cooldown")
    predictions = []
    cached = None
    prefix_count = 0
    for sample in dataset:
        r = sample["record"]
        frames, times = sample["frames"].to(device), sample["timestamps"].to(device)
        key = (
            r["video"],
            r["observation"]["start"],
            sample["cutoff"],
            sample["rule"],
            r.get("monitor_category"),
        )
        reused = bool(settings.get("reuse_video_state", False) and cached is not None and key == cached[0])
        with autocast_for(device, settings.get("precision", "fp32")):
            if reused:
                _, state, alerts, frame_scores = cached
                observation_seconds = 0.0
            else:
                state = model.initial_state(1, device)
                alerts, frame_scores = [], []
                last_alert, was_active = -float("inf"), False
                rule = model.encode_text([sample["rule"]])
                sync(device)
                begin = time.perf_counter()
                for frame, timestamp in zip(frames, times):
                    state, aux = model.observe_step(frame[None], timestamp[None], state, rule)
                    score = float(aux["alert_logits"].sigmoid()[0])
                    t = float(timestamp)
                    frame_scores.append({"time": t, "score": score})
                    active = score >= threshold
                    if sample["rule"] and active and not was_active and t - last_alert >= cooldown:
                        alerts.append({"time": t, "category": r["monitor_category"], "score": score})
                        last_alert = t
                    was_active = active
                sync(device)
                observation_seconds = time.perf_counter() - begin
                prefix_count += 1
                if settings.get("reuse_video_state", False):
                    cached = (key, state, alerts, frame_scores)
            sync(device)
            begin = time.perf_counter()
            answers, evidence = model.answer(state, [sample["question"]], settings.get("max_new_tokens", 16))
            sync(device)
            query_seconds = time.perf_counter() - begin
        predictions.append(
            {
                "id": r["id"],
                "answer": answers[0],
                "alerts": alerts,
                "observed_until": sample["cutoff"],
                "frame_scores": frame_scores,
                "sampled_frames": len(times),
                "observation_compute_seconds": observation_seconds,
                "observation_reused": reused,
                "query_latency_seconds": query_seconds,
                "evidence": {k: v[0].cpu().tolist() for k, v in evidence.items()},
            }
        )
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "predictions.jsonl", predictions)
    report = score_predictions(dataset.records, predictions)
    report["provenance"] = {
        "manifest_sha256": file_hash(config["data"]["manifest"]),
        "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
        "random_weights": checkpoint is None,
        "split": split,
        "environment": environment(),
        "evaluation_config": config,
        "reuse_video_state": settings.get("reuse_video_state", False),
        "observed_prefixes": prefix_count,
        "timing_note": (
            "Sampled prefix replay. Excludes media decoding and rule embedding; not live-camera end-to-end latency. "
            "Reused prefixes have zero observation compute time; query latency is always measured separately."
        ),
    }
    write_json(directory / "scores.json", report)
    print(directory / "scores.json")
    return report
