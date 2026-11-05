"""Synthetic stream resource profiling, separate from semantic benchmark scores."""

from __future__ import annotations

import time
from dataclasses import fields, is_dataclass

import torch

from .config import ModelConfig
from .inference import sync
from .models.model import TraceVL
from .runtime import autocast_for, device_for, environment, seed_everything, write_json


def logical_bytes(value):
    """Tensor element bytes; does not estimate allocator overhead or deduplicate aliases."""
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if is_dataclass(value):
        return sum(logical_bytes(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, (tuple, list)):
        return sum(logical_bytes(v) for v in value)
    return 0


@torch.inference_mode()
def profile_stream(config, output, lengths=(16, 64, 256)):
    seed_everything(config.get("seed", 7))
    cfg = ModelConfig.from_dict(config["model"])
    settings = config.get("evaluation", {})
    device = device_for(settings.get("device", "auto"))
    model = TraceVL(cfg).to(device).eval()
    frame = torch.rand(1, 3, cfg.image_size, cfg.image_size, device=device)
    rows = []
    with autocast_for(device, settings.get("precision", "fp32")):
        rule = model.encode_text(["Monitor changes in the scene."])
        warm = model.initial_state(1, device)
        for step in range(4):
            warm, _ = model.observe_step(frame, torch.tensor([step / 2], device=device), warm, rule)
        del warm
        for length in lengths:
            if length <= 0:
                raise ValueError("profile lengths must be positive")
            state = model.initial_state(1, device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            begin = time.perf_counter()
            for step in range(length):
                state, _ = model.observe_step(frame, torch.tensor([step / 2], device=device), state, rule)
            sync(device)
            elapsed = time.perf_counter() - begin
            rows.append(
                {
                    "frames": length,
                    "seconds": elapsed,
                    "milliseconds_per_frame": elapsed * 1000 / length,
                    "logical_state_bytes": logical_bytes(state),
                    "retained_events": int(state.memory.valid.sum()),
                    "cached_frames_per_layer": [len(c) for c in state.caches],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                }
            )
            del state
    report = {
        "environment": environment(),
        "device": str(device),
        "config": config,
        "rows": rows,
        "parameters": model.parameter_report(),
        "note": "Random-weight synthetic input; observation compute only, not QA quality, camera latency or trained 2B throughput.",
    }
    write_json(output, report)
    return report
