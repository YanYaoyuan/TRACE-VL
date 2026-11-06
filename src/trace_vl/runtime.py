"""Small explicit runtime shared by training and prefix-based inference."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data.dataset import VideoDataset
from .models.model import TraceVL


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name="auto"):
    if name == "auto":
        return torch.device(
            f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible in this process; check driver/device access")
    return device


def autocast_for(device, precision):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    return nullcontext()


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_dataset(config, split):
    d = config["data"]
    return VideoDataset(
        d["manifest"],
        split=split,
        image_size=config["model"]["image_size"],
        sampling_fps=d.get("sampling_fps", 2),
        max_frames=d.get("max_frames", 8),
        media_root=d.get("media_root"),
    )


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def environment():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    return {
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "git_revision": revision,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def load_checkpoint(path, device="cpu"):
    # Checkpoints written here consist only of tensors and primitive containers.
    return torch.load(path, map_location=device, weights_only=True)


def initialize_weights(model, checkpoint):
    """Warm start compatible weights; resource/context changes do not restore Adam or steps."""
    saved = load_checkpoint(checkpoint)
    previous = ModelConfig.from_dict(saved["config"]["model"]).to_dict()
    current = model.cfg.to_dict()
    changed = {key for key in current if current[key] != previous[key]}
    allowed = {
        "spatial_tokens",
        "cache_frames",
        "cache_seconds",
        "read_events",
        "text_limit",
        "local_files_only",
        "normalize_memory",
        # Parameter-free v2 fixes; query_projection adds a module initialized from the tied map.
        "prompt_format",
        "state_update",
        "state_kv_norm",
        "retrieval_gradient",
        "query_projection",
        "token_selection",
        "motion_layout",
    }
    if changed - allowed:
        raise ValueError(f"initialization architecture/backbone mismatch: {sorted(changed - allowed)}")
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    added = {"query_projection.weight", "query_projection.bias"}
    if unexpected or set(missing) - added:
        raise ValueError(f"initialization state mismatch: missing={missing}, unexpected={unexpected}")
    if missing:
        model.reset_query_projection()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "source_step": saved["step"],
        "changed_context_settings": sorted(changed),
        "optimizer_restored": False,
    }


def restore_model(config, checkpoint, device):
    saved = load_checkpoint(checkpoint)
    if ModelConfig.from_dict(saved["config"]["model"]).to_dict() != config["model"]:
        raise ValueError("checkpoint architecture differs from evaluation configuration")
    model = TraceVL(ModelConfig.from_dict(config["model"]))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()
