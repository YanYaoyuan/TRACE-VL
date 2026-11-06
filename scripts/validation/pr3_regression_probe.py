"""Real-Qwen3.5 gradient probe and CPU evaluation-resume regression for PR #3.

Run from a checkout of the PR with its vlm dependencies installed.
The original 143e4c6 is expected to fail both probes; no model weights are included.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from trace_vl.v3.config import load_v3_config

REVIEWED_COMMIT = "143e4c6cb7a5bb9b88bbcfce7a69708cbccaf9a2"


def fixture(root, seconds=3):
    root.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(root / "static.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 2, (448, 448))
    if not writer.isOpened():
        raise RuntimeError("MJPG encoder unavailable")
    for _ in range(math.ceil(seconds * 2) + 1):
        writer.write(np.full((448, 448, 3), 90, np.uint8))
    writer.release()
    record = {
        "id": "static-gray", "source": "software-regression", "task": "presence",
        "video": "static.avi", "query_time": float(seconds), "question": "Is a red box visible?",
        "options": ["Yes", "No"], "answer": 1, "split": "train",
    }
    records = root / "records.jsonl"
    records.write_text(json.dumps(record) + "\n")
    return record, records


def gradient(args):
    from trace_vl.v3.backbone import Backbone
    from trace_vl.v3.model import TraceV3Modules
    from trace_vl.v3.train import StudentLoss, load_clip, teacher_targets

    cfg, _ = load_v3_config(args.config)
    if args.model:
        cfg.backbone.name = args.model
        cfg.backbone.local_files_only = Path(args.model).exists()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.cuda.set_device(args.device)
    bb = Backbone(cfg.backbone, cfg.stream, args.device)
    bb.add_lora(cfg.train.lora_rank, cfg.train.lora_alpha)
    bb.enable_checkpointing()
    record, _ = fixture(args.output / "fixture", args.seconds)
    clips, times = load_clip(record, str(args.output / "fixture"), cfg, args.seconds)
    teacher, captions = teacher_targets(bb, clips, times, record)
    results = {}
    variants = ["original", "diagnostic_after_anchor", "all_i_steps"] if args.controls else ["original"]
    for variant in variants:
        torch.manual_seed(7)
        current = copy.deepcopy(cfg)
        if variant == "all_i_steps":
            current.transition.i_step_every = 1
        modules = TraceV3Modules(current, bb.hidden).to(args.device).train()
        wrapper = StudentLoss(bb, modules, current)
        wrapper.zero_grad(set_to_none=True)
        delta_norms = []

        def observe(_module, _inputs, info):
            indices = [i for i, kind in enumerate(info.kinds) if kind == "delta"]
            if indices:
                delta_norms.extend(info.tokens[indices].detach().float().norm(dim=-1).cpu().tolist())
                if variant == "diagnostic_after_anchor":
                    tokens = info.tokens.clone()
                    for i in indices:
                        tokens[i] = tokens[i] + info.tokens[i - 1].detach()
                    info.tokens = tokens
            return info

        hook = modules.transition.register_forward_hook(observe)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts, stats = wrapper(clips, times, record, teacher, captions)
            loss.backward()
        norms = [(float(p.grad.float().norm()), name) for name, p in wrapper.named_parameters()
                 if p.grad is not None]
        total = math.sqrt(sum(value * value for value, _ in norms))
        results[variant] = {
            "losses": parts, "stats": stats, "unclipped_gradient_norm": total,
            "top5": sorted(norms, reverse=True)[:5],
            "delta_norm_min": min(delta_norms) if delta_norms else None,
            "delta_norm_max": max(delta_norms) if delta_norms else None,
        }
        hook.remove()
        del wrapper, modules, loss
        gc.collect()
        torch.cuda.empty_cache()
    passed = math.isfinite(results["original"]["unclipped_gradient_norm"])
    passed = passed and results["original"]["unclipped_gradient_norm"] < args.max_grad
    return {"probe": "gradient", "pass": passed, "results": results}


def resume(args):
    from trace_vl.v3 import harness

    cfg, _ = load_v3_config(args.config)
    _, records = fixture(args.output / "fixture")
    adapter = args.output / "stub-adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"identity-only stub; never loaded")
    calls = []

    class StubBackbone:
        def __init__(self, *_args, **_kwargs):
            self.adapter = None
            self.model = SimpleNamespace(eval=lambda: None)

        def add_lora(self, *_args, adapter=None, **_kwargs):
            self.adapter = adapter

    class StubReader:
        duration = 3.0

        def __init__(self, *_args, **_kwargs):
            pass

        def step(self, *_args):
            return None

        def close(self):
            pass

    class StubSession:
        def __init__(self, backbone, *_args):
            self.backbone = backbone

        def step(self, *_args):
            return SimpleNamespace(tokens=196, is_i_step=True)

        def ask(self, *_args):
            calls.append(self.backbone.adapter)
            return int(self.backbone.adapter is None), [-0.1, -1.0]

        def stats(self):
            return {"kv_tokens": 1, "kv_bytes": 1, "state_bytes": 1}

    output = args.output / "evaluation"
    with patch.object(harness, "Backbone", StubBackbone), patch.object(harness, "StepReader", StubReader), \
            patch.object(harness, "StreamSession", StubSession):
        first = harness.evaluate(cfg, str(records), str(records.parent), str(output), adapter=str(adapter))
        try:
            second = harness.evaluate(cfg, str(records), str(records.parent), str(output), adapter=None)
            rejected = False
        except ValueError as error:
            second = {"rejected": str(error)}
            rejected = True
    return {"probe": "resume", "pass": rejected, "first": first, "second": second,
            "inference_calls": len(calls), "expected": "reject incompatible adapter before reusing predictions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=["gradient", "resume"])
    parser.add_argument("--config", default="configs/v3/stage1.yaml")
    parser.add_argument("--model", help="Optional local Qwen3.5-2B snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--max-grad", type=float, default=1e6)
    parser.add_argument("--output", type=Path, required=True, help="Use a fresh directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = gradient(args) if args.probe == "gradient" else resume(args)
    result["reviewed_commit"] = REVIEWED_COMMIT
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()

