"""Measure real-batch training capacity without saving model checkpoints or downloading weights."""

from __future__ import annotations

import argparse
import json
import time

import torch

from trace_vl.config import ModelConfig, load_config
from trace_vl.data.dataset import collate_samples
from trace_vl.models.model import TraceVL
from trace_vl.runtime import (
    autocast_for,
    initialize_weights,
    make_dataset,
    move_batch,
    seed_everything,
    write_json,
)
from trace_vl.training import TrainingLoss


def probe(args):
    config = load_config(args.config)
    config["model"].update(
        spatial_tokens=args.spatial_tokens, text_limit=args.text_limit, normalize_memory=args.normalize_memory
    )
    config["data"].update(max_frames=args.frames, sampling_fps=args.fps)
    seed_everything(7, 2)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Bound exploratory OOMs to this allocator; the process exits after each candidate.
    torch.cuda.set_per_process_memory_fraction(0.92, device)
    report = {
        "batch_size": args.batch_size,
        "frames": args.frames,
        "spatial_tokens": args.spatial_tokens,
        "fps": args.fps,
        "text_limit": args.text_limit,
        "config": config,
        "steps": args.steps,
        "detach_every": args.detach_every,
        "precision": args.precision,
        "normalize_memory": args.normalize_memory,
    }
    try:
        dataset = make_dataset(config, "train")
        indices = sorted(
            range(len(dataset)),
            key=lambda i: len(dataset.records[i]["question"]) + len(dataset.records[i]["answer"]),
            reverse=True,
        )
        samples, used = [], set()
        for index in indices:
            group = dataset.records[index]["source_group"]
            if group in used:
                continue
            sample = dataset[index]
            used.add(group)
            if len(sample["frames"]) != args.frames:
                continue
            samples.append(sample)
            if len(samples) == args.batch_size:
                break
        if len(samples) != args.batch_size:
            raise ValueError("not enough distinct training clips with the requested real frame count")
        report["sample_ids"] = [s["record"]["id"] for s in samples]
        model = TraceVL(ModelConfig.from_dict(config["model"])).to(device).train()
        report["initialization"] = initialize_weights(model, args.checkpoint)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        module = TrainingLoss(model, config["training"].get("loss_weights"), args.detach_every)
        batch = move_batch(collate_samples(samples), device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses = []
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            if args.anomaly:
                with torch.autograd.detect_anomaly(), autocast_for(device, args.precision):
                    loss, _ = module(batch)
                    loss.backward()
            else:
                with autocast_for(device, args.precision):
                    loss, _ = module(batch)
                if not torch.isfinite(loss):
                    report.update(status="nonfinite_loss")
                    break
                loss.backward()
            losses.append(float(loss.detach()))
            bad = [
                name
                for name, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()
            ]
            if bad:
                report.update(status="nonfinite_gradient", nonfinite_parameters=bad[:20])
                break
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
            )
            optimizer.step()
        torch.cuda.synchronize(device)
        report.update(
            status=report.get("status", "ok"),
            seconds_per_step=(time.perf_counter() - started) / args.steps,
            losses=losses,
            last_grad_norm=float(norm) if "norm" in locals() else None,
        )
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="oom", error=str(exc).split("\n")[0])
    report.update(
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"config", "initialization", "sample_ids"}}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_2b_pilot_cfr.yaml")
    parser.add_argument("--checkpoint", default="outputs/llava-2b-ddp-5epochs-20261105/best.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--spatial-tokens", type=int, default=192)
    parser.add_argument("--text-limit", type=int, default=512)
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--detach-every", type=int, default=0)
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--normalize-memory", action="store_true")
    probe(parser.parse_args())
