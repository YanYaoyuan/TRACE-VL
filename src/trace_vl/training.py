"""DDP-capable supervised training; optimizer steps, not microbatches, define max_steps."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import ModelConfig
from .data.dataset import collate_samples
from .models.model import TraceVL
from .runtime import (
    autocast_for,
    device_for,
    environment,
    file_hash,
    initialize_weights,
    load_checkpoint,
    make_dataset,
    move_batch,
    seed_everything,
    write_json,
)


class TrainingLoss(nn.Module):
    """Keep non-loss stream states outside DDP's output graph traversal."""

    def __init__(self, model, weights, detach_every):
        super().__init__()
        self.model, self.weights, self.detach_every = model, weights, detach_every

    def forward(self, batch):
        output = self.model(batch, self.weights, self.detach_every)
        return output["loss"], {k: v.detach() for k, v in output["losses"].items()}


def train(config, output, resume=None, init_from=None):
    if resume and init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    settings = config.get("training", {})
    seed_everything(config.get("seed", 7), settings.get("cpu_threads", 2))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = device_for(settings.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        dataset = make_dataset(config, "train")
        sampler = DistributedSampler(dataset, shuffle=True, seed=config.get("seed", 7)) if world > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=settings.get("batch_size", 2),
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=settings.get("workers", 0),
            collate_fn=collate_samples,
            pin_memory=device.type == "cuda",
            persistent_workers=settings.get("workers", 0) > 0,
            multiprocessing_context="spawn" if settings.get("workers", 0) > 0 else None,
        )
        model = TraceVL(ModelConfig.from_dict(config["model"])).to(device).train()
        initialization = initialize_weights(model, init_from) if init_from else None
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=settings.get("lr", 3e-4), weight_decay=settings.get("weight_decay", 0.01)
        )
        step = 0
        fingerprint = file_hash(config["data"]["manifest"])
        if resume:
            saved = load_checkpoint(resume)
            if (
                ModelConfig.from_dict(saved["config"]["model"]).to_dict() != config["model"]
                or saved["manifest_sha256"] != fingerprint
            ):
                raise ValueError("resume architecture/manifest mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            step = saved["step"]
        module = TrainingLoss(model, settings.get("loss_weights"), settings.get("detach_every", 0))
        if world > 1:
            module = DistributedDataParallel(
                module,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        directory = Path(output)
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
            write_json(
                directory / "run.json",
                {
                    "config": config,
                    "environment": environment(),
                    "manifest_sha256": fingerprint,
                    "parameters": model.parameter_report(),
                    "world_size": world,
                    "resume": str(resume) if resume else None,
                    "initialization": initialization,
                    "optimizer_learning_rates": [group["lr"] for group in optimizer.param_groups],
                },
            )
        if world > 1:
            dist.barrier()
        maximum = int(settings.get("max_steps", 10))
        accumulation = int(settings.get("grad_accum", 1))
        if maximum <= 0 or accumulation <= 0:
            raise ValueError("max_steps/grad_accum must be positive")
        keep_steps = set(settings.get("keep_steps", []))
        if any(isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in keep_steps):
            raise ValueError("keep_steps must contain positive integer optimizer steps")
        epoch, iterator = 0, iter(loader)

        def next_batch():
            nonlocal epoch, iterator
            try:
                return next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                return next(iterator)

        start = time.perf_counter()
        while step < maximum:
            optimizer.zero_grad(set_to_none=True)
            # Keep collective tensor shapes identical even when only some ranks have auxiliary labels.
            scalars = {k: 0.0 for k in ("total", "answer", "alert", "write", "utility")}
            for micro in range(accumulation):
                batch = move_batch(next_batch(), device)
                context = module.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
                with context, autocast_for(device, settings.get("precision", "fp32")):
                    loss, components = module(batch)
                    scaled = loss / accumulation
                # All ranks fail together, rather than leaving a peer in an NCCL collective.
                finite = torch.isfinite(loss.detach()).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("nonfinite training loss")
                scaled.backward()
                for k, v in {"total": loss.detach(), **components}.items():
                    scalars[k] = scalars.get(k, 0.0) + float(v) / accumulation
            norm = nn.utils.clip_grad_norm_(
                parameters, settings.get("clip_grad", 1.0), error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            values = torch.tensor(list(scalars.values()), device=device)
            if world > 1:
                dist.all_reduce(values)
                values /= world
            log = {
                "step": step,
                "losses": dict(zip(scalars, values.cpu().tolist())),
                "grad_norm_rank0": float(norm),
                "elapsed_seconds": time.perf_counter() - start,
            }
            if rank == 0:
                with (directory / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(log, allow_nan=False) + "\n")
                print(json.dumps(log), flush=True)
            if step == maximum or step % settings.get("save_every", maximum) == 0 or step in keep_steps:
                if rank == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "step": step,
                        "manifest_sha256": fingerprint,
                    }
                    temporary = directory / "checkpoint.tmp"
                    torch.save(checkpoint, temporary)
                    temporary.replace(directory / "last.pt")
                    if step in keep_steps:
                        # Atomic replacement of last.pt keeps retained hard links immutable.
                        retained = directory / f"step-{step:05d}.pt"
                        if retained.exists():
                            raise FileExistsError(f"refusing to replace retained checkpoint: {retained}")
                        os.link(directory / "last.pt", retained)
                if world > 1:
                    dist.barrier()
        local_resources = {
            "rank": rank,
            "device": str(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "seconds": time.perf_counter() - start,
        }
        rank_resources = [None] * world
        if world > 1:
            dist.all_gather_object(rank_resources, local_resources)
        else:
            rank_resources[0] = local_resources
        if rank == 0:
            write_json(
                directory / "resources.json",
                {
                    "device": str(device),
                    "world_size": world,
                    "rank0_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                    "rank0_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else None,
                    "seconds": time.perf_counter() - start,
                    "ranks": rank_resources,
                },
            )
        return directory / "last.pt"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
