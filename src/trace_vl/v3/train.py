"""Stage 1: distil the full-token backbone (teacher) into the v3 stream (student).

One sample = a clip ending at a question time. The teacher sees all 196 tokens per step with LoRA
disabled; the student sees the v3 stream (CST + transition tokens) with LoRA enabled. Losses:
  KD (answer distribution) + CE (gold letter) + predictor regression + surprise budget
  + optional alert BCE (records with `rules` and `alert_spans`)
  + optional JEPA-style event loss (records with `captions`: [[start, end, text], ...]).
Launch with torchrun for several GPUs (one process per GPU).
"""

from __future__ import annotations

import json
import math
import os
import random
import time as clock
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .config import V3Config
from .data import StepReader, read_jsonl, step_times
from .model import TraceV3Modules
from .modules import CSTCache, TECache, TransitionTracker
from .positions import PositionCounter
from .session import LETTERS


def _prompt(bb: Backbone, rules: list[str]):
    """Prompt ids and per-rule token spans (same layout as StreamSession)."""
    ids, spans = bb.ids(bb.head_text), []
    if rules:
        ids += bb.ids("Monitoring rules:\n")
        for i, rule in enumerate(rules):
            ids += bb.ids(f"{i + 1}. ")
            start = len(ids)
            ids += bb.ids(rule)
            spans.append((start, len(ids)))
            ids += bb.ids("\n")
        ids += bb.ids("Video stream:\n")
    return ids, spans


def build_sequence(bb: Backbone, modules: TraceV3Modules | None, clips, times, record, tbptt: int = 0):
    """Whole-clip LM input (no cache): returns embeds [1, L, d], positions [3, L], per-step info."""
    pos = PositionCounter()
    prompt_ids, spans = _prompt(bb, record.get("rules") or [])
    parts, positions = [bb.embed_ids(prompt_ids)], [pos.text(len(prompt_ids), bb.device)]
    cst_cache, te_cache, tracker = CSTCache(), TECache(), TransitionTracker()
    step_ends, aux = [], {"pred": [], "budget": [], "tokens": [], "i_steps": 0, "clock": []}
    dtype = next(modules.parameters()).dtype if modules is not None else bb.dtype
    length = len(prompt_ids)
    for k, (frames, t) in enumerate(zip(clips, times)):
        with torch.no_grad():
            pixel_values, grid_thw = bb.preprocess(bb.resize_step(frames))
            x = bb.encode(pixel_values, grid_thw)
        h, w = bb.merged_grid(grid_thw)
        if modules is None:
            tokens = x
            rows = torch.arange(h, device=x.device).repeat_interleave(w)
            cols = torch.arange(w, device=x.device).repeat(h)
            aux["i_steps"] += 1
            aux["clock"].append(torch.zeros(4))
        else:
            if tbptt and k and k % tbptt == 0:
                cst_cache.detach()
                te_cache.detach()
            adapted, z = modules.cst(x.to(dtype), (h, w), t, cst_cache)
            info = modules.transition(adapted, z, (h, w), te_cache)
            tokens, rows, cols = info.tokens, info.rows, info.cols
            if "pred_loss" in info.aux:
                aux["pred"].append(info.aux["pred_loss"])
            if not info.is_i_step:
                aux["budget"].append(info.masses.sum())
            aux["i_steps"] += int(info.is_i_step)
            slot_memory = info.aux.get("slot_memory")
            tracker.update(t, info.aux.get("active_centroids", []), slot_memory)
            aux["clock"].append(tracker.clock(t, float(info.surprise.mean())))
            if slot_memory is not None:
                tt = torch.full((slot_memory.shape[0],), t, device=slot_memory.device, dtype=slot_memory.dtype)
                cap = modules.cfg.transition.memory_slots
                cst_cache.memory = slot_memory if cst_cache.memory is None else torch.cat([cst_cache.memory, slot_memory])[-cap:]
                cst_cache.memory_times = tt if cst_cache.memory_times is None else torch.cat([cst_cache.memory_times, tt])[-cap:]
        ts = bb.ids(f"<{t:.1f} seconds>") + [bb.vision_start]
        parts += [bb.embed_ids(ts), tokens[None].to(bb.dtype), bb.embed_ids([bb.vision_end])]
        positions += [pos.text(len(ts), bb.device), pos.points(rows, cols, h, w), pos.text(1, bb.device)]
        length += len(ts) + tokens.shape[0] + 1
        step_ends.append(length - 1)
        aux["tokens"].append(tokens.shape[0])
    return torch.cat(parts, dim=1), torch.cat(positions, dim=1), pos, step_ends, spans, aux


def question_ids(bb: Backbone, record) -> list[int]:
    lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(record["options"])]
    return bb.ids("\n" + record["question"] + "\n" + "\n".join(lines) + "\nAnswer with the option letter only." + bb.tail_text)


class StudentLoss(nn.Module):
    """Holds every trainable parameter (modules + LoRA via the backbone model) so DDP can sync them."""

    def __init__(self, bb: Backbone, modules: TraceV3Modules, cfg: V3Config):
        super().__init__()
        self.bb, self.cfg = bb, cfg
        self.v3 = modules
        self.lm_model = bb.model  # registers LoRA parameters with this module

    def forward(self, clips, times, record, teacher_logp, caption_targets):
        bb, cfg, tc = self.bb, self.cfg, self.cfg.train
        embeds, positions, pos, step_ends, spans, aux = build_sequence(bb, self.v3, clips, times, record, tc.tbptt_steps)
        q_ids = question_ids(bb, record)
        embeds = torch.cat([embeds, bb.embed_ids(q_ids)], dim=1)
        positions = torch.cat([positions, pos.text(len(q_ids), bb.device)], dim=1)
        hidden = bb.lm(embeds, positions, cache=None, use_cache=False)
        logits = bb.logits(hidden[:, -1:])[0, -1].float()
        letters = [bb.ids(letter)[0] for letter in LETTERS[: len(record["options"])]]
        losses = {"ce": F.cross_entropy(logits[None], torch.tensor([letters[record["answer"]]], device=logits.device))}
        temp = tc.kd_temperature
        losses["kd"] = F.kl_div((logits / temp).log_softmax(-1), teacher_logp / temp, log_target=True,
                                reduction="sum") * temp * temp  # fmt: skip
        if aux["pred"]:
            losses["pred"] = torch.stack(aux["pred"]).mean()
        if aux["budget"]:
            target = max(0.0, (tc.target_tokens_per_step - cfg.transition.gist) / 3) * cfg.transition.mass_threshold
            losses["budget"] = F.relu(torch.stack(aux["budget"]) - target).mean()
        dtype = next(self.v3.parameters()).dtype
        step_hidden = hidden[0, step_ends].to(dtype)  # [steps, d]
        if record.get("rules") and record.get("alert_spans"):
            rule_hidden = torch.stack([hidden[0, a:b].mean(0) for a, b in spans]).to(dtype)
            labels = torch.zeros(len(times), len(record["rules"]), device=hidden.device)
            for rule, start, end in record["alert_spans"]:
                for k, t in enumerate(times):
                    if start <= t <= end:
                        labels[k, int(rule)] = 1.0
            clock_feats = torch.stack(aux["clock"]).to(hidden.device, dtype)  # transition-track clocks per step
            alert = torch.stack([self.v3.alert(step_hidden[k], rule_hidden, clock_feats[k]) for k in range(len(times))])
            losses["alert"] = F.binary_cross_entropy_with_logits(alert.float(), labels)
        if caption_targets is not None:
            mask, targets = caption_targets
            if mask.any():
                pred = self.v3.event(step_hidden[mask])
                tgt = self.v3.event.embed_text(targets.to(dtype))[mask]
                sim = pred @ tgt.t() / 0.07
                losses["event"] = F.cross_entropy(sim.float(), torch.arange(sim.shape[0], device=sim.device))
        weights = {"ce": tc.ce_weight, "kd": tc.kd_weight, "pred": tc.pred_weight, "budget": tc.budget_weight,
                   "alert": 1.0, "event": 0.5}  # fmt: skip
        total = sum(weights[k] * v for k, v in losses.items())
        stats = {"tokens_per_step": sum(aux["tokens"]) / max(1, len(aux["tokens"])), "i_step_fraction": aux["i_steps"] / max(1, len(times))}
        return total, {k: float(v) for k, v in losses.items()}, stats


@torch.no_grad()
def teacher_targets(bb: Backbone, clips, times, record):
    """Full-token teacher with LoRA disabled: answer log-probs and caption embeddings per step."""
    context = bb.peft.disable_adapter() if bb.peft is not None else torch.no_grad()
    with context:
        embeds, positions, pos, step_ends, _, _ = build_sequence(bb, None, clips, times, record)
        q_ids = question_ids(bb, record)
        embeds = torch.cat([embeds, bb.embed_ids(q_ids)], dim=1)
        positions = torch.cat([positions, pos.text(len(q_ids), bb.device)], dim=1)
        hidden = bb.lm(embeds, positions, cache=None, use_cache=False)
        logp = bb.logits(hidden[:, -1:])[0, -1].float().log_softmax(-1)
        captions = None
        if record.get("captions"):
            mask = torch.zeros(len(times), dtype=torch.bool, device=bb.device)
            targets = torch.zeros(len(times), bb.hidden, device=bb.device)
            for start, end, text in record["captions"]:
                ids = bb.ids(text)
                if not ids:
                    continue
                pooled = bb.lm(bb.embed_ids(ids), PositionCounter().text(len(ids), bb.device), cache=None,
                               use_cache=False)[0].mean(0)  # fmt: skip
                for k, t in enumerate(times):
                    if start <= t <= end:
                        mask[k], targets[k] = True, pooled.float()
            captions = (mask, targets)
    return logp, captions


def load_clip(record, video_root, cfg: V3Config, clip_seconds: float):
    gap = cfg.stream.frame_gap
    reader = StepReader(record, video_root)
    try:
        end = min(record["query_time"] if record.get("query_time") is not None else reader.duration, reader.duration)
        start = max(0.0, end - clip_seconds)
        times = [t + start for t in step_times(end - start, cfg.stream.step_seconds, gap) if t + start + gap / 2 <= end + 1e-6]
        clips = [reader.step(t, gap) for t in times]
    finally:
        reader.close()
    return clips, times


def train(cfg: V3Config, records_path: str, video_root: str, output: str, clip_seconds: float = 120.0,
          max_steps: int | None = None, init: str | None = None, adapter: str | None = None):  # fmt: skip
    tc = cfg.train
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}"
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    random.seed(tc.seed + rank)
    torch.manual_seed(tc.seed)
    try:
        records = [r for r in read_jsonl(records_path) if r.get("split", "train") == "train"]
        if not records:
            raise ValueError("no training records (split == train)")
        bb = Backbone(cfg.backbone, cfg.stream, device)
        bb.add_lora(tc.lora_rank, tc.lora_alpha, adapter=adapter)
        bb.enable_checkpointing()
        modules = TraceV3Modules.load(init, cfg, bb.hidden) if init else TraceV3Modules(cfg, bb.hidden)
        modules = modules.to(device).train()
        wrapper = StudentLoss(bb, modules, cfg)
        lora_params = [p for p in bb.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params": list(modules.parameters()), "lr": tc.lr, "weight_decay": tc.weight_decay},
             {"params": lora_params, "lr": tc.lora_lr, "weight_decay": 0.0}],
        )  # fmt: skip
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        module = wrapper
        if world > 1:
            module = nn.parallel.DistributedDataParallel(wrapper, device_ids=[local], find_unused_parameters=True)
        output = Path(output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.json").write_text(json.dumps({"config": cfg.to_dict(), "records": records_path,
                                                         "modules": modules.parameter_report(),
                                                         "lora_params": sum(p.numel() for p in lora_params),
                                                         "world": world}, indent=2) + "\n")  # fmt: skip
        total_steps = max_steps or tc.steps
        started = clock.perf_counter()
        for step in range(total_steps):
            factor = min(1.0, (step + 1) / max(1, tc.warmup))
            progress = max(0.0, (step - tc.warmup) / max(1, total_steps - tc.warmup))
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * factor * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            optimizer.zero_grad(set_to_none=True)
            logs, stats = {}, {}
            for micro in range(tc.grad_accum):
                record = random.choice(records)
                clips, times = load_clip(record, video_root, cfg, clip_seconds)
                if not times:
                    continue
                teacher_logp, captions = teacher_targets(bb, clips, times, record)
                sync = world == 1 or micro == tc.grad_accum - 1
                context = module.no_sync() if not sync else torch.enable_grad()
                with context, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, parts, stats = module(clips, times, record, teacher_logp, captions)
                    (loss / tc.grad_accum).backward()
                for k, v in parts.items():
                    logs[k] = logs.get(k, 0.0) + v / tc.grad_accum
            # Clip each group separately: one global norm lets a spike in the new modules shrink LoRA updates.
            norms = {}
            for name, group in zip(("modules", "lora"), optimizer.param_groups):
                params = [p for p in group["params"] if p.grad is not None]
                norms[name] = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if params else 0.0
            if not all(math.isfinite(v) for v in norms.values()):
                raise FloatingPointError(f"nonfinite gradient norm: {norms}")
            optimizer.step()
            if rank == 0:
                line = {"step": step + 1, **{k: round(v, 5) for k, v in logs.items()}, **stats,
                        "grad_norm_modules": norms["modules"], "grad_norm_lora": norms["lora"],
                        "gate_abs_mean": float(modules.cst.gate.abs().mean()),
                        "lr": optimizer.param_groups[0]["lr"], "elapsed": round(clock.perf_counter() - started, 1),
                        "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9}  # fmt: skip
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(line) + "\n")
                print(json.dumps(line), flush=True)
                if (step + 1) % tc.save_every == 0 or step + 1 == total_steps:
                    modules.save(output / "modules.pt", {"step": step + 1})
                    bb.peft.save_pretrained(output / "lora")
        return output
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def gradient_check(cfg: V3Config, steps: int = 6, threshold: float = 1e4, device: str = "cuda") -> dict:
    """Regression check (PR #3 review): unclipped gradient norms on a static clip must stay moderate.

    Static frames make after - before exactly zero; before the anchored-delta fix this gave ~1e12.
    """
    bb = Backbone(cfg.backbone, cfg.stream, device)
    bb.add_lora(cfg.train.lora_rank, cfg.train.lora_alpha)
    bb.enable_checkpointing()
    modules = TraceV3Modules(cfg, bb.hidden).to(device).train()
    wrapper = StudentLoss(bb, modules, cfg)
    frames = torch.full((2, 448, 448, 3), 128, dtype=torch.uint8).numpy()
    clips = [frames] * steps
    times = [k + cfg.stream.frame_gap / 2 for k in range(steps)]
    record = {"question": "What is shown?", "options": ["a gray image", "a cat", "a car", "a tree"], "answer": 0}
    teacher_logp, _ = teacher_targets(bb, clips, times, record)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts, stats = wrapper(clips, times, record, teacher_logp, None)
    loss.backward()
    groups = {"modules": list(modules.parameters()), "lora": [p for p in bb.model.parameters() if p.requires_grad]}
    norms = {}
    for name, params in groups.items():
        grads = [p.grad.float().norm() for p in params if p.grad is not None]
        norms[name] = float(torch.stack(grads).norm()) if grads else 0.0
    top = sorted(((float(p.grad.float().norm()), n) for n, p in modules.named_parameters() if p.grad is not None),
                 reverse=True)[:5]  # fmt: skip
    total = math.sqrt(sum(v * v for v in norms.values()))
    return {"losses": parts, "stats": stats, "grad_norms": norms, "total": total, "top_module_params": top,
            "threshold": threshold, "pass": math.isfinite(total) and total < threshold}  # fmt: skip
