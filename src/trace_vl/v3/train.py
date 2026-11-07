"""Stage 1: distil the full-token backbone (teacher) into the v3 stream (student).

One sample = a clip ending at a question time. The teacher sees all 196 tokens per step with LoRA
disabled; the student sees the v3 stream (CST + transition tokens) with LoRA enabled. Losses:
  KD (answer distribution) + CE (gold letter) + predictor regression + surprise budget
  + optional alert-head classification (Jev-style noul / choice) where records carry `alerts`:
    [{"id", "kind": "noul", "text", "spans": [[start, end], ...]},
     {"id", "kind": "choice", "text", "options": [...], "label_spans": [[label, start, end], ...], "default": 0}]
    (legacy `rules` + `alert_spans` [[rule, start, end], ...] become noul questions).
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

from .alert import DecisionMemory, Question, alert_loss, encode_questions
from .backbone import Backbone
from .config import V3Config
from .data import StepReader, read_jsonl, step_times
from .model import TraceV3Modules, stream_step
from .modules import CSTCache, TECache, TransitionTracker
from .positions import PositionCounter
from .session import LETTERS


def alert_questions(record) -> tuple[list[Question], list[dict]]:
    """Alert questions of a record and their label specs (legacy rules become noul questions)."""
    specs = list(record.get("alerts") or [])
    for i, rule in enumerate(record.get("rules") or []):
        spans = [[a, b] for r, a, b in record.get("alert_spans", []) if int(r) == i]
        specs.append({"id": f"rule{i}", "kind": "noul", "text": rule, "spans": spans})
    return [Question(s["id"], s["kind"], s["text"], s.get("options", [])) for s in specs], specs


def alert_target(spec: dict, t: float):
    if spec["kind"] == "noul":
        return float(any(a <= t <= b for a, b in spec.get("spans", [])))
    label = spec.get("default", 0)
    for value, a, b in spec.get("label_spans", []):
        if a <= t <= b:
            label = int(value)
    return label


def build_sequence(bb: Backbone, modules: TraceV3Modules | None, clips, times, record, tbptt: int = 0):
    """Whole-clip LM input (no cache): returns embeds [1, L, d], positions [3, L], per-step info."""
    pos = PositionCounter()
    prompt_ids = bb.ids(bb.head_text) + [bb.vision_start]  # outer video start (public processor protocol)
    parts, positions = [bb.embed_ids(prompt_ids)], [pos.text(len(prompt_ids), bb.device)]
    cst_cache, te_cache, tracker = CSTCache(), TECache(), TransitionTracker()
    step_ends, aux = [], {"pred": [], "budget": [], "tokens": [], "i_steps": 0, "clock": [], "spans": [], "xy": [],
                          "active": [], "max_mass": [], "memory_depth": 0,
                          "kinds": [], "is_i": []}  # fmt: skip
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
            aux["is_i"].append(True)
            aux["clock"].append(torch.zeros(4))
            kinds = ["frame"] * tokens.shape[0]
        else:
            if tbptt and k and k % tbptt == 0:
                cst_cache.detach()
                te_cache.detach()
            _, _, info = stream_step(modules, x, (h, w), t, cst_cache, te_cache)
            tokens, rows, cols, kinds = info.tokens, info.rows, info.cols, info.kinds
            if "pred_loss" in info.aux:
                aux["pred"].append(info.aux["pred_loss"])
            if not info.is_i_step:
                aux["budget"].append(info.aux["expected_tokens"])
                aux["active"].append(info.aux.get("active_slots", 0))
                aux["max_mass"].append(info.aux.get("max_mass", 0.0))
            aux["i_steps"] += int(info.is_i_step)
            aux["is_i"].append(info.is_i_step)
            slot_memory = info.aux.get("slot_memory")
            tracker.update(t, info.aux.get("active_centroids", []), slot_memory)
            aux["clock"].append(tracker.clock(t, float(info.surprise.mean())))
        ts = bb.ids(f"<{t:.1f} seconds>") + [bb.vision_start]
        parts += [bb.embed_ids(ts), tokens[None].to(bb.dtype), bb.embed_ids([bb.vision_end])]
        positions += [pos.text(len(ts), bb.device), pos.points(rows, cols, h, w), pos.text(1, bb.device)]
        aux["spans"].append((length + len(ts), length + len(ts) + tokens.shape[0]))
        aux["xy"].append(torch.stack([cols.float() / max(1, w - 1), rows.float() / max(1, h - 1)], dim=-1))
        aux["kinds"].append(kinds)
        length += len(ts) + tokens.shape[0] + 1
        step_ends.append(length - 1)
        aux["tokens"].append(tokens.shape[0])
    aux["memory_depth"] = 0 if cst_cache.memory is None else int(cst_cache.memory.shape[0])
    return torch.cat(parts, dim=1), torch.cat(positions, dim=1), pos, step_ends, aux


def question_ids(bb: Backbone, record) -> list[int]:
    lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(record["options"])]
    return [bb.vision_end] + bb.ids("\n" + record["question"] + "\n" + "\n".join(lines) + "\nAnswer with the option letter only." + bb.tail_text)


class StudentLoss(nn.Module):
    """Holds every trainable parameter (modules + LoRA via the backbone model) so DDP can sync them."""

    def __init__(self, bb: Backbone, modules: TraceV3Modules, cfg: V3Config):
        super().__init__()
        self.bb, self.cfg = bb, cfg
        self.v3 = modules
        self.lm_model = bb.model  # registers LoRA parameters with this module

    def forward(self, clips, times, record, teacher_logp, encoded=None, specs=None):
        bb, cfg, tc = self.bb, self.cfg, self.cfg.train
        embeds, positions, pos, step_ends, aux = build_sequence(bb, self.v3, clips, times, record, tc.tbptt_steps)
        q_ids = question_ids(bb, record)
        embeds = torch.cat([embeds, bb.embed_ids(q_ids)], dim=1)
        positions = torch.cat([positions, pos.text(len(q_ids), bb.device)], dim=1)
        taps = cfg.alert.taps if encoded is not None else ()
        hidden, tapped = bb.lm_taps(embeds, positions, cache=None, use_cache=False, taps=taps)
        logits = bb.logits(hidden[:, -1:])[0, -1].float()
        letters = [bb.ids(letter)[0] for letter in LETTERS[: len(record["options"])]]
        losses = {"ce": F.cross_entropy(logits[None], torch.tensor([letters[record["answer"]]], device=logits.device))}
        temp = tc.kd_temperature
        losses["kd"] = F.kl_div((logits / temp).log_softmax(-1), teacher_logp / temp, log_target=True,
                                reduction="sum") * temp * temp  # fmt: skip
        if aux["pred"]:
            losses["pred"] = torch.stack(aux["pred"]).mean()
        if aux["budget"]:
            # Two-sided: pulls activity up when transitions are absent and down when too many fire. The old
            # relu(mass - target) only suppressed surprise and had zero gradient once all slots were off.
            target = float(tc.target_tokens_per_step)
            expected = torch.stack([e if torch.is_tensor(e) else torch.tensor(float(e)) for e in aux["budget"]])
            losses["budget"] = (((expected.float() - target) / target) ** 2).mean()
        stats = {}
        if encoded is not None:
            loss, alert_stats = self._alert_loss(embeds, positions, hidden, tapped, step_ends, aux, times, encoded, specs)
            if loss is not None:
                losses["alert"] = loss
                stats.update(alert_stats)
        weights = {"ce": tc.ce_weight, "kd": tc.kd_weight, "pred": tc.pred_weight, "budget": tc.budget_weight,
                   "alert": tc.alert_weight}  # fmt: skip
        total = sum(weights[k] * v for k, v in losses.items())
        p_tokens = [n for n, i in zip(aux["tokens"], aux["is_i"]) if not i]
        if aux["active"]:
            stats.update(active_slots_mean=sum(aux["active"]) / len(aux["active"]),
                         p_steps_with_active=sum(a > 0 for a in aux["active"]) / len(aux["active"]),
                         active_slots_max=max(aux["active"]), max_mass_mean=sum(aux["max_mass"]) / len(aux["max_mass"]),
                         expected_p_tokens=float(torch.stack(aux["budget"]).float().mean()),
                         cst_memory_depth=aux["memory_depth"])  # fmt: skip
        stats.update(tokens_per_step=sum(aux["tokens"]) / max(1, len(aux["tokens"])),
                     p_step_tokens_mean=sum(p_tokens) / len(p_tokens) if p_tokens else None,
                     total_visual_tokens=sum(aux["tokens"]), i_step_fraction=aux["i_steps"] / max(1, len(times)))  # fmt: skip
        return total, {k: float(v) for k, v in losses.items()}, stats

    def _alert_loss(self, embeds, positions, hidden, tapped, step_ends, aux, times, encoded, specs):
        """Alert head at up to `alert_steps` steps (always the last), with latent memory of earlier steps."""
        bb, cfg = self.bb, self.cfg
        n = len(times)
        chosen = sorted(set(random.sample(range(n), min(n, cfg.alert.train_steps - 1)) + [n - 1]))
        states = None
        if random.random() < 1.0 / max(1, cfg.alert.state_every):
            # End-of-clip GDN states from a no-grad pass with a cache (checkpointed training drops caches).
            # Autocast off (weights are already bf16) and its cache cleared: a no-grad pass inside autocast
            # pollutes the cast-weight cache that checkpoint replay relies on (PR #3 review, CheckpointError).
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                bb.language_model.eval()
                cache = bb.new_cache()
                bb.lm(embeds.detach().to(bb.dtype), positions, cache, use_cache=True)
                bb.language_model.train()
                states = bb.gdn_states(cache, cfg.alert.state_layers)
            torch.clear_autocast_cache()
        memory = DecisionMemory(cfg.alert.memory_tokens)
        losses, logs = [], {}
        for k in range(n):
            if k in chosen:
                start, end = aux["spans"][k]
                results = self.v3.alert(encoded, tapped[:, 0, start:end], aux["xy"][k].to(hidden.device), memory,
                                        times[k], aux["clock"][k], states if k == n - 1 else None)  # fmt: skip
                targets = {spec["id"]: alert_target(spec, times[k]) for spec in specs}
                loss, logs = alert_loss(results, targets)
                if loss is not None:
                    losses.append(loss)
            keep = [i for i, kind in enumerate(aux["kinds"][k]) if kind in {"gist", "after", "delta"}]
            start, end = aux["spans"][k]
            latent = [hidden[0, step_ends[k] : step_ends[k] + 1]] + ([hidden[0, start:end][keep]] if keep else [])
            memory.add(torch.cat(latent), times[k])
        logs["alert_state_used"] = float(states is not None)
        return (torch.stack(losses).mean() if losses else None), logs


@torch.no_grad()
def teacher_targets(bb: Backbone, clips, times, record):
    """Full-token teacher with LoRA disabled: answer log-probs."""
    context = bb.peft.disable_adapter() if bb.peft is not None else torch.no_grad()
    with context:
        embeds, positions, pos, _, _ = build_sequence(bb, None, clips, times, record)
        q_ids = question_ids(bb, record)
        embeds = torch.cat([embeds, bb.embed_ids(q_ids)], dim=1)
        positions = torch.cat([positions, pos.text(len(q_ids), bb.device)], dim=1)
        hidden = bb.lm(embeds, positions, cache=None, use_cache=False)
        return bb.logits(hidden[:, -1:])[0, -1].float().log_softmax(-1)


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
                teacher_logp = teacher_targets(bb, clips, times, record)
                questions, specs = alert_questions(record)
                encoded = encode_questions(bb, questions) if questions else None
                sync = world == 1 or micro == tc.grad_accum - 1
                context = module.no_sync() if not sync else torch.enable_grad()
                with context, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, parts, stats = module(clips, times, record, teacher_logp, encoded, specs)
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
    record["alerts"] = [{"id": "abnormal", "kind": "noul", "text": "Is anything abnormal happening?", "spans": []}]
    questions, specs = alert_questions(record)
    teacher_logp = teacher_targets(bb, clips, times, record)
    encoded = encode_questions(bb, questions)  # before autocast, exactly as in train() (PR #3 review)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts, stats = wrapper(clips, times, record, teacher_logp, encoded, specs)
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
