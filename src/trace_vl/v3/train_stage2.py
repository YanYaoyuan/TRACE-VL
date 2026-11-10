"""Stage 2: train the v3 stream on long videos through the real streaming cache.

Stage 1 sees short clips without a cache, so nothing in it ever learns what the full-attention KV policy
throws away or what the Gated DeltaNet state carries over minutes. Stage 2 runs the inference path:
    prompt -> steps (CST + transition encoder) -> chunk LM forward on the hybrid cache -> TKM eviction
    -> at question times: fork the cache (shared KV, cloned GDN states), close the video, ask, CE
Records come from `trace-vl v3 build-stage2` (long streams with delayed, time-referenced questions and
evidence windows; see data.build_stage2_streams).

Gradients (truncated BPTT):
- The stream is cut into chunks of `stage2.chunk_steps` steps that also end at question times. Each chunk
  is one LM forward on the cache (with grad only when a question is due), losses, backward, then every
  cache (KV, GDN, CST, transition) is detached. Question CE therefore trains LoRA (reading the compressed
  memory) and the modules of the current chunk; it does not reach the evidence steps of earlier chunks.
- That gap is covered by the retention scorer: at every chunk end, every older KV group (transition slots and
  whole steps, so the pooled B4 arm is scored the same way) is rescored with its current age and gets BCE
  against "a question still to come is about this group's time" (labels from the annotations: teacher
  forcing of what to keep). Unqueried distractor clips and already-answered evidence are the negatives. The
  same scores drive the TKM eviction during training, as at inference.
- The Gated DeltaNet cache updates are made functional (backbone.functional_linear_cache), otherwise the
  in-place state copies break autograd.
- Streams have different numbers of chunks per rank, so DDP's hooks would deadlock; gradients are
  all-reduced by hand once per optimizer update.
The LM runs in eval mode (no LoRA dropout) with gradients and without activation checkpointing:
transformers disables the cache under training-mode checkpointing.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time as clock
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .backbone import Backbone, branch_cache, detach_cache, functional_linear_cache
from .config import V3Config
from .data import StepReader, read_jsonl, step_times
from .kv import KINDS, KVMemory, RetentionBook, score_retention
from .model import TraceV3Modules, stream_step
from .modules import CSTCache, TECache
from .positions import PositionCounter
from .session import LETTERS, observe_step
from .tracking import Tracker, enabled_from_env
from .train import question_ids

VISUAL_KINDS = {KINDS[k] for k in ("frame", "gist", "before", "after", "delta")}


# ---------------------------------------------------------------------------- planning (pure, CPU-testable)
def stream_times(record: dict, step_seconds: float, gap: float, max_seconds: float) -> list[float]:
    """Step centres of a stream, each step's window inside the stream and before `max_seconds`."""
    end = min(float(record["duration"]), max_seconds)
    return [t for t in step_times(end, step_seconds, gap) if t + gap / 2 <= end + 1e-6]


def chunk_plan(times: list[float], questions: list[dict], chunk_steps: int, gap: float) -> list[tuple[list[int], list[dict]]]:
    """Split step indices into chunks of at most `chunk_steps`; a chunk also ends after the last step that a
    question can see (step window end <= question time) and carries the questions due there.
    Questions before the first step's window or after the last step are dropped."""
    if not times:
        return []
    stream_end = times[-1] + gap / 2
    due: dict[int, list[dict]] = {}
    for q in questions:
        when = float(q["time"])
        seen = [k for k, t in enumerate(times) if t + gap / 2 <= when + 1e-6]
        # A question timed after the (possibly cut) stream would be asked on a stream missing its tail.
        if seen and when <= stream_end + 1.0:
            due.setdefault(seen[-1], []).append(q)
    plan, current = [], []
    for k in range(len(times)):
        current.append(k)
        if len(current) >= chunk_steps or k in due or k == len(times) - 1:
            plan.append((current, due.get(k, [])))
            current = []
    return plan


def eviction_steps(times: list[float], questions: list[dict], chunk_steps: int, gap: float) -> set[int]:
    """Step indices after which Stage-2 training evicts: the last step of every chunk of `chunk_plan` (chunks
    of `chunk_steps` that also end where a question is due, plus the final partial chunk). Evaluation with
    `evict_schedule="train"` uses exactly this set, so train/eval eviction parity is by construction."""
    return {steps[-1] for steps, _ in chunk_plan(times, questions, chunk_steps, gap)}


def evidence_label(span: tuple[float, float], now: float, questions: list[dict]) -> float:
    """1 if a question still to be asked (time > now) has an evidence window overlapping the group's time span.
    Groups of unqueried (distractor) clips, and groups whose questions were already asked, are negatives."""
    first, last = span
    for q in questions:
        if float(q["time"]) > now and any(a <= last and first <= b for a, b in (q.get("evidence") or [])):
            return 1.0
    return 0.0


def evidence_kept(memory: KVMemory, appended: list[tuple[float, int]], evidence) -> float | None:
    """Fraction of the visual tokens streamed inside the evidence windows that are still in the KV cache."""
    if not evidence:
        return None

    def inside(t):
        return any(a <= t <= b for a, b in evidence)

    total = sum(n for t, n in appended if inside(t))
    if total == 0:
        return None
    kept = sum(1 for k, t in zip(memory.kind, memory.time) if k in VISUAL_KINDS and inside(t))
    return kept / total


# ---------------------------------------------------------------------------- one stream
class Stage2Runner:
    def __init__(self, bb: Backbone, modules: TraceV3Modules, cfg: V3Config):
        self.bb, self.modules, self.cfg = bb, modules, cfg
        self.letter_ids = [bb.ids(letter)[0] for letter in LETTERS]

    def run(self, record: dict, video_root: str, scale: float = 1.0) -> dict:
        """Stream one record, backpropagate every chunk (loss x `scale`), return statistics."""
        bb, cfg, modules = self.bb, self.cfg, self.modules
        s2, tc, gap = cfg.stage2, cfg.train, cfg.stream.frame_gap
        times = stream_times(record, cfg.stream.step_seconds, gap, s2.max_seconds)
        if not times:
            return {"steps": 0}
        questions = record.get("questions", [])
        plan = chunk_plan(times, questions, s2.chunk_steps, gap)
        n_steps = len(times)
        n_questions = max(1, sum(len(d) for _, d in plan))
        cache, pos, memory, book = bb.new_cache(), PositionCounter(), KVMemory(cfg.memory), RetentionBook()
        cst_cache, te_cache = CSTCache(), TECache()
        prompt = bb.ids(bb.head_text) + [bb.vision_start]
        pending = [(bb.embed_ids(prompt), pos.text(len(prompt), bb.device))]
        memory.append(len(prompt), "prompt", -1.0, ("prompt",), protected=True)
        appended: list[tuple[float, int]] = []
        stats = {"steps": n_steps, "chunks": len(plan), "ce": [], "correct": [], "evidence_kept": [], "delays": [],
                 "expected": [], "active": [], "retention": [], "tokens": 0, "ret_pos": 0, "ret_neg": 0}  # fmt: skip
        cuda = bb.device.type == "cuda"
        reader = StepReader(record, video_root)
        try:
            with functional_linear_cache():
                for step_ids, due in plan:
                    parts = {"pred": [], "budget": []}
                    t = times[step_ids[-1]]
                    for k in step_ids:
                        t = times[k]
                        frames = reader.step(t, gap)
                        with torch.no_grad():
                            pixel_values, grid_thw = bb.preprocess(bb.resize_step(frames))
                            x = bb.encode(pixel_values, grid_thw)
                        h, w = bb.merged_grid(grid_thw)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                            _, _, info = stream_step(modules, x, (h, w), t, cst_cache, te_cache)
                        if "pred_loss" in info.aux:
                            parts["pred"].append(info.aux["pred_loss"])
                        if not info.is_i_step:
                            parts["budget"].append(info.aux["expected_tokens"])
                            stats["expected"].append(float(info.aux["expected_tokens"]))
                            stats["active"].append(int(info.aux.get("active_slots", 0)))
                        groups = [("step", k) if g < 0 else ("slot", g) for g in info.groups]
                        ts = bb.ids(f"<{t:.1f} seconds>") + [bb.vision_start]
                        n = info.tokens.shape[0]
                        embeds = torch.cat([bb.embed_ids(ts), info.tokens[None].to(bb.dtype), bb.embed_ids([bb.vision_end])], 1)
                        positions = torch.cat([pos.text(len(ts), bb.device), pos.points(info.rows, info.cols, h, w),
                                               pos.text(1, bb.device)], 1)  # fmt: skip
                        pending.append((embeds, positions))
                        memory.append(len(ts), "text", t, ("step", k))
                        memory.append(n, info.kinds, t, groups)
                        memory.append(1, "text", t, ("step", k))
                        appended.append((t, n))
                        stats["tokens"] += n
                        if cfg.memory.policy in {"tkm", "mass"}:
                            observe_step(book, info, t, k)  # same bookkeeping as StreamSession

                    # One LM forward for the chunk; gradients only matter when a question reads the cache.
                    embeds = torch.cat([e for e, _ in pending], 1)
                    positions = torch.cat([p for _, p in pending], 1)
                    pending = []
                    grad = bool(due)
                    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16,
                                                                      enabled=cuda):  # fmt: skip
                        bb.lm(embeds if grad else embeds.detach(), positions, cache)
                    # Retention: rescore every tracked group with its current age (trainable for tkm), then evict.
                    keys, ret_logits = score_retention(book, memory, modules.retention, t)
                    if ret_logits is not None:
                        ret_labels = [evidence_label(book.span(g), t, questions) for g in keys]
                        stats["ret_pos"] += int(sum(ret_labels))
                        stats["ret_neg"] += len(ret_labels) - int(sum(ret_labels))
                    mask = memory.keep_mask(t)
                    if mask is not None:
                        bb.evict(cache, mask)
                        memory.apply(mask)
                        book.prune(set(memory.group))

                    losses = []
                    for q in due:
                        kept = evidence_kept(memory, appended, q.get("evidence"))
                        if kept is not None:
                            stats["evidence_kept"].append(kept)
                        stats["delays"].append(float(q.get("delay", 0.0)))
                        fork, qpos = branch_cache(cache), copy.deepcopy(pos)
                        ids = question_ids(bb, q)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                            hidden = bb.lm(bb.embed_ids(ids), qpos.text(len(ids), bb.device), fork)
                        logits = bb.logits(hidden[:, -1:])[0, -1].float()
                        letters = self.letter_ids[: len(q["options"])]
                        target = torch.tensor([letters[q["answer"]]], device=logits.device)
                        ce = F.cross_entropy(logits[None], target)
                        losses.append(s2.question_weight * ce / n_questions)
                        stats["ce"].append(float(ce))
                        stats["correct"].append(int(int(logits[letters].argmax()) == q["answer"]))
                        del fork
                    share = len(step_ids) / n_steps
                    if parts["pred"]:
                        losses.append(tc.pred_weight * share * torch.stack(parts["pred"]).mean())
                    if parts["budget"]:
                        target = float(tc.target_p_step_tokens)
                        expected = torch.stack([e if torch.is_tensor(e) else torch.tensor(float(e), device=bb.device)
                                                for e in parts["budget"]]).float()  # fmt: skip
                        losses.append(tc.budget_weight * share * (((expected - target) / target) ** 2).mean())
                    if ret_logits is not None:
                        logit = ret_logits.float()
                        label = torch.tensor(ret_labels, device=logit.device)
                        bce = F.binary_cross_entropy_with_logits(logit, label)
                        losses.append(s2.retention_weight * share * bce)
                        stats["retention"].append(float(bce))
                    total = sum(losses) if losses else None
                    if total is not None and torch.is_tensor(total) and total.requires_grad:
                        (total * scale).backward()
                    detach_cache(cache)
                    cst_cache.detach()
                    te_cache.detach()
        finally:
            reader.close()
        stats["kv_tokens"] = len(memory)
        stats["evicted_tokens"] = memory.evicted_tokens
        return stats


def summarize(stats: list[dict]) -> dict:
    def mean(key):
        values = [v for s in stats for v in s.get(key, [])]
        return sum(values) / len(values) if values else None

    out = {"questions": sum(len(s.get("ce", [])) for s in stats), "q_ce": mean("ce"), "q_acc": mean("correct"),
           "evidence_kept": mean("evidence_kept"), "mean_delay": mean("delays"), "expected_p_tokens": mean("expected"),
           "retention_bce": mean("retention"), "steps": sum(s.get("steps", 0) for s in stats),
           "kv_tokens_end": sum(s.get("kv_tokens", 0) for s in stats) / max(1, len(stats)),
           "ret_pos": sum(s.get("ret_pos", 0) for s in stats), "ret_neg": sum(s.get("ret_neg", 0) for s in stats)}  # fmt: skip
    active = [a for s in stats for a in s.get("active", [])]
    if active:
        out["p_steps_with_active"] = sum(a > 0 for a in active) / len(active)
        out["active_slots_mean"] = sum(active) / len(active)
    steps = max(1, out["steps"])
    out["tokens_per_step"] = sum(s.get("tokens", 0) for s in stats) / steps
    return out


def all_reduce_grads(params: list[torch.nn.Parameter], world: int):
    """Average gradients across ranks in one flat buffer. A parameter missing a gradient on one rank counts as
    zero there; a parameter without a gradient on every rank keeps grad=None, so AdamW skips it (weight decay
    included), exactly as on one GPU (PR #3 review: the unused alert head drifted under decay)."""
    if world <= 1:
        return
    device = params[0].device
    has = torch.tensor([float(p.grad is not None) for p in params], device=device)
    dist.all_reduce(has)
    used = [p for p, h in zip(params, has.tolist()) if h > 0]
    if not used:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in used]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for p, g in zip(used, grads):
        n = g.numel()
        p.grad = flat[offset : offset + n].view_as(g).to(g.dtype)
        offset += n


# ---------------------------------------------------------------------------- trainer
def train_stage2(cfg: V3Config, streams_path: str, video_root: str, output: str, init: str | None = None,
                 adapter: str | None = None, max_steps: int | None = None):  # fmt: skip
    tc, s2 = cfg.train, cfg.stage2
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}"
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    random.seed(tc.seed + rank)
    torch.manual_seed(tc.seed)
    tracker = Tracker()
    try:
        records = [r for r in read_jsonl(streams_path) if r.get("split", "train") == "train" and r.get("questions")]
        if not records:
            raise ValueError("no Stage 2 training streams (split == train with questions); run `v3 build-stage2`")
        bb = Backbone(cfg.backbone, cfg.stream, device)
        bb.add_lora(tc.lora_rank, tc.lora_alpha, dropout=0.0, adapter=adapter)
        bb.language_model.eval()  # see module docstring: cache + grads, no checkpointing
        modules = TraceV3Modules.load(init, cfg, bb.hidden) if init else TraceV3Modules(cfg, bb.hidden)
        modules = modules.to(device).train()
        if world > 1:  # identical start on every rank (no DDP broadcast)
            for p in modules.parameters():
                dist.broadcast(p.data, 0)
        lora_params = [p for p in bb.model.parameters() if p.requires_grad]
        # The alert head is not trained in Stage 2: frozen and out of the optimizer (no weight-decay drift).
        for p in modules.alert.parameters():
            p.requires_grad_(False)
        module_params = [p for p in modules.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params": module_params, "lr": tc.lr * s2.lr_scale, "weight_decay": tc.weight_decay},
             {"params": lora_params, "lr": tc.lora_lr * s2.lr_scale, "weight_decay": 0.0}],
        )  # fmt: skip
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        runner = Stage2Runner(bb, modules, cfg)
        output = Path(output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.json").write_text(json.dumps({"config": cfg.to_dict(), "streams": streams_path,
                                                         "init": init, "adapter": adapter, "world": world,
                                                         "modules": modules.parameter_report(),
                                                         "lora_params": sum(p.numel() for p in lora_params)},
                                                        indent=2) + "\n")  # fmt: skip
        tracker = Tracker.start(rank == 0 and enabled_from_env(),
                                name=os.environ.get("TRACE_WANDB_NAME") or f"{output.parent.name}-{output.name}",
                                group=os.environ.get("TRACE_WANDB_GROUP") or output.parent.parent.name,
                                job_type="train-stage2", config={"config": cfg.to_dict(), "streams": streams_path, "init": init, "world": world},
                                directory=output / "wandb")  # fmt: skip
        total_steps = max_steps or s2.steps
        warmup = min(tc.warmup, max(1, total_steps // 10))
        started = clock.perf_counter()
        for step in range(total_steps):
            factor = min(1.0, (step + 1) / warmup)
            progress = max(0.0, (step - warmup) / max(1, total_steps - warmup))
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * factor * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            optimizer.zero_grad(set_to_none=True)
            stats = [runner.run(random.choice(records), video_root, 1.0 / s2.grad_accum) for _ in range(s2.grad_accum)]
            all_reduce_grads(module_params + lora_params, world)
            retention = list(modules.retention.parameters())
            retention_grad = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in retention if p.grad is not None))
            retention_before = [p.detach().clone() for p in retention]
            norms = {}
            for name, group in zip(("modules", "lora"), optimizer.param_groups):
                params = [p for p in group["params"] if p.grad is not None]
                norms[name] = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if params else 0.0
            if not all(math.isfinite(v) for v in norms.values()):
                raise FloatingPointError(f"nonfinite gradient norm: {norms}")
            optimizer.step()
            retention_update = math.sqrt(sum(float((p.detach() - b).float().pow(2).sum())
                                             for p, b in zip(retention, retention_before)))  # fmt: skip
            if rank == 0:
                line = {"step": step + 1, **summarize(stats), "grad_norm_modules": norms["modules"],
                        "grad_norm_retention": retention_grad, "update_norm_retention": retention_update,
                        "grad_norm_lora": norms["lora"], "gate_abs_mean": float(modules.cst.gate.abs().mean()),
                        "lr": optimizer.param_groups[0]["lr"], "elapsed": round(clock.perf_counter() - started, 1),
                        "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9}  # fmt: skip
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(line) + "\n")
                print(json.dumps(line), flush=True)
                tracker.log(line, step=step + 1)
                if (step + 1) % tc.save_every == 0 or step + 1 == total_steps:
                    modules.save(output / "modules.pt", {"step": step + 1, "stage": 2})
                    bb.peft.save_pretrained(output / "lora")
                    if step + 1 == total_steps:  # only after both final saves returned
                        from .pilot import write_completion

                        identity = output / "train-identity.json"  # written by the runner before training
                        # Stage-2 artifact contract: only the final module/LoRA pair (no train.checkpoint_steps snapshots,
                        # which belong to Stage 1 and may be inherited by the config); the verifier uses the same ().
                        write_completion(output, total_steps, cfg.to_dict(),
                                         json.loads(identity.read_text()) if identity.is_file() else None,
                                         checkpoint_steps=())
        tracker.upload(f"{output.parent.name}-{output.name}-logs", "training-logs",
                       [output / "metrics.jsonl", output / "run.json"], base=output)  # fmt: skip
        return output
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
        tracker.finish()  # after the process group: a slow network at shutdown cannot stall other ranks


# ---------------------------------------------------------------------------- held-out long-stream evaluation
def delay_bucket(delay: float) -> str:
    return "0s" if delay <= 0 else "<60s" if delay < 60 else "60-120s" if delay < 120 else ">=120s"


def summarize_streams(rows: list[dict], replicates: int = 2000, seed: int = 7) -> dict:
    """Accuracy overall and per delay bucket, evidence kept, whole-stream bootstrap CI of accuracy."""
    import numpy as np

    if not rows:
        return {"questions": 0}
    by_stream: dict[str, list[int]] = {}
    for r in rows:
        by_stream.setdefault(r["stream"], []).append(r["correct"])
    streams = sorted(by_stream)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(replicates):
        pick = rng.integers(0, len(streams), len(streams))
        values = [c for i in pick for c in by_stream[streams[i]]]
        accs.append(sum(values) / len(values))
    buckets: dict[str, list[int]] = {}
    for r in rows:
        buckets.setdefault(delay_bucket(r["delay"]), []).append(r["correct"])
    kept = [r["evidence_kept"] for r in rows if r.get("evidence_kept") is not None]
    return {"questions": len(rows), "streams": len(streams), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "by_delay": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(buckets.items())},
            "evidence_kept": sum(kept) / len(kept) if kept else None,
            "kv_tokens_mean": sum(r["kv_tokens"] for r in rows) / len(rows), "run": rows[0].get("run")}  # fmt: skip


def evaluate_streams(cfg: V3Config, streams_path: str, video_root: str, output: str, mode: str = "v3",
                     checkpoint: str | None = None, adapter: str | None = None, policy: str | None = None,
                     device: str = "cuda", split: str = "val", limit: int | None = None, shard=(0, 1),
                     evict_every: int | None = None, evict_schedule: str = "every", audit: bool = False) -> dict:  # fmt: skip
    """Stream each held-out Stage 2 stream once with the inference session; ask its delayed questions at their
    times. `policy` overrides memory.policy at the same budget (tkm vs mass / fifo / window controls)."""
    from .backbone import Backbone
    from .data import write_jsonl
    from .harness import _modules, clean_partial, read_unit, run_fingerprint, write_unit
    from .session import StreamSession

    if policy is not None or evict_every is not None:
        memory = {**cfg.to_dict()["memory"]}
        memory.update({k: v for k, v in (("policy", policy), ("evict_every", evict_every)) if v is not None})
        cfg = V3Config.from_dict({**cfg.to_dict(), "memory": memory})
    if mode == "v3" and not checkpoint:
        raise ValueError("--mode v3 needs a --checkpoint")
    records = [r for r in read_jsonl(streams_path) if r.get("split") == split and r.get("questions")]
    if not records:
        raise ValueError(f"no streams with split == {split}")
    if split == "train":
        raise ValueError("evaluate held-out streams (val), not training streams")
    records = records[:limit] if limit else records
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if evict_schedule not in {"every", "train"}:
        raise ValueError("evict_schedule must be every (memory.evict_every) or train (Stage-2 chunk schedule)")
    fingerprint = run_fingerprint(cfg, streams_path, f"stage2-{mode}-{split}-evict-{evict_schedule}-audit-{int(audit)}", checkpoint, adapter,
                                  video_root, records)  # fmt: skip
    run_file = output / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()).get("id") != fingerprint["id"]:
        raise ValueError(f"{output} holds a different run; use a new --output")
    run_file.write_text(json.dumps(fingerprint, indent=2) + "\n")
    units = output / "streams"  # one atomically written file per completed stream (see harness.write_unit)
    if units.exists():
        clean_partial(units)
    done = {r["id"] for r in records if read_unit(units, r["id"], fingerprint["id"]) is not None}
    todo = [i for i, r in enumerate(records) if r["id"] not in done and i % shard[1] == shard[0]]
    bb = modules = None
    if todo:  # an assembly-only call (everything already evaluated) does not load the model
        bb = Backbone(cfg.backbone, cfg.stream, device)
        if adapter:
            bb.add_lora(0, 0, adapter=adapter)
            bb.model.eval()
        modules = _modules(cfg, bb, checkpoint, device) if mode == "v3" else None
    gap = cfg.stream.frame_gap
    for index, record in enumerate(records):
        if record["id"] in done or index % shard[1] != shard[0]:
            continue
        times = stream_times(record, cfg.stream.step_seconds, gap, cfg.stage2.max_seconds)
        if not times:
            continue
        due = {steps[-1]: qs for steps, qs in chunk_plan(times, record["questions"], len(times), gap) if qs}
        session = StreamSession(bb, cfg, modules, mode=mode)
        if evict_schedule == "train":
            session.evict_steps = eviction_steps(times, record["questions"], cfg.stage2.chunk_steps, gap)
        session.audit = audit
        reader = StepReader(record, video_root)
        rows, appended = [], []
        try:
            started = clock.perf_counter()
            for k, t in enumerate(times):
                result = session.step(reader.step(t, gap), t)
                appended.append((t, result.tokens))
                for q in due.get(k, []):
                    kept = evidence_kept(session.memory, appended, q.get("evidence"))
                    choice, _ = session.ask(q["question"], q["options"])
                    evidence_end = max((b for _, b in q.get("evidence") or []), default=None)
                    rows.append({"stream": record["id"], "id": q["id"], "t": q["time"], "delay": q.get("delay", 0.0),
                                 "evidence_age": None if evidence_end is None else round(float(q["time"]) - evidence_end, 3),
                                 "correct": int(choice == q["answer"]), "evidence_kept": kept,
                                 "kv_tokens": len(session.memory), "kv_mb": bb.kv_bytes(session.cache) / 2**20,
                                 "evicted_tokens": session.memory.evicted_tokens,
                                 "kv_appended_tokens": len(session.memory) + session.memory.evicted_tokens,
                                 "visual_tokens_streamed": sum(n for _, n in appended),
                                 **(session.audit_summary(float(q["time"])) if audit else {}),
                                 "run": fingerprint["id"]})  # fmt: skip
            for row in rows:
                row["stream_seconds_elapsed"] = round(clock.perf_counter() - started, 2)
                row["stream_steps"] = len(times)
        finally:
            reader.close()
        write_unit(units, record["id"], rows, fingerprint["id"])
        del session
        torch.cuda.empty_cache()
    rows, pending = [], 0
    for record in records:
        if not stream_times(record, cfg.stream.step_seconds, gap, cfg.stage2.max_seconds):
            continue
        unit = read_unit(units, record["id"], fingerprint["id"])
        if unit is None:
            if shard[1] == 1:
                raise RuntimeError(f"stream {record['id']} has no complete result")
            pending += 1  # another shard's stream: the last shard to finish assembles
            continue
        rows += unit
    if pending:
        return {"shard": list(shard), "pending_streams_other_shards": pending}
    target = output / "predictions.jsonl"  # assembled from complete streams only; rewritten each time
    write_jsonl(target, rows)
    summary = summarize_streams(read_jsonl(target))
    summary["policy"] = cfg.memory.policy
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
