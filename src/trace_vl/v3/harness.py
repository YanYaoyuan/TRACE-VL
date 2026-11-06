"""M0/M1 harness: streaming evaluation, scoring, profiling, and the backbone-equality check."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time as clock
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .backbone import Backbone
from .config import V3Config
from .data import StepReader, draw_region, read_jsonl, step_times
from .model import TraceV3Modules
from .session import StreamSession


def _modules(cfg: V3Config, backbone: Backbone, checkpoint: str | None, device):
    modules = TraceV3Modules.load(checkpoint, cfg, backbone.hidden) if checkpoint else TraceV3Modules(cfg, backbone.hidden)
    return modules.to(device).eval()


def run_fingerprint(cfg: V3Config, records_path: str, mode: str, checkpoint: str | None) -> dict:
    identity = {
        "config": cfg.to_dict(),
        "records_sha256": hashlib.sha256(Path(records_path).read_bytes()).hexdigest(),
        "mode": mode,
        "checkpoint": None,
    }
    if checkpoint:
        identity["checkpoint"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    identity["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return identity


def evaluate(cfg: V3Config, records_path: str, video_root: str, output: str, mode: str = "native",
             checkpoint: str | None = None, shard=(0, 1), device="cuda", limit: int | None = None,
             adapter: str | None = None) -> dict:  # fmt: skip
    """Stream each video once; answer its questions at their times on forked caches. Resumable."""
    records = read_jsonl(records_path)
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        key = r.get("stream") or r.get("video")
        groups[key].append(r)
    keys = sorted(groups)[shard[0] :: shard[1]]
    if limit:
        keys = keys[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(cfg, records_path, mode, checkpoint)
    fingerprint["adapter"] = str(Path(adapter).resolve()) if adapter else None
    run_file = output / "run.json"
    if run_file.exists() and json.loads(run_file.read_text())["id"] != fingerprint["id"]:
        raise ValueError(f"{output} holds a different run; use a new --output")
    run_file.write_text(json.dumps(fingerprint, indent=2) + "\n")
    target = output / f"predictions.{shard[0]}of{shard[1]}.jsonl"
    done = {json.loads(x)["id"] for x in target.read_text().splitlines() if x.strip()} if target.exists() else set()

    backbone = Backbone(cfg.backbone, cfg.stream, device)
    if adapter:
        backbone.add_lora(0, 0, adapter=adapter)
        backbone.model.eval()
    modules = _modules(cfg, backbone, checkpoint, device) if mode == "v3" else None
    errors = {}
    with target.open("a") as handle:
        for key in keys:
            pending = sorted((r for r in groups[key] if r["id"] not in done),
                             key=lambda r: math.inf if r["query_time"] is None else r["query_time"])  # fmt: skip
            if not pending:
                continue
            try:
                reader = StepReader(pending[0], video_root)
            except (FileNotFoundError, ValueError) as error:
                errors[key] = str(error)
                continue
            try:
                _run_video(cfg, backbone, modules, mode, reader, pending, handle)
            except (ValueError, RuntimeError) as error:
                errors[key] = str(error)
            finally:
                reader.close()
    (output / f"errors.{shard[0]}of{shard[1]}.json").write_text(json.dumps(errors, indent=2) + "\n")
    return {"run": fingerprint["id"], "videos": len(keys), "errors": len(errors), "predictions": str(target)}


def _run_video(cfg, backbone, modules, mode, reader, records, handle):
    gap, step_s = cfg.stream.frame_gap, cfg.stream.step_seconds
    offline = all(r["query_time"] is None for r in records)
    last_query = reader.duration if offline else max(r["query_time"] for r in records)
    times = step_times(min(last_query, reader.duration), step_s, gap, cfg.stream.max_steps if offline else 0)
    session = StreamSession(backbone, cfg, modules, mode)
    started, consumed, steps_tokens, i_steps = clock.perf_counter(), 0, 0, 0
    queue = list(records)
    for t in times:
        # Answer every question whose time has come before consuming frames beyond it (causality).
        while queue and queue[0]["query_time"] is not None and queue[0]["query_time"] < t + gap / 2:
            _answer(session, reader, queue.pop(0), handle, consumed, steps_tokens, i_steps, started, gap)
        result = session.step(reader.step(t, gap), t)
        consumed += 1
        steps_tokens += result.tokens
        i_steps += int(result.is_i_step)
    for record in queue:
        _answer(session, reader, record, handle, consumed, steps_tokens, i_steps, started, gap)


def _answer(session, reader, record, handle, consumed, steps_tokens, i_steps, started, gap):
    marker = None
    t_q = record["query_time"] if record["query_time"] is not None else reader.duration
    if record.get("region"):
        t_m = max(gap / 2, min(t_q, reader.duration) - gap / 2)
        marker = (draw_region(reader.step(t_m, gap), record["region"]), t_m)
    t0 = clock.perf_counter()
    choice, scores = session.ask(record["question"], record["options"], marker)
    stats = session.stats()
    handle.write(json.dumps({
        "id": record["id"], "source": record["source"], "task": record["task"], "answer": record["answer"],
        "prediction": choice, "scores": [round(s, 4) for s in scores], "query_time": record["query_time"],
        "steps": consumed, "lm_tokens_per_step": steps_tokens / max(1, consumed), "i_step_fraction": i_steps / max(1, consumed),
        "kv_tokens": stats["kv_tokens"], "kv_mb": stats["kv_bytes"] / 2**20, "state_mb": stats["state_bytes"] / 2**20,
        "answer_seconds": round(clock.perf_counter() - t0, 3), "elapsed_seconds": round(clock.perf_counter() - started, 1),
        "camera": record.get("camera"), "gap": record.get("gap"), "video_key": record.get("stream") or record.get("video"),
    }) + "\n")  # fmt: skip
    handle.flush()


def score(prediction_dir: str, output: str | None = None, samples: int = 2000, seed: int = 7) -> dict:
    rows = [json.loads(x) for p in sorted(Path(prediction_dir).glob("predictions.*.jsonl"))
            for x in p.read_text().splitlines() if x.strip()]  # fmt: skip
    by_task = defaultdict(list)
    for r in rows:
        by_task[(r["source"], r["task"])].append(r["prediction"] == r["answer"])
    groups = defaultdict(list)
    for r in rows:
        # Resample independent units: cameras for MEVA, whole videos otherwise (questions share video).
        groups[r.get("camera") or r["video_key"]].append(r)
    rng = random.Random(seed)
    keys = sorted(groups)
    boots = []
    for _ in range(samples if len(keys) > 1 else 0):
        pick = [x for k in rng.choices(keys, k=len(keys)) for x in groups[k]]
        boots.append(sum(x["prediction"] == x["answer"] for x in pick) / len(pick))
    boots.sort()
    report = {
        "n": len(rows),
        "accuracy": sum(r["prediction"] == r["answer"] for r in rows) / max(1, len(rows)),
        "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if boots else None,
        "by_task": {f"{s}/{t}": {"n": len(v), "accuracy": sum(v) / len(v)} for (s, t), v in sorted(by_task.items())},
        "lm_tokens_per_step": float(np.mean([r["lm_tokens_per_step"] for r in rows])) if rows else None,
        "kv_mb_mean": float(np.mean([r["kv_mb"] for r in rows])) if rows else None,
        "kv_mb_max": float(np.max([r["kv_mb"] for r in rows])) if rows else None,
        "answer_seconds_mean": float(np.mean([r["answer_seconds"] for r in rows])) if rows else None,
    }
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


@torch.no_grad()
def profile(cfg: V3Config, steps: int = 600, mode: str = "native", checkpoint: str | None = None, device="cuda",
            height: int = 448, width: int = 448, seed: int = 0) -> dict:  # fmt: skip
    """Latency and memory on synthetic frames (slowly drifting noise with occasional changes)."""
    rng = np.random.default_rng(seed)
    backbone = Backbone(cfg.backbone, cfg.stream, device)
    modules = _modules(cfg, backbone, checkpoint, device) if mode == "v3" else None
    session = StreamSession(backbone, cfg, modules, mode)
    base = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    lm, vision, tokens, curve = [], [], [], []
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    for k in range(steps):
        if k % 30 == 0:
            y, x = rng.integers(0, height - 64), rng.integers(0, width - 64)
            base[y : y + 64, x : x + 64] = rng.integers(0, 255, 3, dtype=np.uint8)
        frames = np.stack([base, base])
        result = session.step(frames, k * cfg.stream.step_seconds + cfg.stream.frame_gap / 2)
        lm.append(result.seconds_lm)
        vision.append(result.seconds_vision)
        tokens.append(result.tokens)
        if k % 60 == 59:
            stats = session.stats()
            curve.append({"step": k + 1, "kv_tokens": stats["kv_tokens"], "kv_mb": stats["kv_bytes"] / 2**20})
    stats = session.stats()
    return {
        "mode": mode, "policy": cfg.memory.policy, "steps": steps,
        "lm_ms_per_step": 1000 * float(np.mean(lm[5:])), "vision_ms_per_step": 1000 * float(np.mean(vision[5:])),
        "tokens_per_step": float(np.mean(tokens)), "kv_curve": curve,
        "state_mb": stats["state_bytes"] / 2**20, "kv_kinds": stats["kv_kinds"],
        "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else None,
    }  # fmt: skip


@torch.no_grad()
def check_equality(cfg: V3Config, video: str | None = None, steps: int = 4, device="cuda") -> dict:
    """M1 gate (a): our incremental streaming path reproduces the reference HF forward, and the zero-gated
    v3 path (all I-steps) reproduces the native path."""
    backbone = Backbone(cfg.backbone, cfg.stream, device)
    rng = np.random.default_rng(0)
    if video:
        reader = StepReader({"video": Path(video).name}, Path(video).parent)
        clips = [reader.step(t, cfg.stream.frame_gap) for t in step_times(steps, 1.0, cfg.stream.frame_gap)[:steps]]
        reader.close()
    else:
        clips = [rng.integers(0, 255, (2, 448, 448, 3), dtype=np.uint8) for _ in range(steps)]
    times = [k + cfg.stream.frame_gap / 2 for k in range(steps)]
    question, options = "What is shown in the video?", ["a person", "a car", "an animal", "nothing"]

    native = StreamSession(backbone, cfg, None, "native")
    for frames, t in zip(clips, times):
        native.step(frames, t)
    _, native_scores = native.ask(question, options)

    cfg_eq = V3Config.from_dict({**cfg.to_dict(), "transition": {**cfg.to_dict()["transition"], "i_step_every": 1}})
    modules = TraceV3Modules(cfg_eq, backbone.hidden).to(device).eval()
    v3 = StreamSession(backbone, cfg_eq, modules, "v3")
    for frames, t in zip(clips, times):
        v3.step(frames, t)
    _, v3_scores = v3.ask(question, options)

    # Reference: one HF forward over the same token sequence; positions come from get_rope_index.
    ids, pixels, grids = backbone.ids(backbone.head_text), [], []
    types = [0] * len(ids)
    for frames, t in zip(clips, times):
        pv, grid = backbone.preprocess(native._resize(frames))
        h, w = backbone.merged_grid(grid)
        ts = backbone.ids(f"<{t:.1f} seconds>") + [backbone.vision_start]
        ids += ts + [backbone.video_token] * (h * w) + [backbone.vision_end]
        types += [0] * len(ts) + [2] * (h * w) + [0]
        pixels.append(pv)
        grids.append(grid)
    lines = [f"{'ABCD'[i]}. {o}" for i, o in enumerate(options)]
    tail = backbone.ids("\n" + question + "\n" + "\n".join(lines) + "\nAnswer with the option letter only." + backbone.tail_text)
    ids += tail
    types += [0] * len(tail)
    backbone.model.model.rope_deltas = None
    out = backbone.model(
        input_ids=torch.tensor([ids], device=backbone.device),
        mm_token_type_ids=torch.tensor([types], device=backbone.device),
        pixel_values_videos=torch.cat(pixels).to(backbone.device),
        video_grid_thw=torch.cat(grids).to(backbone.device),
        use_cache=False,
    )
    ref = out.logits[0, -1].float().log_softmax(-1)[native.letter_ids[: len(options)]].tolist()
    diff_ref = max(abs(a - b) for a, b in zip(ref, native_scores))
    diff_v3 = max(abs(a - b) for a, b in zip(v3_scores, native_scores))
    return {
        "reference_scores": ref, "native_scores": native_scores, "v3_zero_gate_scores": v3_scores,
        "max_abs_diff_native_vs_reference": diff_ref, "max_abs_diff_v3_vs_native": diff_v3,
        "pass": diff_ref < 5e-2 and diff_v3 < 1e-3, "modules": modules.parameter_report(),
        "note": "bf16 incremental vs one-shot prefill can differ slightly; v3 vs native must be ~exact.",
    }  # fmt: skip
