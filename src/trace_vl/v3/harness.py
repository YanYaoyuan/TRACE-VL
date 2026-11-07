"""M0/M1 harness: streaming evaluation, scoring, profiling, and the backbone-equality check."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time as clock
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .backbone import Backbone
from .config import V3Config
from .data import StepReader, draw_region, read_jsonl, step_times
from .model import TraceV3Modules
from .modules import CSTCache, TECache
from .session import StreamSession


def _modules(cfg: V3Config, backbone: Backbone, checkpoint: str | None, device):
    modules = TraceV3Modules.load(checkpoint, cfg, backbone.hidden) if checkpoint else TraceV3Modules(cfg, backbone.hidden)
    return modules.to(device).eval()


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dir_identity(path: str | None) -> dict | None:
    if not path:
        return None
    root = Path(path)
    files = sorted(f for f in root.rglob("*") if f.is_file()) if root.is_dir() else [root]
    return {str(f.relative_to(root) if root.is_dir() else f.name): _sha256(f) for f in files}


def _video_identity(records: list[dict], video_root: str) -> str:
    """Cheap identity of every referenced video: relative path, size and modification time."""
    entries = []
    for r in records:
        for clip in r.get("clips") or [{"video": r["video"]}]:
            f = Path(video_root) / clip["video"]
            stat = f.stat() if f.exists() else None
            entries.append((clip["video"], stat.st_size if stat else None, int(stat.st_mtime) if stat else None))
    return hashlib.sha256(json.dumps(sorted(set(entries))).encode()).hexdigest()


def run_fingerprint(cfg: V3Config, records_path: str, mode: str, checkpoint: str | None = None,
                    adapter: str | None = None, video_root: str | None = None, records: list | None = None) -> dict:  # fmt: skip
    """Everything that changes a prediction, hashed before the id is derived (PR #3 review)."""
    identity = {
        "config": cfg.to_dict(),
        "records_sha256": _sha256(records_path),
        "mode": mode,
        "checkpoint": _sha256(checkpoint) if checkpoint else None,
        "adapter": _dir_identity(adapter),
        "videos": _video_identity(records, video_root) if records is not None and video_root else None,
    }
    identity["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return identity


def evaluate(cfg: V3Config, records_path: str, video_root: str, output: str, mode: str = "native",
             checkpoint: str | None = None, shard=(0, 1), device="cuda", limit: int | None = None,
             adapter: str | None = None) -> dict:  # fmt: skip
    """Stream each video once; answer its questions at their times on forked caches. Resumable."""
    records = read_jsonl(records_path)
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        # Resume skips by id: a duplicated id would silently skip unprocessed sibling questions.
        dup = sorted(i for i, n in Counter(ids).items() if n > 1)[:5]
        raise ValueError(f"{records_path} has duplicate record ids (e.g. {dup}); regenerate with unique ids")
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        key = r.get("stream") or r.get("video")
        groups[key].append(r)
    keys = sorted(groups)[shard[0] :: shard[1]]
    if limit:
        keys = keys[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(cfg, records_path, mode, checkpoint, adapter, video_root, records)
    run_file = output / "run.json"
    # Validate before touching anything: never relabel an existing run's predictions.
    if run_file.exists():
        previous = json.loads(run_file.read_text())
        if previous.get("id") != fingerprint["id"]:
            changed = sorted(k for k in fingerprint if k != "id" and previous.get(k) != fingerprint[k])
            raise ValueError(f"{output} holds a different run (differs in {changed}); use a new --output")
    target = output / f"predictions.{shard[0]}of{shard[1]}.jsonl"
    done = set()
    if target.exists():
        for line in target.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run") != fingerprint["id"]:
                raise ValueError(f"{target} has predictions from run {row.get('run')}; use a new --output")
            done.add(row["id"])
    if not run_file.exists():
        run_file.write_text(json.dumps(fingerprint, indent=2) + "\n")

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
                _run_video(cfg, backbone, modules, mode, reader, pending, handle, fingerprint["id"])
            except (ValueError, RuntimeError) as error:
                errors[key] = str(error)
            finally:
                reader.close()
    (output / f"errors.{shard[0]}of{shard[1]}.json").write_text(json.dumps(errors, indent=2) + "\n")
    return {"run": fingerprint["id"], "videos": len(keys), "errors": len(errors), "predictions": str(target)}


def _run_video(cfg, backbone, modules, mode, reader, records, handle, run_id):
    gap, step_s = cfg.stream.frame_gap, cfg.stream.step_seconds
    offline = all(r["query_time"] is None for r in records)
    last_query = reader.duration if offline else max(r["query_time"] for r in records)
    times = step_times(min(last_query, reader.duration), step_s, gap, cfg.stream.max_steps if offline else 0)
    session = StreamSession(backbone, cfg, modules, mode)
    started, consumed, steps_tokens, i_steps = clock.perf_counter(), 0, 0, 0
    p_tokens = 0
    queue = list(records)
    for t in times:
        # Answer every question whose time has come before consuming frames beyond it (causality).
        while queue and queue[0]["query_time"] is not None and queue[0]["query_time"] < t + gap / 2:
            _answer(session, reader, queue.pop(0), handle, consumed, steps_tokens, i_steps, started, gap, run_id)
        result = session.step(reader.step(t, gap), t)
        consumed += 1
        steps_tokens += result.tokens
        i_steps += int(result.is_i_step)
        p_tokens += 0 if result.is_i_step else result.tokens
        session.p_step_tokens = p_tokens  # running total, read by _answer for the budget fields
    for record in queue:
        _answer(session, reader, record, handle, consumed, steps_tokens, i_steps, started, gap, run_id)


def _answer(session, reader, record, handle, consumed, steps_tokens, i_steps, started, gap, run_id):
    marker = None
    t_q = record["query_time"] if record["query_time"] is not None else reader.duration
    if record.get("region"):
        t_m = max(gap / 2, min(t_q, reader.duration) - gap / 2)
        marker = (draw_region(reader.step(t_m, gap), record["region"]), t_m)
    t0 = clock.perf_counter()
    choice, scores = session.ask(record["question"], record["options"], marker)
    stats = session.stats()
    handle.write(json.dumps({
        "run": run_id, "id": record["id"], "source": record["source"], "task": record["task"], "answer": record["answer"],
        "prediction": choice, "scores": [round(s, 4) for s in scores], "query_time": record["query_time"],
        "steps": consumed, "lm_tokens_per_step": steps_tokens / max(1, consumed), "i_step_fraction": i_steps / max(1, consumed),
        "p_step_tokens_mean": getattr(session, "p_step_tokens", 0) / max(1, consumed - i_steps) if consumed > i_steps else None,
        "total_visual_tokens": steps_tokens,
        "kv_tokens": stats["kv_tokens"], "kv_mb": stats["kv_bytes"] / 2**20, "state_mb": stats["state_bytes"] / 2**20,
        "answer_seconds": round(clock.perf_counter() - t0, 3), "elapsed_seconds": round(clock.perf_counter() - started, 1),
        "camera": record.get("camera"), "gap": record.get("gap"), "video_key": record.get("stream") or record.get("video"),
    }) + "\n")  # fmt: skip
    handle.flush()


def score(prediction_dir: str, output: str | None = None, samples: int = 2000, seed: int = 7) -> dict:
    rows = [json.loads(x) for p in sorted(Path(prediction_dir).glob("predictions.*.jsonl"))
            for x in p.read_text().splitlines() if x.strip()]  # fmt: skip
    runs = {r.get("run") for r in rows}
    if len(runs) > 1:
        raise ValueError(f"{prediction_dir} mixes predictions from runs {sorted(map(str, runs))}")
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
        "run": next(iter(runs)) if rows else None,
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
def check_equality(cfg: V3Config, video: str | None = None, steps: int = 4, device="cuda",
                   dtype: str = "float32") -> dict:  # fmt: skip
    """M1/M0 gate: the streaming path reproduces Qwen3.5's PUBLIC processor protocol.

    The reference starts from `processor.apply_chat_template` with a video item (as an official pipeline
    would), not from a hand-built layout (PR #3 review: the old gate built both sides from the same
    layout and missed the processor's outer <|vision_start|> ... <|vision_end|> wrapper). Our native and
    zero-gated v3 sessions consume the processor's own per-temporal-patch tensors, and we compare
    (1) the exact token ids, (2) letter log-probs vs the full HF forward. float32 is the gate; bf16
    differs by ~0.1 from rounding alone and is informational.
    """
    from transformers.video_utils import VideoMetadata, read_video_opencv

    cfg = V3Config.from_dict({**cfg.to_dict(), "backbone": {**cfg.to_dict()["backbone"], "dtype": dtype}})
    backbone = Backbone(cfg.backbone, cfg.stream, device)
    processor, vp = backbone.processor, backbone.processor.video_processor
    if video:
        frames, meta = read_video_opencv(video, sample_indices_fn=vp.sample_frames)  # processor defaults
    else:
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (2 * steps, 448, 448, 3), dtype=np.uint8)
        meta = VideoMetadata(total_num_frames=2 * steps, fps=2.0, width=448, height=448, duration=float(steps),
                             video_backend="opencv", frames_indices=list(range(2 * steps)))  # fmt: skip
    timestamps = processor._calculate_timestamps(meta.frames_indices, meta.fps, vp.temporal_patch_size)
    question, options = "What is shown in the video?", ["a person", "a car", "an animal", "nothing"]
    lines = [f"{'ABCD'[i]}. {o}" for i, o in enumerate(options)]
    text = "\n" + question + "\n" + "\n".join(lines) + "\nAnswer with the option letter only."
    messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                           add_generation_prompt=True, enable_thinking=cfg.backbone.thinking,
                                           video_metadata=[meta], do_sample_frames=False)  # fmt: skip
    grid = inputs["video_grid_thw"][0]
    nt, gh, gw = (int(v) for v in grid)
    pixels, per_patch = inputs["pixel_values_videos"], gh * gw
    patch_grid = grid.clone()[None]
    patch_grid[0, 0] = 1

    def run(session):
        for k, t in enumerate(timestamps[:nt]):
            session.step(None, float(t), pixels=(pixels[k * per_patch : (k + 1) * per_patch], patch_grid))
        return session.ask(question, options)[1]

    native = StreamSession(backbone, cfg, None, "native", trace=True)
    native_scores = run(native)
    ours = native.trace_ids + native.trace_query
    ref_ids = inputs["input_ids"][0].tolist()
    first_mismatch = next((i for i, (a, b) in enumerate(zip(ours, ref_ids)) if a != b),
                          None if len(ours) == len(ref_ids) else min(len(ours), len(ref_ids)))  # fmt: skip

    cfg_eq = V3Config.from_dict({**cfg.to_dict(), "transition": {**cfg.to_dict()["transition"], "i_step_every": 1}})
    modules = TraceV3Modules(cfg_eq, backbone.hidden).to(device, getattr(torch, dtype)).eval()
    v3_scores = run(StreamSession(backbone, cfg_eq, modules, "v3"))

    backbone.model.model.rope_deltas = None
    data = {k: v.to(backbone.device) for k, v in inputs.items() if torch.is_tensor(v)}
    data["pixel_values_videos"] = data["pixel_values_videos"].to(backbone.dtype)
    # Same scoring on both sides: fp32 letter rows on the public forward's final hidden state.
    last = backbone.model.model(**data, use_cache=False).last_hidden_state[0, -1]
    ref = backbone.letter_logprobs(last, native.letter_ids[: len(options)]).tolist()
    diff_ref = max(abs(a - b) for a, b in zip(ref, native_scores))
    diff_v3 = max(abs(a - b) for a, b in zip(v3_scores, native_scores))
    gate = dtype == "float32"
    tokens_equal = first_mismatch is None
    return {
        "dtype": dtype, "video": video, "temporal_patches": nt, "grid": [nt, gh, gw],
        "tokens_equal": tokens_equal, "token_lengths": [len(ours), len(ref_ids)], "first_token_mismatch": first_mismatch,
        "mismatch_context": None if tokens_equal else {
            "ours": backbone.tokenizer.decode(ours[max(0, first_mismatch - 3) : first_mismatch + 3]),
            "processor": backbone.tokenizer.decode(ref_ids[max(0, first_mismatch - 3) : first_mismatch + 3])},
        "reference_scores": ref, "native_scores": native_scores, "v3_zero_gate_scores": v3_scores,
        "max_abs_diff_native_vs_reference": diff_ref, "max_abs_diff_v3_vs_native": diff_v3,
        "thresholds": {"native_vs_reference": 1e-3 if gate else 0.25, "v3_vs_native": 1e-5 if gate else 1e-3},
        "pass": tokens_equal and diff_ref < (1e-3 if gate else 0.25) and diff_v3 < (1e-5 if gate else 1e-3),
        "is_gate": gate, "modules": modules.parameter_report(),
    }  # fmt: skip


@torch.no_grad()
def budget_audit(arms: list[tuple[str, V3Config, str | None]], records_path: str, video_root: str, device="cuda",
                 max_videos: int = 8, max_seconds: float = 180.0, tolerance: float = 0.05) -> dict:  # fmt: skip
    """Visual-token budget per arm on fixed calibration videos (no LM, no accuracy): the matched-budget gate
    for transition vs B4 (PR #3 review). Calibration records must not come from a test split."""
    records = read_jsonl(records_path)
    if any(r.get("split") == "test" for r in records):
        raise ValueError("budget calibration must not use test-split records")
    videos, seen = [], set()
    for r in records:
        key = r.get("stream") or r.get("video")
        if key not in seen:
            seen.add(key)
            videos.append(r)
    videos = sorted(videos, key=lambda r: r.get("stream") or r.get("video"))[:max_videos]
    backbone = Backbone(arms[0][1].backbone, arms[0][1].stream, device)
    report = {"videos": [r.get("stream") or r.get("video") for r in videos], "arms": {}}
    for name, cfg, checkpoint in arms:
        modules = _modules(cfg, backbone, checkpoint, device)
        dtype = next(modules.parameters()).dtype
        p_tokens, i_tokens, steps = [], [], 0
        for record in videos:
            reader = StepReader(record, video_root)
            try:
                times = step_times(min(reader.duration, max_seconds), cfg.stream.step_seconds, cfg.stream.frame_gap)
                cst, te = CSTCache(), TECache()
                for t in times:
                    pv, grid = backbone.preprocess(backbone.resize_step(reader.step(t, cfg.stream.frame_gap)))
                    x = backbone.encode(pv, grid)
                    h, w = backbone.merged_grid(grid)
                    adapted, z = modules.cst(x.to(dtype), (h, w), t, cst)
                    info = modules.transition(adapted, z, (h, w), te)
                    (i_tokens if info.is_i_step else p_tokens).append(info.tokens.shape[0])
                    steps += 1
            finally:
                reader.close()
        total = sum(p_tokens) + sum(i_tokens)
        report["arms"][name] = {
            "checkpoint": checkpoint, "p_step_mode": cfg.transition.p_step_mode, "steps": steps,
            "i_step_fraction": len(i_tokens) / max(1, steps),
            "p_step_tokens_mean": float(np.mean(p_tokens)) if p_tokens else None,
            "i_step_tokens_mean": float(np.mean(i_tokens)) if i_tokens else None,
            "total_visual_tokens": total, "tokens_per_step": total / max(1, steps),
        }  # fmt: skip
    names = [a[0] for a in arms]
    if len(names) == 2:
        a, b = (report["arms"][n]["total_visual_tokens"] for n in names)
        gap = abs(b - a) / max(1, a)
        report["budget_gap"] = gap
        report["matched"] = gap <= tolerance
        report["tolerance"] = tolerance
    return report
