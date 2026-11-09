"""Qualitative demos: what each arm sees and answers, for the same validation questions across arms.

For each selected question the video is streamed through the inference session (native or v3), and per step
the second frame is rendered with a heat overlay of where the LM-facing P-step tokens pool from (transition /
after: gist + slot weights; b4: grid regions; I-steps and native: all tokens, no overlay). The clip is written
as an animated GIF plus a keyframe strip; the answer, gold label and option probabilities are recorded.

The question selection depends only on the records and the seed, so every arm renders the same questions.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from .data import StepReader, read_jsonl, step_times


def select_questions(records: list[dict], n: int, seed: int = 7, split: str = "val") -> list[dict]:
    pool = sorted((r for r in records if r.get("split", split) == split and r.get("kind", "mcq") == "mcq"
                   and r.get("options")), key=lambda r: r["id"])  # fmt: skip
    return random.Random(seed).sample(pool, min(n, len(pool)))


def overlay(frame: np.ndarray, weights, grid: tuple[int, int] | None, label: str, width: int = 320) -> np.ndarray:
    """Frame [H, W, 3] uint8 with a red heat overlay of a [h*w] weight map (relative to its maximum) and a label."""
    import cv2

    h0, w0 = frame.shape[:2]
    height = max(1, round(h0 * width / w0))
    out = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    if weights is not None and grid is not None:
        heat = np.asarray(weights, dtype=np.float32).reshape(grid)
        heat = heat / max(float(heat.max()), 1e-12)
        heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_NEAREST)[..., None]
        red = np.zeros_like(out)
        red[..., 0] = 255.0
        out = out * (1 - 0.55 * heat) + red * 0.55 * heat
    out = out.clip(0, 255).astype(np.uint8)
    cv2.rectangle(out, (0, 0), (width, 18), (0, 0, 0), -1)
    cv2.putText(out, label[:48], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_gif(frames: list[np.ndarray], path: Path, max_frames: int = 40, ms: int = 500):
    from PIL import Image

    if len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=ms, loop=0)


def write_strip(frames: list[np.ndarray], path: Path, count: int = 6):
    from PIL import Image

    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)]
    Image.fromarray(np.concatenate(picks, axis=1)).save(path)


def make_demos(cfg, records_path: str, video_root: str, output: str, mode: str = "v3", checkpoint: str | None = None,
               adapter: str | None = None, n: int = 12, device: str = "cuda", seed: int = 7) -> list[dict]:  # fmt: skip
    import torch

    from .backbone import Backbone
    from .harness import _modules
    from .session import StreamSession

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    chosen = select_questions(read_jsonl(records_path), n, seed)
    bb = Backbone(cfg.backbone, cfg.stream, device)
    if adapter:
        bb.add_lora(0, 0, adapter=adapter)
        bb.model.eval()
    modules = _modules(cfg, bb, checkpoint, device) if mode == "v3" else None
    if modules is not None:
        modules.transition.diagnostics = True  # weight maps only; outputs are unchanged (tested)
    gap, unit = cfg.stream.frame_gap, cfg.stream.unit
    rows = []
    for record in chosen:
        reader = StepReader(record, video_root)
        session = StreamSession(bb, cfg, modules, mode=mode)
        frames, tokens = [], []
        try:
            end = min(record["query_time"] if record.get("query_time") is not None else reader.duration, reader.duration)
            for t in [t for t in step_times(end, cfg.stream.step_seconds, gap) if t + gap / 2 <= end + 1e-6]:
                clip = reader.step(t, gap)
                result = session.step(clip, t)
                tokens.append(result.tokens)
                resized = bb.resize_step(clip)
                grid = (resized.shape[1] // unit, resized.shape[2] // unit)
                info = session.last_info if mode == "v3" else None
                weights = None
                if info is not None and not info.is_i_step and info.aux.get("weight_map") is not None:
                    weights = info.aux["weight_map"].numpy()
                    if weights.size != grid[0] * grid[1]:
                        weights = None
                kind = "I" if result.is_i_step else "P"
                frames.append(overlay(clip[-1], weights, grid, f"t={t:.1f}s {kind}-step {result.tokens} tok"))
            with torch.no_grad():
                choice, scores = session.ask(record["question"], record["options"])
        finally:
            reader.close()
        top = max(scores)
        probs = [math.exp(s - top) for s in scores]
        probs = [p / sum(probs) for p in probs]
        stem = record["id"].replace("/", "_")
        gif, strip = output / f"{stem}.gif", output / f"{stem}.png"
        write_gif(frames, gif)
        write_strip(frames, strip)
        rows.append({"id": record["id"], "video": record["video"], "question": record["question"],
                     "options": record["options"], "gold": record["answer"], "prediction": int(choice),
                     "correct": int(choice) == record["answer"], "probabilities": [round(p, 4) for p in probs],
                     "steps": len(tokens), "tokens_per_step": sum(tokens) / max(1, len(tokens)),
                     "gif": str(gif), "strip": str(strip)})  # fmt: skip
        del session
    (output / "demos.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def log_demos(tracker, rows: list[dict], arm: str):
    """One wandb table per arm: animated clip, keyframe strip, question, options, gold, prediction, probabilities."""
    if not tracker.active or not rows:
        return
    import wandb

    letters = "ABCDEFGHIJ"
    table = wandb.Table(columns=["arm", "id", "clip", "keyframes", "question", "options", "gold", "prediction",
                                 "correct", "probabilities", "tokens_per_step"])  # fmt: skip
    for r in rows:
        options = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(r["options"]))
        table.add_data(arm, r["id"], wandb.Video(r["gif"]), wandb.Image(r["strip"]), r["question"], options,
                       letters[r["gold"]], letters[r["prediction"]], r["correct"], json.dumps(r["probabilities"]),
                       round(r["tokens_per_step"], 2))  # fmt: skip
    tracker.log_media({f"demos/{arm}": table})
