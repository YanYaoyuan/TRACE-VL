"""Decode explicit, causal observations for core video-QA experiments.

Sampling is uniform over frame indices inside the declared observation, not over
the answer or question semantics. Presentation timestamps come from ffprobe;
ffmpeg decodes those exact frame indices. Both full/grid arms reuse these frames.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.v3.backbone import frame_size

from .qa import VisualStep, question_and_answer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_record(row: dict, root: Path) -> dict:
    """Accept normalized MC/text QA and the existing LLaVA text manifest."""
    result = dict(row)
    if not isinstance(result.get("id"), str) or not result["id"]:
        raise ValueError("record requires a nonempty id")
    path = Path(result["video"])
    result["video"] = str((root / path).resolve())
    if "options" not in result and "answer_text" not in result:
        if result.get("answer_type") == "text" and isinstance(result.get("answer"), str):
            result["answer_text"] = result["answer"]
    question_and_answer(result)
    return result


def read_split(path: Path, split: str) -> list[dict]:
    """Honor the saved split; reject cross-split videos and duplicate row IDs."""
    rows, ids, videos = [], set(), {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = normalize_record(json.loads(line), Path(path).parent)
        if row["id"] in ids:
            raise ValueError(f"duplicate record id: {row['id']}")
        ids.add(row["id"])
        part = row.get("split")
        if not isinstance(part, str) or not part:
            raise ValueError("every record requires a saved split")
        # source_group catches alternate paths/transcodes when the producer
        # supplies it; this is not a global benchmark contamination audit.
        for key in (row["video"], row.get("source_group", row["video"])):
            if key in videos and videos[key] != part:
                raise ValueError(f"video crosses splits: {key}")
            videos[key] = part
        if part == split:
            rows.append(row)
    if not rows:
        raise ValueError(f"no records for split {split}")
    return rows


def decode_observation(row: dict, *, steps: int, pixels: int, timeout: int = 180):
    if type(steps) is not int or steps < 1 or type(pixels) is not int or pixels < 1024:
        raise ValueError("positive steps and at least 1024 pixels required")
    video = Path(row["video"])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time",
         "-of", "json", str(video)], check=True, capture_output=True, timeout=timeout)
    metadata = json.loads(probe.stdout)
    frames, stream = metadata["frames"], metadata["streams"][0]
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in frames])
    if not len(pts) or not np.isfinite(pts).all() or (np.diff(pts) <= 0).any():
        raise ValueError("video requires finite strictly increasing presentation timestamps")
    origin = float(pts[0])
    times = pts - origin
    observation = row.get("observation", {})
    start = float(observation.get("start", 0.))
    end = observation.get("end", "video_end")
    end = float(times[-1]) if end == "video_end" else float(end)
    query = row.get("query_time")
    if query is None:
        if row.get("protocol") != "offline":
            raise ValueError("online examples require query_time")
        query = end
    query = float(query)
    if not all(math.isfinite(x) for x in (start, end, query)) or not 0 <= start <= end or query < start:
        raise ValueError("invalid observation/query bounds")
    indices = np.flatnonzero((times >= start) & (times <= min(end, query)))
    if len(indices) < 2 * steps:
        raise ValueError(f"need {2 * steps} distinct causal frames, found {len(indices)}")
    chosen = indices[np.linspace(0, len(indices) - 1, 2 * steps).round().astype(int)]
    h, w = frame_size(stream["height"], stream["width"], pixels, 32)
    expression = "+".join(f"eq(n\\,{int(i)})" for i in chosen)
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1", "-noautorotate",
         "-i", str(video), "-map", "0:v:0", "-vf", f"select={expression},scale={w}:{h}",
         "-vsync", "0", "-filter_threads", "1", "-threads", "1", "-pix_fmt", "rgb24",
         "-f", "rawvideo", "pipe:1"], check=True, capture_output=True, timeout=timeout)
    expected = 2 * steps * h * w * 3
    if len(decoded.stdout) != expected:
        raise ValueError(f"decoded byte count {len(decoded.stdout)} != expected {expected}")
    images = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(steps, 2, h, w, 3).copy()
    actual = times[chosen].reshape(steps, 2)
    resolved = {**row, "query_time": query}
    try:
        source_fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        source_fps = 0.
    audit = {"video_sha256": sha256(video), "pts_origin": origin,
             "frame_indices": chosen.tolist(), "frame_times": actual.tolist(),
             "frame_hw": [h, w], "query_time": query,
             "source_fps": source_fps, "source_frame_count": len(pts),
             "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
             "sampling": "uniform-frame-index-within-observation-v1"}
    return images, actual, resolved, audit


@torch.no_grad()
def encode_observation(backbone, images: np.ndarray, times: np.ndarray):
    result = []
    for frames, pair in zip(images, times, strict=True):
        pixels, grid = backbone.preprocess(frames)
        x = backbone.encode(pixels, grid)
        result.append(VisualStep(x.detach(), backbone.merged_grid(grid), float(pair[-1]),
                                 tuple(float(t) for t in pair)))
    return result
