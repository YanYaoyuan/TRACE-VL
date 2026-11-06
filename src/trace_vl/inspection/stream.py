"""Single pass over each camera stream: detect transitions and snapshot evidence at query times.

One decode per stream serves every method and budget. For each question the snapshot keeps:
the query frame, `max_frames` uniformly spaced frames over the question window, the `max_frames`
most recent frames, and the transitions known at the query time (crops are stored once per stream).
Uniform frames, recent frames and transitions are exactly what each method may use, so later
packing under any budget never touches video again and never sees the future.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from .change import ChangeDetector, DetectorConfig
from .memory import TransitionStore


@dataclass
class SnapshotConfig:
    sample_fps: float = 2.0  # detector rate
    buffer_fps: float = 1.0  # frames kept for frame-based baselines
    buffer_long_side: int = 640
    query_long_side: int = 1280
    max_window: float = 1800.0
    max_frames: int = 16
    max_transitions: int = 32
    capacity: int = 128
    caption_every: float = 10.0
    jpeg_quality: int = 90
    expected_fps: float | None = 30.0  # annotations are in frames; refuse videos at another rate

    @classmethod
    def from_dict(cls, values: dict | None) -> "SnapshotConfig":
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown snapshot settings: {sorted(unknown)}")
        return cls(**values)


def resize_long(image: np.ndarray, side: int) -> np.ndarray:
    ratio = side / max(image.shape[:2])
    if ratio >= 1:
        return image
    return cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode frame")
    return data.tobytes()


def write_jpeg(path: Path, rgb: np.ndarray, quality: int = 90):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_jpeg(rgb, quality))


def read_stream(clips, video_root, sample_fps, expected_fps=None):
    """Yield (stream_time, RGB frame) at `sample_fps` over consecutive clips with known offsets."""
    last_t, next_sample = -math.inf, -math.inf
    for clip in clips:
        path = Path(video_root) / clip["video"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open {path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError(f"invalid fps in {path}")
            if expected_fps and abs(fps - expected_fps) > 0.05:
                raise ValueError(f"{path} has {fps:.3f} fps; annotations assume {expected_fps}")
            index = 0
            while capture.grab():
                t = clip["offset"] + index / fps
                index += 1
                if t + 1e-6 < next_sample or t <= last_t:
                    continue
                ok, frame = capture.retrieve()
                if not ok:
                    continue
                last_t = t
                next_sample = max(next_sample + 1 / sample_fps, t + 0.5 / sample_fps)
                yield t, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()


def _transition_record(view, stream_dir: Path, written: set, quality: int):
    crops = {}
    for name in ("before", "after", "after_end"):
        image = getattr(view, name)
        if image is None:
            continue
        path = stream_dir / "transitions" / f"{view.id:06d}_{name}.jpg"
        if (view.id, name) not in written:
            write_jpeg(path, image, quality)
            written.add((view.id, name))
        crops[name] = str(path.relative_to(stream_dir.parent.parent))
    return {
        "id": view.id,
        "kind": view.kind,
        "box": [round(v, 5) for v in view.box],
        "t_start": round(view.t_start, 3),
        "t_known": round(view.t_known, 3),
        "t_end": None if view.t_end is None else round(view.t_end, 3),
        "t_end_known": None if view.t_end_known is None else round(view.t_end_known, 3),
        "status": view.status,
        "crops": crops,
    }


def prepare_stream(stream_id, spec, questions, video_root, output, detector_cfg, snapshot_cfg):
    """Run one stream; write per-question snapshots, the transition log and caption frames."""
    cfg = snapshot_cfg
    output = Path(output)
    stream_dir = output / "streams" / stream_id
    stream_dir.mkdir(parents=True, exist_ok=True)
    pending = sorted(questions, key=lambda q: q["query_time"])
    detector = ChangeDetector(detector_cfg)
    store = TransitionStore(cfg.capacity)
    buffer = deque(maxlen=int(cfg.max_window * cfg.buffer_fps) + cfg.max_frames + 2)
    next_buffer, next_caption = -math.inf, 0.0
    written, captions, log = set(), [], []
    previous = None  # (t, frame) of the last processed frame

    def snapshot(question, t_frame, frame):
        qdir = output / "snapshots" / question["id"]
        qdir.mkdir(parents=True, exist_ok=True)
        tq, window = question["query_time"], question["window"]
        write_jpeg(qdir / "current.jpg", resize_long(frame, cfg.query_long_side), cfg.jpeg_quality)
        frames = {}
        in_window = [item for item in buffer if tq - window <= item[0] <= tq]
        if in_window:
            count = min(cfg.max_frames, len(in_window))
            picks = np.linspace(0, len(in_window) - 1, count).round().astype(int)
            frames["uniform"] = [in_window[i] for i in sorted(set(picks.tolist()))]
        frames["recent"] = list(buffer)[-cfg.max_frames :]
        listed = {}
        for name, items in frames.items():
            listed[name] = []
            for i, (t, data) in enumerate(items):
                path = qdir / f"{name}_{i:02d}.jpg"
                path.write_bytes(data)
                listed[name].append({"t": round(t, 3), "image": str(path.relative_to(output))})
        known = store.snapshot(tq)
        ranked = store.rank(known, tq, question.get("region"), window)[: cfg.max_transitions]
        record = {
            "id": question["id"],
            "stream": stream_id,
            "query_time": tq,
            "frame_time": round(t_frame, 3),
            "window": window,
            "region": question.get("region"),
            "current": str((qdir / "current.jpg").relative_to(output)),
            "uniform": listed.get("uniform", []),
            "recent": listed.get("recent", []),
            "transitions": [_transition_record(v, stream_dir, written, cfg.jpeg_quality) for v in ranked],
            "transitions_known": len(known),
            "transitions_evicted": store.evicted,
        }
        (qdir / "snapshot.json").write_text(json.dumps(record) + "\n")

    expected_size = tuple(spec["frame_size"]) if spec.get("frame_size") else None
    for t, frame in read_stream(spec["clips"], video_root, cfg.sample_fps, cfg.expected_fps):
        if expected_size and (frame.shape[1], frame.shape[0]) != expected_size:
            raise ValueError(f"frame size {frame.shape[1]}x{frame.shape[0]} differs from {expected_size}")
        # Questions are answered with the last frame at or before their time, never a later one.
        while pending and pending[0]["query_time"] < t and previous is not None:
            snapshot(pending.pop(0), *previous)
        events = detector.step(frame, t)
        store.update(events)
        for event, transition in events:
            log.append(
                {
                    "event": event,
                    "time": round(t, 3),
                    **_transition_record(transition, stream_dir, written, cfg.jpeg_quality),
                }
            )
        if t >= next_buffer:
            buffer.append((t, encode_jpeg(resize_long(frame, cfg.buffer_long_side), cfg.jpeg_quality)))
            next_buffer = t + 1 / cfg.buffer_fps - 1e-6
        if t >= next_caption:
            path = stream_dir / "captions" / f"{t:09.2f}.jpg"
            write_jpeg(path, resize_long(frame, cfg.buffer_long_side), cfg.jpeg_quality)
            captions.append({"t": round(t, 3), "image": str(path.relative_to(output))})
            next_caption = t + cfg.caption_every - 1e-6
        previous = (t, frame)
    while pending and previous is not None and pending[0]["query_time"] <= previous[0] + 1.0:
        snapshot(pending.pop(0), *previous)
    (stream_dir / "transitions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    (stream_dir / "caption_frames.json").write_text(json.dumps(captions) + "\n")
    missed = [q["id"] for q in pending]
    return {"stream": stream_id, "snapshots": len(questions) - len(missed), "missed": missed}


def _run(args):
    try:
        return prepare_stream(*args)
    except (OSError, ValueError) as error:
        # One unreadable clip (missing download, IR camera at another size) must not stop the rest.
        return {"stream": args[0], "snapshots": 0, "missed": [q["id"] for q in args[2]], "error": str(error)}


def prepare(
    questions_path,
    streams_path,
    video_root,
    output,
    detector=None,
    snapshot=None,
    workers=1,
    shard=None,
    split=None,
):
    """Prepare snapshots for every stream that has questions; `shard=(i, n)` splits streams."""
    questions = [json.loads(line) for line in Path(questions_path).read_text().splitlines() if line.strip()]
    if split:
        questions = [q for q in questions if q["split"] in set(split)]
    streams = json.loads(Path(streams_path).read_text())
    detector_cfg, snapshot_cfg = DetectorConfig.from_dict(detector), SnapshotConfig.from_dict(snapshot)
    by_stream = {}
    for question in questions:
        by_stream.setdefault(question["stream"], []).append(question)
    names = sorted(by_stream)
    if shard is not None:
        names = names[shard[0] :: shard[1]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(n, streams[n], by_stream[n], video_root, output, detector_cfg, snapshot_cfg) for n in names]
    if workers > 1:
        with get_context("spawn").Pool(workers) as pool:
            results = list(pool.imap_unordered(_run, jobs))
    else:
        results = [_run(job) for job in jobs]
    suffix = "" if shard is None else f".{shard[0]}of{shard[1]}"
    report = {
        "streams": len(results),
        "snapshots": sum(r["snapshots"] for r in results),
        "missed": [m for r in results for m in r["missed"]],
        "errors": {r["stream"]: r["error"] for r in results if "error" in r},
        "detector": detector_cfg.__dict__,
        "snapshot": snapshot_cfg.__dict__,
    }
    (output / f"prepare_report{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
