"""Timed multiple-choice records, benchmark converters, and a causal step reader.

Record (one JSON line):
    {"id", "source", "task", "video" | "clips": [{"video", "offset"}], "query_time": float | null,
     "question", "options": [...], "answer": int, "region": [x1, y1, x2, y2] | null, "split"}
`query_time = null` means "after the whole video" (offline benchmarks). Questions are answered from
frames at or before `query_time` only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# ---------------------------------------------------------------------------- converters
OVO_MCQ_TASKS = {
    "EPM": "backward", "ASI": "backward", "HLD": "backward",
    "STU": "realtime", "OJR": "realtime", "ATR": "realtime", "ACR": "realtime", "OCR": "realtime", "FPD": "realtime",
}  # fmt: skip


def convert_ovo(annotation: str, output: str, video_root: str | None = None) -> dict:
    """OVO-Bench (`data/ovo_bench_new.json`): backward and real-time multiple-choice tasks.

    Forward-active tasks (REC, SSR, CRR) poll the model at several times and need a separate protocol;
    they are counted and skipped here.
    """
    rows, skipped, missing = [], {}, 0
    for item in json.loads(Path(annotation).read_text()):
        task = item["task"]
        if task not in OVO_MCQ_TASKS:
            skipped[task] = skipped.get(task, 0) + 1
            continue
        if video_root and not (Path(video_root) / item["video"]).is_file():
            missing += 1  # partial download: keep only questions whose video is on disk
            continue
        rows.append({
            "id": f"ovo-{item['id']}", "source": "ovo-bench", "task": f"{OVO_MCQ_TASKS[task]}/{task}",
            "video": item["video"], "query_time": float(item["realtime"]), "question": item["question"],
            "options": item["options"], "answer": int(item["gt"]), "region": None, "split": "test",
        })  # fmt: skip
    write_jsonl(output, rows)
    return {"records": len(rows), "skipped_forward_tasks": skipped, "missing_videos_skipped": missing,
            "videos": len({r["video"] for r in rows})}  # fmt: skip


def convert_videomme(parquet: str, output: str) -> dict:
    """Video-MME (lmms-lab/Video-MME parquet). Offline: questions after the whole video."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet).to_pylist()
    rows = []
    for item in table:
        options = [re.sub(r"^[A-D]\.\s*", "", o) for o in item["options"]]
        rows.append({
            "id": f"videomme-{item['question_id']}", "source": "video-mme", "task": f"{item['duration']}/{item['task_type']}",
            "video": f"{item['videoID']}.mp4", "query_time": None, "question": item["question"], "options": options,
            "answer": "ABCD".index(item["answer"].strip()[0]), "region": None, "split": "test",
        })  # fmt: skip
    write_jsonl(output, rows)
    return {"records": len(rows)}


def convert_meva(questions: str, streams: str, output: str, split: str | None = "test") -> dict:
    """MEVA long-gap questions (inspection benchmark) as multi-clip streams with a query region."""
    specs = json.loads(Path(streams).read_text())
    rows = []
    for q in read_jsonl(questions):
        if split and q["split"] != split:
            continue
        clips = [{"video": c["video"], "offset": c["offset"]} for c in specs[q["stream"]]["clips"]]
        rows.append({
            "id": f"meva-{q['id']}", "source": "meva-longgap", "task": q["type"], "clips": clips, "stream": q["stream"],
            "query_time": float(q["query_time"]), "question": q["question"], "options": q["options"],
            "answer": int(q["answer"]), "region": q.get("region"), "split": q["split"], "gap": q.get("gap"),
            "camera": q.get("camera"),
        })  # fmt: skip
    write_jsonl(output, rows)
    return {"records": len(rows)}


# ---------------------------------------------------------------------------- reading steps
class StepReader:
    """Decode the two frames of each step (t - gap/2, t + gap/2) from one video or a chain of clips."""

    def __init__(self, record: dict, video_root: str | Path):
        import cv2

        self.cv2 = cv2
        root = Path(video_root)
        clips = record.get("clips") or [{"video": record["video"], "offset": 0.0}]
        self.clips = []
        for clip in clips:
            capture = cv2.VideoCapture(str(root / clip["video"]))
            if not capture.isOpened():
                raise FileNotFoundError(root / clip["video"])
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or count <= 0:
                raise ValueError(f"bad metadata in {clip['video']}")
            self.clips.append({"cap": capture, "fps": fps, "count": count, "offset": float(clip["offset"]),
                               "next": 0, "last": None})  # fmt: skip
        last = self.clips[-1]
        self.duration = last["offset"] + (last["count"] - 1) / last["fps"]

    def close(self):
        for clip in self.clips:
            clip["cap"].release()

    def _frame(self, t: float) -> np.ndarray:
        clip = next((c for c in reversed(self.clips) if c["offset"] <= t), self.clips[0])
        index = int(min(max(round((t - clip["offset"]) * clip["fps"]), 0), clip["count"] - 1))
        cap = clip["cap"]
        if clip["last"] is not None and clip["last"][0] == index:
            return clip["last"][1]
        if index < clip["next"] or index - clip["next"] > 3 * clip["fps"]:
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            clip["next"] = index
        while clip["next"] < index:
            cap.grab()
            clip["next"] += 1
        ok, frame = cap.read()
        clip["next"] += 1
        if not ok:
            if clip["last"] is None:
                raise ValueError(f"cannot decode frame {index}")
            return clip["last"][1]  # tolerate a broken tail frame
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        clip["last"] = (index, rgb)
        return rgb

    def step(self, t: float, gap: float) -> np.ndarray:
        """Two frames centred on t; never later than t + gap/2 (the step's own window)."""
        a = self._frame(max(0.0, t - gap / 2))
        b = self._frame(min(self.duration, t + gap / 2))
        if a.shape != b.shape:
            b = self.cv2.resize(b, (a.shape[1], a.shape[0]))
        return np.stack([a, b])


def step_times(end: float, step_seconds: float, gap: float, max_steps: int = 0) -> list[float]:
    """Centres of steps covering [0, end]; uniformly subsampled to `max_steps` when > 0 (offline)."""
    times = list(np.arange(gap / 2, max(end - gap / 2, gap / 2) + 1e-6, step_seconds))
    if max_steps and len(times) > max_steps:
        picks = np.linspace(0, len(times) - 1, max_steps).round().astype(int)
        times = [times[i] for i in sorted(set(picks.tolist()))]
    return [round(float(t), 3) for t in times]


def draw_region(frames: np.ndarray, region) -> np.ndarray:
    """Red rectangle on a copy of the frames (query-time marker for region questions)."""
    import cv2

    out = frames.copy()
    h, w = out.shape[1:3]
    x1, y1, x2, y2 = (int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h))
    for frame in out:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), max(2, round(max(h, w) / 200)))
    return out
