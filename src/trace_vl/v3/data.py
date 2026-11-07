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


# OVO-Bench forward-active prompt templates, verbatim from the official repo (constant.py).
OVO_REC_TEMPLATE = """
You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""
OVO_SSR_TEMPLATE = """
You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""
OVO_CRR_TEMPLATE = """
You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""


def _ovo_forward(item) -> list[dict]:
    """One record per polling time (test_info entry), answered with the official prompt; scoring follows the
    official scorer (REC: digits of the response == count; SSR/CRR: Yes/No vs `type`)."""
    rows, task = [], item["task"]
    for i, info in enumerate(item["test_info"]):
        if task == "REC":
            prompt = OVO_REC_TEMPLATE.format("How many times did they " + item["activity"] + "?")
            kind, answer = "count", str(info["count"])
        elif task == "SSR":
            prompt, kind, answer = OVO_SSR_TEMPLATE.format(info["step"]), "yesno", "Yes" if info["type"] == 1 else "No"
        else:  # CRR
            prompt, kind, answer = OVO_CRR_TEMPLATE.format(item["question"]), "yesno", "Yes" if info["type"] == 1 else "No"
        rows.append({
            "id": f"ovo-{item['id']}-{i}", "source": "ovo-bench", "task": f"forward/{task}", "video": item["video"],
            "query_time": float(info["realtime"]), "kind": kind, "prompt": prompt.strip(), "answer": answer,
            "region": None, "split": "test",
        })  # fmt: skip
    return rows


def convert_ovo(annotation: str, output: str, video_root: str | None = None, forward: bool = False) -> dict:
    """OVO-Bench (`data/ovo_bench_new.json`): backward and real-time multiple-choice tasks.

    Forward-active tasks (REC, SSR, CRR) poll the model at several times and need a separate protocol;
    they are counted and skipped here.
    """
    rows, skipped, missing = [], {}, 0
    for item in json.loads(Path(annotation).read_text()):
        task = item["task"]
        if video_root and not (Path(video_root) / item["video"]).is_file():
            missing += 1  # partial download: keep only questions whose video is on disk
            continue
        if task not in OVO_MCQ_TASKS:
            if forward and task in {"REC", "SSR", "CRR"}:
                rows.extend(_ovo_forward(item))
            else:
                skipped[task] = skipped.get(task, 0) + 1
            continue
        rows.append({
            "id": f"ovo-{item['id']}", "source": "ovo-bench", "task": f"{OVO_MCQ_TASKS[task]}/{task}",
            "video": item["video"], "query_time": float(item["realtime"]), "question": item["question"],
            "options": item["options"], "answer": int(item["gt"]), "region": None, "split": "test",
        })  # fmt: skip
    write_jsonl(output, rows)
    return {"records": len(rows), "skipped_forward_tasks": skipped, "missing_videos_skipped": missing,
            "videos": len({r["video"] for r in rows})}  # fmt: skip


def convert_videomme(parquet: str, output: str, video_root: str | None = None) -> dict:
    """Video-MME (lmms-lab/Video-MME parquet). Offline: questions after the whole video."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet).to_pylist()
    rows, missing = [], 0
    for item in table:
        if video_root and not (Path(video_root) / f"{item['videoID']}.mp4").is_file():
            missing += 1  # partial download (e.g. a pinned 30-video shard)
            continue
        options = [re.sub(r"^[A-D]\.\s*", "", o) for o in item["options"]]
        rows.append({
            "id": f"videomme-{item['question_id']}", "source": "video-mme", "task": f"{item['duration']}/{item['task_type']}",
            "video": f"{item['videoID']}.mp4", "query_time": None, "question": item["question"], "options": options,
            "answer": "ABCD".index(item["answer"].strip()[0]), "region": None, "split": "test",
        })  # fmt: skip
    write_jsonl(output, rows)
    return {"records": len(rows), "videos": len({r["video"] for r in rows}), "missing_videos_skipped": missing}


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


def convert_tocbench(answer_json: str, output: str, video_root: str | None = None) -> dict:
    """TOC-Bench (answer_final.json) -> offline records. mcq_4 / statement-pair / numerical become lettered
    options; ordering items are generated and compared as a letter sequence. This is a local option-likelihood
    protocol, not the official TOC-Bench prompting or extractor."""
    data = json.loads(Path(answer_json).read_text())
    items = next(v for v in data.values() if isinstance(v, list)) if isinstance(data, dict) else data
    rows, missing, skipped = [], 0, 0
    for item in items:
        video = f"videos/{item['video_id']}.mp4"
        if video_root and not (Path(video_root) / video).is_file():
            missing += 1
            continue
        fmt, meta = item["format"], item.get("metadata", {})
        base = {"id": f"toc-{item['qa_id']}", "source": "toc-bench", "task": f"{fmt}/{meta.get('dim')}",
                "video": video, "query_time": None, "region": None, "split": "test", "tier": meta.get("tier")}  # fmt: skip
        if fmt == "mcq_4":
            options = [item[f"option_{c}"] for c in "ABCD"]
            rows.append({**base, "question": item["question"], "options": options, "answer": "ABCD".index(item["correct_answer"])})
        elif fmt == "sp":
            options = [item["statement_A"], item["statement_B"]]
            question = item["question"] + " Which statement is true?"
            rows.append({**base, "question": question, "options": options, "answer": "AB".index(item["correct_answer"])})
        elif fmt == "numerical":
            options = list(item["allowed_answers"])
            if item["correct_answer"] not in options:
                skipped += 1
                continue
            rows.append({**base, "question": item["question"], "options": options, "answer": options.index(item["correct_answer"])})
        elif fmt.startswith("ordering"):
            events = "\n".join(f"{e['label']}. {e['event_text']}" for e in item["events"])
            prompt = (f"{item['question']}\n{events}\nAnswer with the letters in order from earliest to latest, "
                      "separated by commas (e.g., B, A, C).")  # fmt: skip
            rows.append({**base, "kind": "order", "prompt": prompt, "answer": "".join(item["correct_order"]),
                         "n_events": len(item["events"])})  # fmt: skip
        else:
            skipped += 1
    write_jsonl(output, rows)
    return {"records": len(rows), "missing_videos_skipped": missing, "skipped": skipped,
            "videos": len({r["video"] for r in rows}), "by_format": {f: sum(r["task"].startswith(f) for r in rows)
                                                                    for f in ("mcq_4", "sp", "numerical", "ordering")}}  # fmt: skip


def parse_count(text: str) -> str:
    """Official OVO-Bench REC extraction: concatenate every digit run in the response."""
    return "".join(re.findall(r"\d+", text or ""))


def parse_order(text: str, n: int) -> str:
    """Letters A.. in order of first appearance (as standalone tokens), first n of them."""
    letters = "ABCDEFGH"[:n]
    seen = []
    for match in re.findall(r"\b([A-H])\b", text or ""):
        if match in letters and match not in seen:
            seen.append(match)
    return "".join(seen[:n])


OPTION_LINE = re.compile(r"^\s*\(?([A-H])[\.\):]\s*(.+?)\s*$")


def convert_llava_mc(annotations: list[str], output: str, video_root: str | None = None, val_fraction: float = 0.05,
                     seed: int = 7) -> dict:  # fmt: skip
    """LLaVA-Video-178K multiple-choice QA files -> Stage 1 records (offline: question after the clip).

    Each human turn holds the question followed by option lines ("A. ..."); the gpt turn starts with
    the letter. Unparseable turns are counted and skipped. Split by video (hash), never by question.

    Question identity: the official `item.id` repeats across different questions of the same video
    (PR #3 review: 5,496 questions -> 1,296 ids), so ids are derived from video + normalized question +
    ordered options; the upstream id and turn are kept as provenance. Exact repeats are deduplicated and
    the same question with conflicting answers is rejected (all copies dropped and counted).
    """
    import hashlib

    by_key: dict[str, dict] = {}
    conflicts: set[str] = set()
    skipped, missing, duplicates = 0, 0, 0
    for path in annotations:
        subset = Path(path).stem
        for item in json.loads(Path(path).read_text()):
            video = item.get("video")
            if not video:
                skipped += 1
                continue
            if video_root and not (Path(video_root) / video).is_file():
                missing += 1
                continue
            turns = item.get("conversations", [])
            for k in range(0, len(turns) - 1, 2):
                human, gpt = turns[k].get("value", ""), turns[k + 1].get("value", "")
                lines = [x for x in human.replace("<image>", "").replace("<video>", "").splitlines() if x.strip()]
                options, question = [], []
                for line in lines:
                    match = OPTION_LINE.match(line)
                    if match and len(options) == "ABCDEFGH".index(match.group(1)):
                        options.append(match.group(2))
                    elif not options:
                        question.append(line.strip())
                answer = re.match(r"\s*\(?([A-H])\b", gpt)
                if len(options) < 2 or not question or not answer or "ABCDEFGH".index(answer.group(1)) >= len(options):
                    skipped += 1
                    continue
                text = " ".join(question)
                normalized = " ".join(text.lower().split())
                identity = json.dumps([video, normalized, [" ".join(o.lower().split()) for o in options]])
                key = hashlib.sha256(identity.encode()).hexdigest()[:20]
                label = "ABCDEFGH".index(answer.group(1))
                if key in by_key:
                    if by_key[key]["answer"] != label:
                        conflicts.add(key)
                    else:
                        duplicates += 1
                    continue
                digest = hashlib.sha256(f"{seed}:{video}".encode()).hexdigest()
                split = "val" if int(digest[:8], 16) / 0xFFFFFFFF < val_fraction else "train"
                by_key[key] = {
                    "id": f"llava-mc-{key}", "source": "llava-video-178k-mc", "task": subset, "video": video,
                    "query_time": None, "question": text, "options": options, "answer": label, "region": None,
                    "split": split, "upstream_id": item.get("id"), "upstream_turn": k // 2,
                }  # fmt: skip
    rows = [r for key, r in by_key.items() if key not in conflicts]
    write_jsonl(output, rows)
    splits = {s: sum(r["split"] == s for r in rows) for s in ("train", "val")}
    return {"records": len(rows), "videos": len({r["video"] for r in rows}), "splits": splits,
            "skipped_unparseable": skipped, "missing_videos_skipped": missing, "duplicates_removed": duplicates,
            "conflicting_questions_dropped": len(conflicts), "unique_ids": len({r["id"] for r in rows})}  # fmt: skip


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


# ---------------------------------------------------------------------------- Stage 2 streams
def _video_meta(path: Path) -> tuple[float, int, int] | None:
    """(duration seconds, height, width) of a video, or None when unreadable."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if fps <= 0 or count <= 1 or height <= 0 or width <= 0:
            return None
        return (count - 1) / fps, height, width
    finally:
        capture.release()


def stage2_question(record: dict, time: float, evidence, delay: float, prefix: str | None = None) -> dict:
    text = record["question"] if prefix is None else prefix + record["question"]
    return {"id": record["id"], "time": round(float(time), 3), "question": text, "options": record["options"],
            "answer": record["answer"], "evidence": evidence, "delay": round(float(delay), 3)}  # fmt: skip


def build_stage2_streams(records_path: str, output: str, video_root: str, split: str = "train",
                         stream_seconds: float = 300.0, max_clips: int = 24, questions_per_clip: int = 2,
                         delay_min: float = 30.0, delay_max: float = 240.0, immediate_fraction: float = 0.25,
                         pixels: int = 448 * 448, unit: int = 32, seed: int = 7) -> dict:  # fmt: skip
    """Long training streams with delayed, time-referenced questions and evidence windows (Stage 2).

    - Records without `query_time` (offline clips, e.g. LLaVA-Video MC): clips of the same frame-grid bucket
      are concatenated into streams of about `stream_seconds`. Each question of a clip is asked either right
      after the clip (`immediate_fraction`) or `delay_min..delay_max` seconds later, while later clips keep
      streaming, and says which part of the stream it is about ("from 37 to 61 seconds", matching the
      `<t seconds>` step markers). Evidence = that clip's interval.
    - Records with `query_time` (timed questions on one long video, e.g. MEVA train): grouped per video into
      one stream, asked at their own times; evidence = the record's `evidence` windows when present.
    Questions are only asked at or after the end of their evidence, so the answer must come from memory.
    """
    import hashlib

    from .backbone import frame_size

    rng = np.random.default_rng(seed)
    root = Path(video_root)
    rows = [r for r in read_jsonl(records_path) if r.get("split", "train") == split]
    if not rows:
        raise ValueError(f"no records with split == {split}")
    if not 0 <= immediate_fraction <= 1 or delay_min < 0 or delay_max < delay_min or stream_seconds <= 0:
        raise ValueError("invalid delay / stream settings")
    timed: dict[str, list[dict]] = {}
    offline: dict[str, list[dict]] = {}
    for r in rows:
        (timed if r.get("query_time") is not None else offline).setdefault(r["video"], []).append(r)
    streams, skipped = [], 0

    def stream_id(parts) -> str:
        return "s2-" + hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]

    for video, items in sorted(timed.items()):
        meta = _video_meta(root / video)
        if meta is None:
            skipped += len(items)
            continue
        duration = min(meta[0], max(float(r["query_time"]) for r in items))
        questions = [stage2_question(r, float(r["query_time"]), r.get("evidence"), 0.0) for r in items
                     if float(r["query_time"]) <= meta[0]]  # fmt: skip
        if questions:
            streams.append({"id": stream_id([video]), "split": split, "source": "timed", "duration": round(duration, 3),
                            "clips": [{"video": video, "offset": 0.0}],
                            "questions": sorted(questions, key=lambda q: q["time"])})  # fmt: skip

    buckets: dict[tuple, list[tuple[str, float]]] = {}
    for video in sorted(offline):
        meta = _video_meta(root / video)
        if meta is None:
            skipped += len(offline[video])
            continue
        buckets.setdefault(frame_size(meta[1], meta[2], pixels, unit), []).append((video, meta[0]))
    for bucket in sorted(buckets):
        videos = buckets[bucket]
        order = rng.permutation(len(videos)).tolist()
        while order:
            clips, offset = [], 0.0
            while order and offset < stream_seconds and len(clips) < max_clips:
                video, duration = videos[order.pop()]
                clips.append({"video": video, "offset": round(offset, 3), "end": round(offset + duration, 3)})
                offset += duration
            questions = []
            for clip in clips:
                picks = offline[clip["video"]]
                picks = [picks[i] for i in rng.permutation(len(picks))[:questions_per_clip]]
                start, end = clip["offset"], clip["end"]
                for r in picks:
                    delay = 0.0 if rng.random() < immediate_fraction else float(rng.uniform(delay_min, delay_max))
                    ask = min(end + delay, offset)
                    prefix = f"Consider the part of the video from {start:.0f} to {end:.0f} seconds. "
                    questions.append(stage2_question(r, ask, [[start, end]], ask - end, prefix))
            streams.append({"id": stream_id([c["video"] for c in clips]), "split": split, "source": "concat",
                            "duration": round(offset, 3), "clips": clips,
                            "questions": sorted(questions, key=lambda q: q["time"])})  # fmt: skip
    write_jsonl(output, streams)
    delays = [q["delay"] for s in streams for q in s["questions"]]
    return {"streams": len(streams), "questions": len(delays), "skipped_records": skipped,
            "hours": round(sum(s["duration"] for s in streams) / 3600, 2),
            "delayed_fraction": round(sum(d > 0 for d in delays) / max(1, len(delays)), 3),
            "mean_delay": round(sum(delays) / max(1, len(delays)), 1)}  # fmt: skip
