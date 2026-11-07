"""Alert-head data and evaluation on frame-labelled video anomaly sets (ShanghaiTech Campus, CUHK Avenue).

Licences: neither project page states a licence (both ask for citation only). Treat them as academic research
data: each tester downloads them from the official pages, nothing derived from them is committed or
redistributed, and results are reported with citations. Record the download date/version in
configs/datasets/downloads.json.

Layouts read here (official archives):
  ShanghaiTech  training/videos/<scene>_<id>.avi                 (normal only)
                testing/frames/<scene>_<id>/<n>.jpg              (frame folders)
                testing/test_frame_mask/<scene>_<id>.npy         (0/1 per frame)
  Avenue        training_videos/NN.avi (normal only), testing_videos/NN.avi
                ground_truth_demo/testing_label_mask/N_label.mat (volLabel: one pixel mask per frame; needs scipy:
                `uv run --with scipy ...`), or --labels DIR with <stem>.npy frame labels from any source.

Records are alert-only (no `question`): one per video, `alerts` = a noul question with the abnormal spans in
seconds, `query_time` = video end. The standard protocols train only on normal videos; the alert head needs
positives, so test videos are split video-disjointly: by a given list (e.g. the Zhong et al. 2019 weakly
supervised ShanghaiTech split, 238/199 videos) or by a seeded hash. The other half is never trained on.

Evaluation (`evaluate_alerts`): the stream runs once per video and the alert head answers the canonical
question at every step (no decoding). Labels are the frame label at the step centre (as in training). Metrics:
micro step-level ROC-AUC and AP over all test videos (the usual frame-level protocol, at step resolution), with
a whole-video bootstrap CI, plus milliseconds per decision. `method="lm"` is the decoding baseline: the LM
answers a Yes/No prompt on a forked cache every step (p = softmax of the fp32 Yes/No log-probs).
"""

from __future__ import annotations

import hashlib
import json
import math
import time as clock
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import StepReader, read_jsonl, step_times, write_jsonl

CANONICAL = "Is anything abnormal or unusual happening in the scene right now?"
PARAPHRASES = [
    CANONICAL,
    "Is there an abnormal event in the scene at this moment?",
    "Does the current activity look unusual for this place?",
    "Is someone or something behaving abnormally right now?",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------- labels
def frame_spans(labels, fps: float) -> list[list[float]]:
    """Runs of abnormal frames -> [[start_s, end_s], ...] (frame i is at i / fps)."""
    labels = np.asarray(labels).astype(bool).reshape(-1)
    spans, start = [], None
    for i, value in enumerate(labels.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append([round(start / fps, 3), round((i - 1) / fps, 3)])
            start = None
    return spans


def step_label(spans, t: float) -> int:
    return int(any(a <= t <= b for a, b in spans))


def _avenue_mat(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("Avenue .mat labels need scipy: `uv run --with scipy trace-vl v3 convert-anomaly ...`, "
                          "or pass --labels with per-video .npy frame labels") from error  # fmt: skip
    masks = loadmat(str(path))["volLabel"].reshape(-1)
    return np.array([int(np.asarray(m).sum() > 0) for m in masks])


def _split_of(name: str, train_list: set[str] | None, seed: int, train_fraction: float) -> str:
    if train_list is not None:
        return "train" if name in train_list else "test"
    digest = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    return "train" if int(digest[:8], 16) / 0xFFFFFFFF < train_fraction else "test"


def _video_frames(path: Path) -> tuple[int, float | None]:
    if path.is_dir():
        return len([f for f in path.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES]), None
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS) or None
    finally:
        capture.release()


def _record(dataset: str, name: str, video: str, labels, fps: float, split: str, camera: str, n_frames: int) -> dict:
    n = min(len(labels), n_frames) if labels is not None else n_frames
    spans = frame_spans(labels[:n], fps) if labels is not None else []
    text = PARAPHRASES[int(hashlib.sha256(name.encode()).hexdigest()[:4], 16) % len(PARAPHRASES)]
    if split != "train":
        text = CANONICAL  # evaluation always asks the canonical question
    record = {"id": f"{dataset}-{name}", "source": dataset, "video": video, "camera": camera,
              "query_time": round((n - 1) / fps, 3), "split": split, "fps": fps, "alert_only": True,
              "alerts": [{"id": "abnormal", "kind": "noul", "text": text, "spans": spans}],
              "abnormal_fraction": round(float(np.mean(labels[:n])) if labels is not None and n else 0.0, 4)}  # fmt: skip
    return record


def convert_anomaly(dataset: str, root: str, output: str, labels_dir: str | None = None, train_list: str | None = None,
                    train_fraction: float = 0.5, normal_train: int = 100, fps: float | None = None,
                    seed: int = 7, prefix: str = "") -> dict:  # fmt: skip
    """ShanghaiTech / Avenue -> alert-only records. Video paths are `prefix` + path relative to `root`, so with a
    prefix they resolve against a common parent --video-root shared with other datasets."""
    root_path = Path(root)
    listed = None
    if train_list:
        listed = {Path(x.strip()).stem for x in Path(train_list).read_text().splitlines() if x.strip()}
    rows, missing = [], []
    if dataset == "shanghaitech":
        rate = fps or 24.0  # frame folders carry no rate; ShanghaiTech is usually processed at 24 fps
        frames_root = root_path / "testing" / "frames"
        for folder in sorted(p for p in frames_root.iterdir() if p.is_dir()):
            mask = Path(labels_dir) / f"{folder.name}.npy" if labels_dir else root_path / "testing" / "test_frame_mask" / f"{folder.name}.npy"
            if not mask.is_file():
                missing.append(folder.name)
                continue
            n, _ = _video_frames(folder)
            split = _split_of(folder.name, listed, seed, train_fraction)
            rows.append(_record(dataset, folder.name, prefix + str(folder.relative_to(root_path)), np.load(mask), rate, split,
                                folder.name.split("_")[0], n))  # fmt: skip
        videos = sorted((root_path / "training" / "videos").glob("*.avi"))
        normals = [videos[i] for i in sorted(np.random.default_rng(seed).permutation(len(videos))[:normal_train])]
    elif dataset == "avenue":
        for video in sorted((root_path / "testing_videos").glob("*.avi")):
            n, rate = _video_frames(video)
            rate = fps or rate or 25.0
            npy = Path(labels_dir) / f"{video.stem}.npy" if labels_dir else None
            mat = root_path / "ground_truth_demo" / "testing_label_mask" / f"{int(video.stem)}_label.mat"
            if npy is not None and npy.is_file():
                labels = np.load(npy)
            elif npy is None and mat.is_file():
                labels = _avenue_mat(mat)
            else:
                missing.append(video.stem)
                continue
            split = _split_of(video.stem, listed, seed, train_fraction)
            rows.append(_record(dataset, f"test{video.stem}", prefix + str(video.relative_to(root_path)), labels, rate, split,
                                "avenue", n))  # fmt: skip
        videos = sorted((root_path / "training_videos").glob("*.avi"))
        normals = videos[:normal_train]
    else:
        raise ValueError("dataset must be shanghaitech or avenue")
    for video in normals:  # normal-only training videos: negatives (no spans)
        n, rate = _video_frames(video)
        if n < 2:
            continue
        camera = video.stem.split("_")[0] if dataset == "shanghaitech" else "avenue"
        rows.append(_record(dataset, f"train{video.stem}", prefix + str(video.relative_to(root_path)), None, fps or rate or 25.0,
                            "train", camera, n))  # fmt: skip
    if not rows:
        raise ValueError(f"no labelled videos found under {root}")
    write_jsonl(output, rows)
    report = {"records": len(rows), "missing_labels": missing[:10], "missing_count": len(missing)}
    for split in ("train", "test"):
        part = [r for r in rows if r["split"] == split]
        report[split] = {"videos": len(part), "with_abnormal": sum(bool(r["alerts"][0]["spans"]) for r in part),
                         "seconds": round(sum(r["query_time"] for r in part), 1),
                         "abnormal_fraction": round(sum(r["abnormal_fraction"] * r["query_time"] for r in part)
                                                    / max(1e-9, sum(r["query_time"] for r in part)), 4)}  # fmt: skip
    return report


# ---------------------------------------------------------------------------- metrics
def roc_auc(labels, scores) -> float | None:
    """Mann-Whitney ROC-AUC with average ranks for ties; None when one class is absent."""
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(labels, scores) -> float | None:
    """AP (area under the step-wise precision-recall curve, ties grouped)."""
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # end of each tie group
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def score_alerts(predictions: str, output: str | None = None, replicates: int = 2000, seed: int = 7) -> dict:
    rows = read_jsonl(predictions)
    runs = {r.get("run") for r in rows}
    if len(runs) > 1:
        raise ValueError(f"{predictions} mixes runs {sorted(map(str, runs))}")
    by_video: dict[str, list] = defaultdict(list)
    for r in rows:
        by_video[r["id"]].append(r)
    videos = sorted(by_video)
    labels = np.array([r["label"] for v in videos for r in by_video[v]])
    scores = np.array([r["p"] for v in videos for r in by_video[v]])
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(replicates):
        pick = rng.integers(0, len(videos), len(videos))
        y = np.concatenate([[r["label"] for r in by_video[videos[i]]] for i in pick])
        s = np.concatenate([[r["p"] for r in by_video[videos[i]]] for i in pick])
        value = roc_auc(y, s)
        if value is not None:
            aucs.append(value)
    ms = [r["ms"] for r in rows if r.get("ms") is not None]
    result = {"videos": len(videos), "steps": len(rows), "positive_fraction": float(labels.mean()) if len(rows) else None,
              "auc": roc_auc(labels, scores), "ap": average_precision(labels, scores),
              "auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else None,
              "ms_per_decision_mean": float(np.mean(ms)) if ms else None, "run": next(iter(runs), None)}  # fmt: skip
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


# ---------------------------------------------------------------------------- evaluation
def evaluate_alerts(cfg, records_path: str, video_root: str, output: str, checkpoint: str | None = None,
                    adapter: str | None = None, method: str = "head", device: str = "cuda", split: str = "test",
                    limit: int | None = None) -> dict:  # fmt: skip
    """Per-step abnormality scores on held-out videos. Resumable per video; refuses to mix runs."""
    import torch

    from .alert import Question
    from .backbone import Backbone
    from .harness import _modules, clean_partial, read_unit, run_fingerprint, write_unit
    from .session import StreamSession

    if method not in {"head", "lm"}:
        raise ValueError("method must be head or lm")
    if method == "head" and not checkpoint:
        raise ValueError("the alert head needs a trained --checkpoint")
    records = [r for r in read_jsonl(records_path) if r.get("split") == split and r.get("alerts")]
    if not records:
        raise ValueError(f"no alert records with split == {split}")
    if limit:
        records = records[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(cfg, records_path, f"alerts-{method}-{split}", checkpoint, adapter, video_root, records)
    run_file = output / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()).get("id") != fingerprint["id"]:
        raise ValueError(f"{output} holds a different run; use a new --output")
    run_file.write_text(json.dumps(fingerprint, indent=2) + "\n")
    units = output / "videos"  # one atomically written file per completed video (resume = file present + complete)
    if units.exists():
        clean_partial(units)
    done = {r["id"] for r in records if read_unit(units, r["id"], fingerprint["id"]) is not None}
    backbone = Backbone(cfg.backbone, cfg.stream, device)
    if adapter:
        backbone.add_lora(0, 0, adapter=adapter)
        backbone.model.eval()
    modules = _modules(cfg, backbone, checkpoint, device) if checkpoint else None
    mode = "v3" if modules is not None else "native"
    question = Question("abnormal", "noul", CANONICAL)
    prompt = CANONICAL + " Answer Yes or No."
    gap = cfg.stream.frame_gap
    for record in records:
        if record["id"] in done:
            continue
        spans = record["alerts"][0]["spans"]
        session = StreamSession(backbone, cfg, modules, mode=mode,
                                questions=[question] if method == "head" else None)  # fmt: skip
        reader = StepReader(record, video_root)
        rows = []
        try:
            end = min(float(record["query_time"]), reader.duration)
            for t in [t for t in step_times(end, cfg.stream.step_seconds, gap) if t + gap / 2 <= end + 1e-6]:
                result = session.step(reader.step(t, gap), t)
                if method == "head":
                    p, ms = result.alerts[0]["noul"], result.seconds_alert * 1000
                else:
                    started = clock.perf_counter()
                    _, scores = session.ask_yesno(prompt)
                    ms = (clock.perf_counter() - started) * 1000
                    p = 1.0 / (1.0 + math.exp(scores[1] - scores[0]))
                rows.append({"id": record["id"], "t": t, "label": step_label(spans, t), "p": round(float(p), 5),
                             "ms": round(ms, 3), "camera": record.get("camera"), "run": fingerprint["id"]})  # fmt: skip
        finally:
            reader.close()
        write_unit(units, record["id"], rows, fingerprint["id"])
        del session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = []
    for record in records:  # every selected video must be complete before scoring
        unit = read_unit(units, record["id"], fingerprint["id"])
        if unit is None:
            raise RuntimeError(f"video {record['id']} has no complete result")
        rows += unit
    target = output / "predictions.jsonl"  # assembled from complete videos only; rewritten each time
    write_jsonl(target, rows)
    return score_alerts(str(target), str(output / "summary.json"))
