"""Stage-2 transfer check on OVO-Bench (experiments.md §14.3): does S2 - S1 under FIFO hold on an external streaming
benchmark? Evaluation only, frozen checkpoints, no training.

Population, fixed before any model is run:
- Videos (v2): every video of the ORIGINAL §13 diagnostic hash half (seed 7, fraction 0.5). v1 used only the 200
  videos of the §13 manifest (`split/diagnostic.jsonl` of the `9653e2a` run). Those 200 already informed the decision to close the transition-token track, so this is an
  **external-dataset development evaluation**, not a held-out confirmation. The frozen final half (hash split,
  `final_frozen.jsonl`) is never touched; disjointness is checked on video IDs.
- Questions: ALL official multiple-choice questions on those videos from the official annotation via `convert_ovo`
  (backward EPM/ASI/HLD and real-time STU/OJR/ATR/ACR/OCR/FPD), with query time <= `max_query_seconds` (600 s,
  the Stage-2 maximum). This is a NEW, larger question population than the 475-question §13 manifest
  (which held ASI/REC/SSR and real-time tasks, sampled per task) and is named `ovo-transfer-dev-v2`.
- Eligibility depends only on data and the stream protocol (query time, video readable, at least one complete
  step before the query time), never on model outputs; every exclusion is counted.

Streams: one per video, cut at the latest eligible query time; questions are answered after the last step whose
two-frame window ends at or before the query time (no future frames) on a forked cache (the live stream is never
changed by a question). Every arm streams the identical step times.

Endpoints (predeclared): primary S2-FIFO - S1-FIFO on pooled backward questions (question-weighted), video-cluster
bootstrap (4,000 draws, seed 7). OVO has no evidence spans: backward results are never relabelled "evidence older
than the window".
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .data import OVO_MCQ_TASKS, read_jsonl, write_jsonl
from .temporal import task_code, video_half

PROTOCOL = "ovo-transfer-dev-v2"
# v1 (e90d586): only the 200 videos sampled into the §13 manifest. §13 had dropped EPM/HLD before sampling, so v1 held
# 51 backward questions (ASI only) and failed adequacy on the A100 inventory. v2: every video of the ORIGINAL
# diagnostic hash half (same seed and fraction as the §13 split), all official MC questions on them.
POPULATIONS = {"sampled-200": "ovo-transfer-dev-v1", "diagnostic-half": "ovo-transfer-dev-v2"}
BACKWARD = ("EPM", "ASI", "HLD")
REALTIME = ("STU", "OJR", "ATR", "ACR", "OCR", "FPD")
ARMS = ("s1_fifo", "s2_fifo", "s1_full", "s2_full")
# Coverage adequacy, fixed before GPU work: below any of these, the inventory is reported and the GPU stages stop
# unless the owner accepts it explicitly (an underpowered run is not evidence that the gain does not transfer).
MIN_BACKWARD_QUESTIONS = 150
MIN_BACKWARD_VIDEOS = 30
MAX_VIDEO_SHARE = 0.10


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_id(video: str) -> str:
    return "ovo-" + hashlib.sha1(video.encode()).hexdigest()[:12]


def probe_duration(path: Path) -> float | None:
    """Duration in seconds from container metadata (CPU, no decoding); None if unreadable."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return (count - 1) / fps if fps > 0 and count > 0 else None
    finally:
        cap.release()


ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def _archive_members(path: Path) -> list[str]:
    import tarfile
    import zipfile

    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return [m for m in z.namelist() if not m.endswith("/")]
    with tarfile.open(path) as t:
        return [m.name for m in t.getmembers() if m.isfile()]


def _extract_member(archive: Path, member: str, target: Path):
    import shutil
    import tarfile
    import zipfile

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z, z.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    else:
        with tarfile.open(archive) as t, t.extractfile(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    tmp.replace(target)


def ensure_media(videos: list[str], video_root: str | Path, search_roots: list[str | Path], out_root: str | Path,
                 probe=None) -> dict:  # fmt: skip
    """Media for the population as a symlink overlay `out_root/media/<video>` (the evaluation video root).

    A video readable under `video_root` is linked in place. Only a missing or unreadable video is looked up in the
    archives already downloaded under `search_roots` (member path ending in the video's relative path, else the same
    file name) and that single member is extracted into `out_root/media-extracted/`. Archives and the original
    media are only read. Nothing is extracted when every video is available."""
    probe = probe or probe_duration
    out = Path(out_root)
    overlay, extracted_root = out / "media", out / "media-extracted"
    available, missing, extracted = [], [], {}

    def link(video: str, source: Path):
        dest = overlay / video
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(source.resolve())

    for v in videos:
        src = Path(video_root) / v
        if src.is_file() and probe(src) is not None:
            link(v, src)
            available.append(v)
        elif (extracted_root / v).is_file() and probe(extracted_root / v) is not None:
            link(v, extracted_root / v)  # extracted by an earlier attempt of this run
            extracted[v] = "previously extracted"
        else:
            missing.append(v)
    archives = []
    if missing:
        for root in search_roots:
            root = Path(root)
            if root.is_dir():
                archives += sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(ARCHIVE_SUFFIXES))
        archives = sorted(set(archives))
        wanted = set(missing)
        for archive in archives:
            if not wanted:
                break
            try:
                members = _archive_members(archive)
            except Exception:  # noqa: BLE001 - an unreadable archive is reported, not fatal
                continue
            for v in sorted(wanted):
                hit = next((m for m in members if m.endswith(v)), None) or next(
                    (m for m in members if Path(m).name == Path(v).name), None)
                if hit is None:
                    continue
                _extract_member(archive, hit, extracted_root / v)
                if probe(extracted_root / v) is not None:
                    link(v, extracted_root / v)
                    extracted[v] = f"{archive}::{hit}"
                    wanted.discard(v)
        missing = sorted(wanted)
    return {"videos": len(videos), "available_in_root": len(available), "extracted": extracted,
            "still_missing": missing, "archives_scanned": [str(a) for a in archives], "overlay": str(overlay),
            "video_root": str(video_root)}  # fmt: skip


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    pick = lambda q: round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 1)  # noqa: E731
    return {"min": round(v[0], 1), "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": round(v[-1], 1)}


def _visible(query_time: float, step_seconds: float, gap: float, end: float, max_seconds: float) -> bool:
    """True if chunk_plan would ask this question on a stream cut at `end`: at least one complete step window
    ends at or before the query time."""
    from .train_stage2 import chunk_plan, stream_times

    times = stream_times({"duration": end}, step_seconds, gap, max_seconds)
    plan = chunk_plan(times, [{"time": query_time}], len(times) + 1, gap)
    return any(qs for _, qs in plan)


def half_counts(mc_records: str, seed: int = 7, fraction: float = 0.5) -> dict:
    """Questions and videos per MC task over the complete conversion and over each hash half (§13 seed/fraction)."""
    out: dict = {}
    for r in read_jsonl(mc_records):
        code = task_code(r)
        if code not in OVO_MCQ_TASKS:
            continue
        for scope in ("all", video_half(r["video"], seed, fraction)):
            cell = out.setdefault(scope, {}).setdefault(code, {"questions": 0, "videos": set()})
            cell["questions"] += 1
            cell["videos"].add(r["video"])
    return {scope: {t: {"questions": c["questions"], "videos": len(c["videos"])} for t, c in sorted(tasks.items())}
            for scope, tasks in sorted(out.items())}  # fmt: skip


def build_transfer(mc_records: str, diagnostic: str, final_frozen: str, video_root: str, out_dir: str,
                   step_seconds: float, gap: float, max_seconds: float, max_query_seconds: float = 600.0,
                   durations: dict | None = None, population: str = "diagnostic-half", seed: int = 7,
                   fraction: float = 0.5) -> dict:  # fmt: skip
    """Inventory + `questions.jsonl` + `streams.jsonl` (split "dev") of the transfer population. CPU only.

    population "diagnostic-half" (v2): every video of the §13 diagnostic hash half (`seed`, `fraction` from the §13
    split manifest); "sampled-200" (v1): only the videos of `diagnostic` (kept for the old/new comparison).
    Every video hashing to the final half is excluded, whether or not it appears in `final_frozen`.
    `durations` (video -> seconds or None) replaces probing, for tests."""
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {sorted(POPULATIONS)}")
    protocol = POPULATIONS[population]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    diag_rows, final_rows = read_jsonl(diagnostic), read_jsonl(final_frozen)
    sampled = sorted({r["video"] for r in diag_rows})
    final_videos = {r["video"] for r in final_rows}
    mc_all = read_jsonl(mc_records)
    if population == "sampled-200":
        diag_videos = sampled
    else:
        diag_videos = sorted({r["video"] for r in mc_all if video_half(r["video"], seed, fraction) == "diagnostic"})
    overlap = sorted(set(diag_videos) & final_videos)
    hashed_final = sorted(v for v in diag_videos if video_half(v, seed, fraction) != "diagnostic")
    if overlap or hashed_final or any(video_half(v, seed, fraction) != "diagnostic" for v in sampled):
        raise AssertionError(f"diagnostic videos overlap the frozen final half: {overlap[:5]} {hashed_final[:5]}")

    exclusions: Counter = Counter()
    excluded_by_task: dict = defaultdict(Counter)
    seen_ids, kept = set(), []
    in_scope = [r for r in mc_all if r["video"] in set(diag_videos) and task_code(r) in OVO_MCQ_TASKS]
    if durations is None:
        durations = {v: probe_duration(Path(video_root) / v) for v in diag_videos}
    for r in sorted(in_scope, key=lambda r: r["id"]):
        q = r.get("query_time")
        options = r.get("options") or []
        reason = None
        if r["id"] in seen_ids:
            reason = "duplicate_id"
        elif q is None:
            reason = "query_time_missing"
        elif not (len(options) >= 2 and isinstance(r.get("answer"), int) and 0 <= r["answer"] < len(options)):
            reason = "invalid_options_or_answer"
        elif float(q) > max_query_seconds:
            reason = f"query_time_above_{int(max_query_seconds)}s"
        elif durations.get(r["video"]) is None:
            reason = "video_missing_or_unreadable"
        elif float(q) > durations[r["video"]] + 1.0:
            reason = "query_after_video_end"
        elif not _visible(float(q), step_seconds, gap, min(float(q), durations[r["video"]]), max_seconds):
            reason = "no_complete_step_before_query"
        if reason:
            exclusions[reason] += 1
            excluded_by_task[task_code(r)][reason] += 1
        else:
            kept.append(r)
        seen_ids.add(r["id"])

    by_video = defaultdict(list)
    for r in kept:
        by_video[r["video"]].append(r)
    streams, questions = [], []
    for video in sorted(by_video):
        rows = sorted(by_video[video], key=lambda r: (float(r["query_time"]), r["id"]))
        end = min(durations[video], max(float(r["query_time"]) for r in rows))
        qs = [{"id": r["id"], "time": float(r["query_time"]), "question": r["question"], "options": r["options"],
               "answer": int(r["answer"]), "task": task_code(r), "family": OVO_MCQ_TASKS[task_code(r)], "delay": 0.0}
              for r in rows]  # fmt: skip
        streams.append({"id": stream_id(video), "video": video, "duration": round(end, 3), "split": "dev",
                        "source": "ovo-bench", "protocol": protocol, "questions": qs})  # fmt: skip
        questions += [{**q, "stream": stream_id(video), "video": video, "protocol": protocol} for q in qs]
    write_jsonl(out / "streams.jsonl", streams)
    write_jsonl(out / "questions.jsonl", questions)

    def family_stats(tasks) -> dict:
        rows = [q for q in questions if q["task"] in tasks]
        videos = Counter(q["video"] for q in rows)
        share = max(videos.values()) / len(rows) if rows else None
        eff = (len(rows) ** 2 / sum(c * c for c in videos.values())) if rows else 0.0
        return {"questions": len(rows), "videos": len(videos), "max_video_share": None if share is None else round(share, 4),
                "effective_videos": round(eff, 1), "query_time_s": _quantiles([q["time"] for q in rows]),
                "per_task": {t: {"questions": sum(q["task"] == t for q in rows),
                                 "videos": len({q["video"] for q in rows if q["task"] == t}),
                                 "query_time_s": _quantiles([q["time"] for q in rows if q["task"] == t])}
                             for t in tasks}}  # fmt: skip

    backward, realtime = family_stats(BACKWARD), family_stats(REALTIME)
    problems = []
    if backward["questions"] < MIN_BACKWARD_QUESTIONS:
        problems.append(f"backward questions {backward['questions']} < {MIN_BACKWARD_QUESTIONS}")
    if backward["videos"] < MIN_BACKWARD_VIDEOS:
        problems.append(f"backward videos {backward['videos']} < {MIN_BACKWARD_VIDEOS}")
    if backward["max_video_share"] is not None and backward["max_video_share"] > MAX_VIDEO_SHARE:
        problems.append(f"one video holds {backward['max_video_share']:.1%} of backward questions (> {MAX_VIDEO_SHARE:.0%})")
    in_scope_tasks = Counter(task_code(r) for r in in_scope)
    inventory = {
        "protocol": protocol, "population_rule": population,
        "population": ("all official MC questions on every video of the §13 diagnostic hash half" if population == "diagnostic-half"
                       else "all official MC questions on the 200 videos of the §13 diagnostic manifest")
                      + "; NOT the 475-question manifest",
        "status_of_videos": "external-dataset development evaluation (the 200 sampled videos informed the §13 transition "
                            "decision; the other diagnostic-half videos were never evaluated)",
        "hash_split": {"seed": seed, "fraction": fraction},
        "sampled_200_videos_in_population": len(set(sampled) & set(diag_videos)),
        "sources": {"mc_records": str(mc_records), "mc_records_sha256": _sha(mc_records),
                    "diagnostic": str(diagnostic), "diagnostic_sha256": _sha(diagnostic),
                    "final_frozen": str(final_frozen), "final_frozen_sha256": _sha(final_frozen), "video_root": str(video_root)},
        "diagnostic_videos": len(diag_videos), "diagnostic_videos_sha256": hashlib.sha256("\n".join(diag_videos).encode()).hexdigest(),
        "frozen_final_videos": len(final_videos), "overlap_with_frozen_final": len(overlap),
        "diagnostic_videos_hashing_to_final": len(hashed_final),
        "videos_unreadable": sum(durations.get(v) is None for v in diag_videos),
        "video_duration_s": _quantiles([d for d in durations.values() if d is not None]),
        "protocol_params": {"max_query_seconds": max_query_seconds, "step_seconds": step_seconds, "frame_gap": gap,
                            "stage2_max_seconds": max_seconds},
        "in_scope_questions": len(in_scope), "in_scope_per_task": dict(sorted(in_scope_tasks.items())),
        "excluded": dict(sorted(exclusions.items())),
        "excluded_by_task": {t: dict(sorted(c.items())) for t, c in sorted(excluded_by_task.items())},
        "eligible_questions": len(questions), "streams": len(streams),
        "stream_seconds": _quantiles([s["duration"] for s in streams]),
        "stream_seconds_total": round(sum(s["duration"] for s in streams), 1),
        "backward": backward, "realtime": realtime,
        "adequacy": {"adequate": not problems, "problems": problems,
                     "rule": f">= {MIN_BACKWARD_QUESTIONS} backward questions, >= {MIN_BACKWARD_VIDEOS} backward videos, "
                             f"no video > {MAX_VIDEO_SHARE:.0%} of backward questions"},
        "streams_sha256": _sha(out / "streams.jsonl"), "questions_sha256": _sha(out / "questions.jsonl"),
    }  # fmt: skip
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


# ---------------------------------------------------------------------------- report
def _cluster_bootstrap(stat, videos: list[str], samples: int = 4000, seed: int = 7) -> list[float | None]:
    """Same convention as memory_pilot._stream_bootstrap: 4,000 draws, seed 7, percentile indices 2.5% / 97.5%-1."""
    if not videos:
        return [None, None]
    rng = random.Random(seed)
    draws = sorted(stat(rng.choices(videos, k=len(videos))) for _ in range(samples))
    return [round(draws[int(0.025 * samples)], 3), round(draws[int(0.975 * samples) - 1], 3)]


def check_coverage(preds: dict[str, list[dict]], questions: list[dict]) -> list[str]:
    """Complete, unique, identically-scheduled per-question coverage in every arm."""
    expected = {q["id"]: q for q in questions}
    problems, schedule = [], {}
    for arm in ARMS:
        rows = preds.get(arm)
        if rows is None:
            problems.append(f"{arm}: no predictions")
            continue
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            problems.append(f"{arm}: duplicate question rows")
        missing, extra = set(expected) - set(ids), set(ids) - set(expected)
        if missing or extra:
            problems.append(f"{arm}: {len(missing)} missing, {len(extra)} unexpected questions")
        for r in rows:
            q = expected.get(r["id"])
            if q is None:
                continue
            if r["stream"] != q["stream"] or abs(float(r["t"]) - q["time"]) > 1e-6:
                problems.append(f"{arm}: {r['id']} stream/time differs from the manifest")
                break
            end = r.get("step_window_end")
            if end is None or end > q["time"] + 1e-6:
                problems.append(f"{arm}: {r['id']} answered with frames after the query time (window end {end})")
                break
        schedule[arm] = {r["id"]: (r.get("t_step"), r.get("stream_steps")) for r in rows}
    arms = [a for a in ARMS if a in schedule]
    for arm in arms[1:]:
        if schedule[arm] != schedule[arms[0]]:
            problems.append(f"{arm}: step schedule differs from {arms[0]}")
    return problems


def transfer_report(eval_root: str | Path, questions_path: str | Path, inventory: dict | None = None,
                    samples: int = 4000, seed: int = 7) -> dict:  # fmt: skip
    root = Path(eval_root)
    questions = read_jsonl(questions_path)
    meta = {q["id"]: q for q in questions}
    preds = {a: read_jsonl(root / a / "predictions.jsonl") for a in ARMS if (root / a / "predictions.jsonl").is_file()}
    problems = check_coverage(preds, questions)
    out: dict = {"protocol": questions[0].get("protocol", PROTOCOL) if questions else PROTOCOL, "questions": len(questions),
                 "coverage_problems": problems}
    if problems:
        out["decision"] = {"status": "invalid_coverage", "continue_stage2_scaling": False,
                           "meaning": "missing or invalid coverage: no conclusion about transfer"}  # fmt: skip
        return out
    correct = {a: {r["id"]: int(r["correct"]) for r in rows} for a, rows in preds.items()}
    families = {"backward": BACKWARD, "realtime": REALTIME, "all": BACKWARD + REALTIME}

    def ids_of(tasks):
        return [k for k in meta if meta[k]["task"] in tasks]

    out["accuracy_pct"] = {}
    for arm in ARMS:
        acc = {}
        for name, tasks in families.items():
            ids = ids_of(tasks)
            acc[name] = round(100 * sum(correct[arm][k] for k in ids) / len(ids), 3) if ids else None
        per_task = {t: round(100 * sum(correct[arm][k] for k in ids_of((t,))) / len(ids_of((t,))), 3)
                    for t in BACKWARD + REALTIME if ids_of((t,))}  # fmt: skip
        acc["per_task"] = per_task
        bt = [per_task[t] for t in BACKWARD if t in per_task]
        acc["backward_task_macro"] = round(sum(bt) / len(bt), 3) if bt else None
        out["accuracy_pct"][arm] = acc

    def paired(a: str, b: str) -> dict:
        tallies = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # video -> task -> [sum delta, n]
        for k, q in meta.items():
            cell = tallies[q["video"]][q["task"]]
            cell[0] += correct[a][k] - correct[b][k]
            cell[1] += 1
        videos = sorted(tallies)

        def pooled(tasks):
            def stat(sample):
                num = sum(tallies[v][t][0] for v in sample for t in tasks if t in tallies[v])
                den = sum(tallies[v][t][1] for v in sample for t in tasks if t in tallies[v])
                return 100 * num / den if den else 0.0

            n = sum(tallies[v][t][1] for v in videos for t in tasks if t in tallies[v])
            nv = sum(any(t in tallies[v] for t in tasks) for v in videos)
            return {"n": n, "videos": nv, "delta_pp": round(stat(videos), 3) if n else None,
                    "ci95_pp": _cluster_bootstrap(stat, videos, samples, seed) if n else [None, None]}  # fmt: skip

        def macro(sample):
            deltas = []
            for t in BACKWARD:
                num = sum(tallies[v][t][0] for v in sample if t in tallies[v])
                den = sum(tallies[v][t][1] for v in sample if t in tallies[v])
                if den:
                    deltas.append(100 * num / den)
            return sum(deltas) / len(deltas) if deltas else 0.0

        res = {name: pooled(tasks) for name, tasks in families.items()}
        res["per_task"] = {t: pooled((t,)) for t in BACKWARD + REALTIME if ids_of((t,))}
        res["backward_task_macro"] = {"delta_pp": round(macro(videos), 3),
                                      "ci95_pp": _cluster_bootstrap(macro, videos, samples, seed)}  # fmt: skip
        return res

    out["paired"] = {"s2_fifo_minus_s1_fifo": paired("s2_fifo", "s1_fifo"),
                     "s2_full_minus_s1_full": paired("s2_full", "s1_full"),
                     "s2_fifo_minus_s2_full": paired("s2_fifo", "s2_full")}  # fmt: skip
    kv = {a: round(sum(r["kv_tokens"] for r in preds[a]) / len(preds[a]), 2) for a in ARMS}
    out["kv_tokens_mean"] = kv
    primary = out["paired"]["s2_fifo_minus_s1_fifo"]["backward"]
    delta, (lo, hi) = primary["delta_pp"], primary["ci95_pp"]
    if delta >= 2.0 and lo > 0:
        status, meaning = "continue", "S2-FIFO - S1-FIFO >= +2 pp on pooled backward with CI lower bound > 0"
    elif hi >= 2.0:
        status, meaning = "threshold_not_met_inconclusive", "the CI still includes +2 pp: too uncertain to tell"
    else:
        status, meaning = "threshold_not_met", "the CI excludes +2 pp on this population"
    adequate = None if inventory is None else inventory["adequacy"]["adequate"]
    out["decision"] = {
        "status": status, "continue_stage2_scaling": status == "continue", "meaning": meaning,
        "primary": {"endpoint": "S2-FIFO - S1-FIFO, pooled EPM/ASI/HLD, question-weighted accuracy", **primary},
        "inventory_adequate": adequate,
        "scope": "external-dataset development evaluation; a failed threshold does not establish that the gain can never "
                 "transfer, and backward questions are not 'evidence older than the window' (OVO has no evidence spans)",
        "secondary_no_gate": ["all questions", "real-time (no-harm: CI lower bound > -2 pp)", "S2-full - S1-full",
                              "backward per task and task-macro"],
    }  # fmt: skip
    rt = out["paired"]["s2_fifo_minus_s1_fifo"]["realtime"]
    out["decision"]["realtime_no_harm"] = None if rt["ci95_pp"][0] is None else rt["ci95_pp"][0] > -2.0
    return out


def flow_check(eval_root: str | Path, streams_path: str | Path, arms=("s1_fifo", "s2_fifo")) -> dict:
    """Small GPU flow check on a few streams: complete coverage, no future frames, identical schedule across arms,
    and seconds of wall time per stream second (to estimate the four full evaluations)."""
    streams = read_jsonl(streams_path)
    questions = [{**q, "stream": s["id"], "video": s["video"]} for s in streams for q in s["questions"]]
    preds = {a: read_jsonl(Path(eval_root) / a / "predictions.jsonl") for a in arms
             if (Path(eval_root) / a / "predictions.jsonl").is_file()}  # fmt: skip
    expected = {q["id"] for q in questions}
    problems = []
    for a in arms:
        rows = preds.get(a, [])
        if {r["id"] for r in rows} != expected or len(rows) != len(expected):
            problems.append(f"{a}: {len({r['id'] for r in rows} & expected)}/{len(expected)} questions")
        late = [r["id"] for r in rows if r.get("step_window_end") is None or r["step_window_end"] > float(r["t"]) + 1e-6]
        if late:
            problems.append(f"{a}: {len(late)} questions answered with frames after the query time")
    sched = [{r["id"]: r.get("t_step") for r in preds.get(a, [])} for a in arms]
    if len(sched) > 1 and any(s != sched[0] for s in sched[1:]):
        problems.append("step schedule differs between arms")
    seconds = {}
    for a, rows in preds.items():
        per_stream = {r["stream"]: r.get("stream_seconds_elapsed", 0.0) for r in rows}
        stream_seconds = sum(s["duration"] for s in streams if s["id"] in per_stream)
        seconds[a] = round(sum(per_stream.values()) / stream_seconds, 4) if stream_seconds else None
    return {"streams": len(streams), "questions": len(questions), "problems": problems, "passed": not problems,
            "wall_seconds_per_stream_second_per_gpu": seconds}  # fmt: skip
