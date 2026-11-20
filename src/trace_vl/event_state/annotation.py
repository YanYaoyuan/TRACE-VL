"""P1 step 1: 100-sample annotation feasibility check (docs/event_state/causal_label_protocol.md). CPU only.

Tools for the evidence-time labelling check, nothing for training:
  sample         stratified draw of review items from UCA (official UCF-Crime test videos excluded), written as a
                 review sheet the annotators fill in
  export-prefix  for each item, video clips that END at a candidate time (and a few earlier offsets), so the
                 reviewer confirming `evidence_ready` sees nothing after that moment; no audio, no total duration
  validate       check filled records against the record format
  report         accepted / cannot-determine / excluded rates, evidence_ready - raw start distribution (size AND
                 direction), agreement of the double-annotated items, time spent

UCA format (verified against the pinned UCFCrime_Val.json): {video: {duration, timestamps [[a, b]], sentences [str]}}.
UCA sentences are long multi-clause descriptions and often interpret intent ("seemed to be scolding"), so each item
carries a `rule_draft` the annotator must rewrite into an observable, checkable `rule`; the strata below are
keyword heuristics used ONLY to spread the sample, never as labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RECORD_VERSION = "p1-evidence-record-v1"
DECISIONS = ("accepted", "cannot_determine", "excluded")
NEGATIVE_TYPES = (None, "never", "not_yet", "near_miss")
RULE_SOURCES = ("uca_rewrite", "manual_state")
RULE_SEMANTICS = ("state", "historical")  # historical facts ("someone took the bag") have no valid_end
STRATA = {"simple_action": 35, "state_like": 20, "compound": 25, "negative_not_yet": 10, "negative_mismatch": 10}
DOUBLE_ANNOTATED = 30
PREFIX_OFFSETS = (0.0, -2.0, -4.0)  # the candidate time, and earlier moments to check the label is not late
VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov")

_INTERPRETIVE = re.compile(r"\b(seem|seems|seemed|appear|appears|appeared|probably|maybe|perhaps|intend|intends|"
                           r"intended|want|wants|wanted|owner|own|suspect|suspected|thief|trying to|tried to)\b", re.I)
_STATE = re.compile(r"\b(is|are|was|were|remain|remains|remained|stay|stays|stayed|keep|keeps|kept)\s+"
                    r"(lying|sitting|standing|parked|burning|on fire|open|closed|empty|blocked|unattended|still)\b", re.I)
_COMPOUND = re.compile(r"(\.\s+\S)|\b(and then|then|after that|afterwards|while|before|after)\b", re.I)


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _key(seed: int, *parts) -> str:
    return hashlib.sha256(":".join([str(seed), *map(str, parts)]).encode()).hexdigest()


def load_uca(uca_dir: str | Path) -> list[dict]:
    """One row per UCA sentence from UCFCrime_{Train,Val,Test}.json; fails on any other layout."""
    rows = []
    for split in ("Train", "Val", "Test"):
        path = Path(uca_dir) / f"UCFCrime_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        data = json.loads(path.read_text())
        for video in sorted(data):
            entry = data[video]
            if set(entry) != {"duration", "timestamps", "sentences"} or len(entry["timestamps"]) != len(entry["sentences"]):
                raise ValueError(f"{path}: unexpected UCA entry layout for {video}: {sorted(entry)}")
            for index, ((start, end), sentence) in enumerate(zip(entry["timestamps"], entry["sentences"])):
                rows.append({"video": video, "uca_split": split.lower(), "index": index, "duration": float(entry["duration"]),
                             "raw_start": float(start), "raw_end": float(end), "source_sentence": sentence.strip()})  # fmt: skip
    return rows


def video_stem(name: str) -> str:
    """UCF-Crime list entries look like 'Abuse/Abuse028_x264.mp4'; UCA keys are 'Abuse028_x264'."""
    return Path(name.strip().split()[0]).stem if name.strip() else ""


def official_test_videos(path: str | Path) -> set[str]:
    """Stems of the official UCF-Crime anomaly-detection test list (Anomaly_Test.txt or the temporal annotation
    file): these videos never enter training, calibration or this feasibility sample."""
    stems = {video_stem(line) for line in Path(path).read_text().splitlines()}
    stems.discard("")
    if not stems:
        raise ValueError(f"{path}: no video names found")
    return stems


def category(video: str) -> str:
    return "Normal" if video.startswith("Normal") else re.sub(r"\d.*$", "", video)


def stratum_guess(sentence: str) -> str:
    """Keyword heuristic used only to spread the sample over rule types."""
    if _COMPOUND.search(sentence):
        return "compound"
    return "state_like" if _STATE.search(sentence) else "simple_action"


def rule_draft(sentence: str) -> str:
    """First clause of the description, as a starting point the annotator must rewrite into an observable rule."""
    return re.split(r"(?<=[.!?])\s+", sentence.strip())[0].rstrip(".")


def _valid_span(row: dict) -> bool:
    return 0 <= row["raw_start"] < row["raw_end"] <= row["duration"] + 1.0 and len(row["source_sentence"]) >= 8


def sample_feasibility(uca_rows: list[dict], test_videos: set[str], seed: int = 7, strata: dict | None = None,
                       double: int = DOUBLE_ANNOTATED, min_lead: float = 10.0) -> tuple[list[dict], dict]:  # fmt: skip
    """Review items for the feasibility check, at most one per video, official test videos excluded.

    Negative candidates are only PROPOSALS for the annotator to confirm per rule:
      negative_not_yet   the rule's own video, cut before the event (needs >= `min_lead` s of earlier footage)
      negative_mismatch  the rule paired with a different video of the same category (never / near-miss candidate)
    """
    strata = dict(strata or STRATA)
    pool = [r for r in uca_rows if r["video"] not in test_videos and _valid_span(r)]
    excluded = {"official_test": sum(r["video"] in test_videos for r in uca_rows),
                "invalid_span_or_text": sum(r["video"] not in test_videos and not _valid_span(r) for r in uca_rows)}  # fmt: skip
    pool.sort(key=lambda r: _key(seed, r["video"], r["index"]))
    used: set[str] = set()
    by_category = defaultdict(list)
    for r in pool:
        by_category[category(r["video"])].append(r["video"])
    items = []

    def take(name: str, accept) -> None:
        want = strata.get(name, 0)
        for r in pool:
            if want == 0:
                break
            if r["video"] in used or not accept(r):
                continue
            item = {"record_version": RECORD_VERSION, "id": f"p1f-{len(items):03d}", "stratum": name,
                    "video": r["video"], "review_video": r["video"], "uca_split": r["uca_split"], "uca_index": r["index"],
                    "duration": r["duration"], "raw_start": r["raw_start"], "raw_end": r["raw_end"],
                    "source_sentence": r["source_sentence"], "rule_draft": rule_draft(r["source_sentence"]),
                    "interpretive_wording": bool(_INTERPRETIVE.search(r["source_sentence"])),
                    "rule_source": "uca_rewrite", "negative_candidate": None, "clip_end": None}  # fmt: skip
            if name == "negative_not_yet":
                item.update({"negative_candidate": "not_yet", "clip_end": round(r["raw_start"] - 2.0, 2)})
            elif name == "negative_mismatch":
                others = sorted({v for v in by_category[category(r["video"])] if v != r["video"] and v not in used},
                                key=lambda v: _key(seed, "mismatch", r["video"], v))  # fmt: skip
                if not others:
                    continue
                item.update({"negative_candidate": "never_or_near_miss", "review_video": others[0]})
                used.add(others[0])
            items.append(item)
            used.add(r["video"])
            want -= 1
        strata[name] = want  # what could not be filled

    take("state_like", lambda r: stratum_guess(r["source_sentence"]) == "state_like")
    take("simple_action", lambda r: stratum_guess(r["source_sentence"]) == "simple_action")
    take("compound", lambda r: stratum_guess(r["source_sentence"]) == "compound")
    take("negative_not_yet", lambda r: r["raw_start"] >= min_lead + 2.0)
    take("negative_mismatch", lambda r: True)
    order = sorted(range(len(items)), key=lambda i: _key(seed, "double", items[i]["id"]))
    per_stratum = Counter()
    doubles = set()
    for i in order:  # spread the double-annotated items over the strata
        if len(doubles) >= double:
            break
        s = items[i]["stratum"]
        if per_stratum[s] < max(1, round(double * (STRATA.get(s, 0) / sum(STRATA.values())))):
            doubles.add(i)
            per_stratum[s] += 1
    for i in order:
        if len(doubles) >= min(double, len(items)):
            break
        doubles.add(i)
    for i, item in enumerate(items):
        item["double_annotated"] = i in doubles
    summary = {"seed": seed, "items": len(items), "per_stratum": dict(Counter(i["stratum"] for i in items)),
               "unfilled": {k: v for k, v in strata.items() if v}, "double_annotated": len(doubles),
               "excluded_before_sampling": excluded, "pool_sentences": len(pool),
               "pool_videos": len({r["video"] for r in pool}),
               "interpretive_wording": sum(i["interpretive_wording"] for i in items),
               "note": "strata and interpretive flags are keyword heuristics for spreading the sample, not labels; "
                       "manual state rules are annotated separately and reported separately"}  # fmt: skip
    return items, summary


def blank_record(item: dict, annotator: str = "") -> dict:
    """The record an annotator fills in. Times are seconds in the review video; a time may be a number or an
    interval [lo, hi] when the boundary cannot be pinned down (the interval is excluded from loss and scoring)."""
    return {**{k: item[k] for k in ("record_version", "id", "stratum", "video", "review_video", "duration", "raw_start",
                                    "raw_end", "source_sentence", "rule_draft", "rule_source", "negative_candidate",
                                    "clip_end", "double_annotated")},
            "annotator": annotator, "rule": "", "rule_semantics": "state", "decision": "", "exclusion_reason": "",
            "event_start": None, "candidate_ready": None, "evidence_ready": None, "valid_end": None,
            "uncertain_intervals": [], "evidence_note": "", "negative_type": None,
            "prefix_checked": False, "earlier_checked": False, "seconds_spent": None}  # fmt: skip


def _time(value, duration: float, name: str, problems: list[str]):
    """None, a number, or [lo, hi]; returns (lo, hi) or None."""
    if value is None:
        return None
    pair = value if isinstance(value, list) else [value, value]
    if len(pair) != 2 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in pair):
        problems.append(f"{name} must be a number or [lo, hi]")
        return None
    lo, hi = float(pair[0]), float(pair[1])
    if not 0 <= lo <= hi <= duration + 1.0:
        problems.append(f"{name} {value} outside the video (0..{duration}) or reversed")
        return None
    return lo, hi


def validate_record(rec: dict) -> list[str]:
    """Problems of one filled record (empty list = valid)."""
    problems: list[str] = []
    if rec.get("record_version") != RECORD_VERSION:
        problems.append("record_version mismatch")
    if rec.get("decision") not in DECISIONS:
        problems.append(f"decision must be one of {DECISIONS}")
        return problems
    if not rec.get("annotator"):
        problems.append("annotator missing")
    seconds = rec.get("seconds_spent")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
        problems.append("seconds_spent must be a positive number")
    if rec.get("rule_source") not in RULE_SOURCES:
        problems.append(f"rule_source must be one of {RULE_SOURCES}")
    if rec["decision"] != "accepted":
        if not rec.get("exclusion_reason"):
            problems.append("exclusion_reason is required unless accepted")
        return problems
    duration = float(rec["duration"])
    if not rec.get("rule", "").strip():
        problems.append("rule text missing (rewrite rule_draft into an observable, checkable rule)")
    if rec.get("rule_semantics") not in RULE_SEMANTICS:
        problems.append(f"rule_semantics must be one of {RULE_SEMANTICS}")
    if rec.get("negative_type") not in NEGATIVE_TYPES:
        problems.append(f"negative_type must be one of {NEGATIVE_TYPES}")
    if not rec.get("evidence_note", "").strip():
        problems.append("evidence_note missing")
    for interval in rec.get("uncertain_intervals") or []:
        span = _time(interval, duration, "uncertain interval", problems)
        if span is not None and not span[0] < span[1]:
            problems.append(f"uncertain interval {interval} is empty")
    ready = _time(rec.get("evidence_ready"), duration, "evidence_ready", problems)
    end = _time(rec.get("valid_end"), duration, "valid_end", problems)
    _time(rec.get("event_start"), duration, "event_start", problems)
    if rec.get("negative_type") is not None:
        if rec.get("evidence_ready") is not None or rec.get("valid_end") is not None:
            problems.append("a negative record has no evidence_ready or valid_end")
        return problems
    if ready is None:
        problems.append("an accepted positive record needs evidence_ready")
        return problems
    if not (rec.get("prefix_checked") and rec.get("earlier_checked")):
        problems.append("evidence_ready must be confirmed on the prefix clip and checked against earlier moments")
    if rec.get("rule_semantics") == "historical" and rec.get("valid_end") is not None:
        problems.append("a historical-fact rule has no valid_end")
    if end is not None and end[0] <= ready[1]:
        problems.append("valid_end must be after evidence_ready")
    clip_end = rec.get("clip_end")
    if clip_end is not None and ready[1] > clip_end + 1e-6:
        problems.append("evidence_ready lies after the end of the reviewed clip")
    return problems


def _point(value) -> float | None:
    if value is None:
        return None
    return (value[0] + value[1]) / 2 if isinstance(value, list) else float(value)


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    pick = lambda q: round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 2)  # noqa: E731
    return {"n": len(v), "min": round(v[0], 2), "p10": pick(0.1), "p25": pick(0.25), "median": pick(0.5),
            "p75": pick(0.75), "p90": pick(0.9), "max": round(v[-1], 2)}  # fmt: skip


def feasibility_report(records: list[dict], near_seconds: float = 2.0) -> dict:
    """What step 1 must deliver: decisions and reasons, evidence_ready - raw_start (size and direction, by
    stratum), agreement of double-annotated items, time spent. `records` may hold two annotators per item."""
    invalid = {f"{r.get('id')}/{r.get('annotator')}": p for r in records if (p := validate_record(r))}
    valid = [r for r in records if not validate_record(r)]
    by_item = defaultdict(list)
    for r in valid:
        by_item[r["id"]].append(r)
    first = [sorted(rs, key=lambda r: r["annotator"])[0] for rs in by_item.values()]  # one record per item
    decisions = Counter(r["decision"] for r in first)
    reasons = Counter(r["exclusion_reason"] for r in first if r["decision"] != "accepted")
    positives = [r for r in first if r["decision"] == "accepted" and r.get("negative_type") is None]
    negatives = [r for r in first if r["decision"] == "accepted" and r.get("negative_type") is not None]
    diffs = [(r, _point(r["evidence_ready"]) - r["raw_start"]) for r in positives]

    def direction(rows):
        d = [x for _, x in rows]
        n = max(1, len(d))
        return {"n": len(d), "evidence_minus_raw_start_s": _quantiles(d),
                "abs_within_near_share": round(sum(abs(x) < near_seconds for x in d) / n, 4),
                # raw label EARLIER than the evidence: training on it rewards alerting before the rule is confirmed
                "raw_earlier_than_evidence_share": round(sum(x >= near_seconds for x in d) / n, 4),
                "raw_later_than_evidence_share": round(sum(x <= -near_seconds for x in d) / n, 4)}  # fmt: skip

    by_stratum = defaultdict(list)
    for r, x in diffs:
        by_stratum[r["stratum"]].append((r, x))
    largest = sorted(diffs, key=lambda rx: -abs(rx[1]))[:10]
    pairs = [rs for rs in by_item.values() if len({r["annotator"] for r in rs}) >= 2]
    gaps, same_decision, both_positive = [], 0, 0
    for rs in pairs:
        a, b = sorted(rs, key=lambda r: r["annotator"])[:2]
        same_decision += a["decision"] == b["decision"] and a.get("negative_type") == b.get("negative_type")
        if all(r["decision"] == "accepted" and r.get("negative_type") is None for r in (a, b)):
            both_positive += 1
            gaps.append(abs(_point(a["evidence_ready"]) - _point(b["evidence_ready"])))
    uncertain = sum(hi - lo for r in positives for lo, hi in (r.get("uncertain_intervals") or []))
    interval_ready = sum(isinstance(r["evidence_ready"], list) for r in positives)
    seconds = [r["seconds_spent"] for r in valid]
    return {
        "record_version": RECORD_VERSION, "records": len(records), "invalid_records": invalid, "items": len(first),
        "decisions": dict(decisions), "cannot_determine_share": round(decisions["cannot_determine"] / max(1, len(first)), 4),
        "exclusion_reasons": dict(reasons.most_common()),
        "per_rule_source": dict(Counter(r["rule_source"] for r in first)),
        "positives": len(positives), "negatives": dict(Counter(r["negative_type"] for r in negatives)),
        "valid_end_given": sum(r.get("valid_end") is not None for r in positives),
        "historical_rules": sum(r.get("rule_semantics") == "historical" for r in positives),
        "evidence_ready_as_interval": interval_ready, "uncertain_seconds": round(uncertain, 1),
        "near_seconds": near_seconds, "time_difference": direction(diffs),
        "time_difference_by_stratum": {k: direction(v) for k, v in sorted(by_stratum.items())},
        "largest_differences": [{"id": r["id"], "stratum": r["stratum"], "rule": r["rule"], "difference_s": round(x, 2)}
                                for r, x in largest],
        "agreement": {"double_annotated_items": len(pairs), "same_decision": same_decision,
                      "both_positive": both_positive, "evidence_ready_gap_s": _quantiles(gaps),
                      "gap_within_near_share": round(sum(g < near_seconds for g in gaps) / max(1, len(gaps)), 4)},
        "seconds_spent": {**_quantiles(seconds), "total_hours": round(sum(seconds) / 3600, 2)},
        "reading": "descriptive only: no threshold here decides whether the A/B experiment runs; a small share of "
                   "raw-earlier-than-evidence items can still be the hard cases",
    }  # fmt: skip


# ---------------------------------------------------------------------------- prefix clips
def find_videos(root: str | Path) -> dict[str, list[Path]]:
    index = defaultdict(list)
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            index[path.stem].append(path)
    return index


def export_prefix(video: str | Path, end_seconds: float, out_path: str | Path, start_seconds: float = 0.0) -> dict:
    """Re-encode frames with time in [start, end] into a new clip: no audio, no later frame, and no metadata of
    the source's total length. Returns what was written, including the last frame's time."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"bad frame rate in {video}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".part" + out_path.suffix)
    writer, frames, index, last = None, 0, 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            index += 1
            if t < start_seconds - 1e-9:
                continue
            if t > end_seconds + 1e-9:
                break  # nothing after the candidate moment is ever written
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            frames, last = frames + 1, t
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"no frames of {video} in [{start_seconds}, {end_seconds}]")
    tmp.replace(out_path)
    return {"clip": str(out_path), "frames": frames, "fps": round(fps, 3), "first_time": round(max(start_seconds, 0.0), 3),
            "last_frame_time": round(last, 3), "end_seconds": end_seconds}  # fmt: skip


def prefix_plan(record: dict, offsets=PREFIX_OFFSETS) -> list[dict]:
    """Clip end times for one record: the candidate time (annotator's `candidate_ready`, else the raw start) and
    earlier offsets. Never beyond `clip_end` (negative_not_yet items are cut before the event)."""
    candidate = _point(record.get("candidate_ready"))
    candidate = float(record["raw_start"]) if candidate is None else candidate
    limit = record.get("clip_end")
    ends = []
    for offset in offsets:
        end = candidate + offset
        end = min(end, limit) if limit is not None else end
        if end > 0.5 and round(end, 2) not in ends:
            ends.append(round(end, 2))
    # the file name carries a hash, not the time, so the clip's name does not hint at where the event is
    return [{"id": record["id"], "end_seconds": e, "name": f"{record['id']}-{_key(7, record['id'], e)[:8]}.mp4"} for e in ends]


def export_prefixes(records: list[dict], video_root: str | Path, out_dir: str | Path) -> dict:
    index = find_videos(video_root)
    out_dir = Path(out_dir)
    written, problems = [], []
    for record in records:
        paths = index.get(record["review_video"], [])
        if len(paths) != 1:
            problems.append({"id": record["id"], "video": record["review_video"],
                             "problem": "missing" if not paths else f"ambiguous: {[str(p) for p in paths]}"})  # fmt: skip
            continue
        for plan in prefix_plan(record):
            try:
                info = export_prefix(paths[0], plan["end_seconds"], out_dir / plan["name"])
            except ValueError as error:
                problems.append({"id": record["id"], "video": record["review_video"], "problem": str(error)})
                continue
            written.append({**plan, **info})
    # the key (clip name -> end time) is kept apart from the clips handed to the reviewer
    (out_dir.parent / f"{out_dir.name}-key.json").write_text(json.dumps(written, indent=2) + "\n")
    return {"clips": len(written), "records": len(records), "problems": problems}


# ---------------------------------------------------------------------------- command line
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("sample")
    c.add_argument("--uca-dir", required=True)
    c.add_argument("--official-test-list", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("export-prefix")
    c.add_argument("--records", required=True)
    c.add_argument("--video-root", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser("validate")
    c.add_argument("--records", required=True, nargs="+")
    c = sub.add_parser("report")
    c.add_argument("--records", required=True, nargs="+")
    c.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "sample":
        items, summary = sample_feasibility(load_uca(args.uca_dir), official_test_videos(args.official_test_list), args.seed)
        out = Path(args.output)
        write_jsonl(out / "items.jsonl", items)
        write_jsonl(out / "sheet-annotator-1.jsonl", [blank_record(i) for i in items])
        write_jsonl(out / "sheet-annotator-2.jsonl", [blank_record(i) for i in items if i["double_annotated"]])
        (out / "sample-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    elif args.command == "export-prefix":
        print(json.dumps(export_prefixes(read_jsonl(args.records), args.video_root, args.output), indent=2))
    elif args.command == "validate":
        records = [r for p in args.records for r in read_jsonl(p)]
        problems = {f"{r.get('id')}/{r.get('annotator')}": p for r in records if (p := validate_record(r))}
        print(json.dumps({"records": len(records), "invalid": problems}, indent=2, ensure_ascii=False))
        raise SystemExit(1 if problems else 0)
    else:
        report = feasibility_report([r for p in args.records for r in read_jsonl(p)])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
