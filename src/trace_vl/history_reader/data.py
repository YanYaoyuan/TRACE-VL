"""Align human-timestamped ActivityNet Captions to the downloaded source media."""
from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trace_vl.core_vlm.data import read_split, sha256
from trace_vl.core_vlm.prepare import aliases, digest, split_for, write_json
from trace_vl.streaming_vision.data import write_rows
from trace_vl.v3.ovo_transfer import ensure_media, probe_duration

ANNOTATION_URL = "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip"
ANNOTATION_SHA256 = "7ed4f4d613d965683fbdbb9f01baad4fe2a439b99fc39cfd2f6fa6cf9201fd2b"
EXTENSIONS = (".mp4", ".mkv", ".webm")
VIDEO_ID = re.compile(r"(?:v_)?([A-Za-z0-9_-]{11})")


def source_id(value):
    match = VIDEO_ID.fullmatch(Path(value).stem)
    return match[1] if match else None


def tasks(raw, source, split, seed, recent, lookback, block):
    """Generate before/after MC items from three ordered human captions."""
    duration = float(raw["duration"])
    if not math.isfinite(duration) or duration < recent + 2 * block:
        return []
    captions = []
    for (start, end), text in zip(raw["timestamps"], raw["sentences"], strict=True):
        start, end, text = float(start), float(end), " ".join(text.split())
        # The official end may equal the labelled video duration.
        if text and 0 <= start < end <= duration + 0.05 and end - start >= 1:
            captions.append((start, end, text))
    captions.sort()
    candidates = []
    for left, anchor, right in zip(captions, captions[1:], captions[2:]):
        if left[1] > anchor[0] + .05 or anchor[1] > right[0] + .05:
            continue
        if len({x[2].casefold() for x in (left, anchor, right)}) != 3:
            continue
        query = max(right[1], min(duration, right[1] + recent + 1))
        for direction, gold, other in (("before", left, right), ("after", right, left)):
            if gold[1] > query - recent - 1 or gold[1] < max(0., query - lookback) + block:
                continue
            item = digest([source, direction, left, anchor, right])
            order = int(digest([seed, item])[:8], 16) % 2
            options = [gold[2], other[2]] if not order else [other[2], gold[2]]
            question = f'Which event happened {direction} the following event? "{anchor[2]}"'
            candidates.append({
                "id": f"actnet-{item[:24]}", "source_group": source, "split": split,
                "task": direction, "question": question, "options": options, "answer": order,
                "evidence": list(gold[:2]), "duration": duration, "query_time": query,
                "observation": {"start": 0., "end": query}, "protocol": "causal-prefix",
            })
    # Select one triple and direction per video. All choices are selected from
    # captions of that video. Labels remain separate from visual routing.
    return sorted(candidates, key=lambda row: digest([seed, row["id"]]))


def populations(annotation, blocked, args):
    split_rows = {"train": {}, "dev": {}}
    with zipfile.ZipFile(annotation) as archive:
        reserved = {source_id(key) for name in ("val_1.json", "val_2.json")
                    for key in json.loads(archive.read(name))}
        for key, content in json.loads(archive.read("train.json")).items():
            source = source_id(key)
            if source is None or source in blocked or source in reserved:
                continue
            # The downloaded LLaVA ActivityNet media belongs to the official
            # training pool. Hold out source IDs before looking at availability.
            split = "dev" if split_for(source, args.seed, .2) == "val" else "train"
            rows = tasks(content, source, split, args.seed, args.recent, args.lookback, args.block)
            if rows:
                split_rows[split][source] = rows[0]
    return split_rows


def local_media(roots, wanted):
    found = defaultdict(set)
    for root in roots:
        for file in Path(root).rglob("*"):
            if file.suffix.lower() in EXTENSIONS and source_id(file) in wanted and file.is_file():
                found[source_id(file)].add(file.resolve())
    return found


def match_media(out, candidates, roots, args):
    wanted = set(candidates["train"]) | set(candidates["dev"])
    local = local_media(roots, wanted)
    originals = out / "media-index"
    originals.mkdir(exist_ok=True)
    ambiguities = {key for key, paths in local.items() if len(paths) > 1}
    for key, paths in local.items():
        if key not in ambiguities:
            source = next(iter(paths))
            dest = originals / ("v_" + key + source.suffix.lower())
            if not dest.exists():
                dest.symlink_to(source)
    archive_roots = [p for p in roots if p.name.endswith("_activitynetqa")]
    if not archive_roots:
        archive_roots = roots
    missing = wanted - local.keys()
    names = [f"v_{key}{suffix}" for key in sorted(missing) for suffix in EXTENSIONS]
    archive_audit = {}
    if names:
        # Emit output near archive boundaries rather than on every member.
        def progress(**state):
            if "archives_finished" in state and not state.get("archives_active"):
                print("ARCHIVES", state["archives_finished"], "/", state["archives_total"], flush=True)
        archive_audit = ensure_media(names, originals, archive_roots, out / "archive-media",
                                     archive_workers=args.archive_workers, progress=progress)
    archived = local_media([out / "archive-media" / "media"], missing)
    local.update(archived)
    ambiguities.update(key for key, paths in archived.items() if len(paths) > 1)
    return {key: next(iter(paths)) for key, paths in local.items() if key not in ambiguities}, {
        "archive": archive_audit, "ambiguous_source_ids": sorted(ambiguities),
    }


def prepare_activitynet(args, root):
    annotation = Path(args.annotation)
    if not args.allow_annotation_fixture and sha256(annotation) != ANNOTATION_SHA256:
        raise ValueError(f"ActivityNet annotation archive hash differs: {annotation}")
    ovo = read_split(Path(args.ovo), "val") if args.ovo else []
    blocked = set()
    for row in ovo:
        for value in [row["video"], *row.get("source_aliases", [])]:
            for alias in aliases(value):
                if (video := source_id(alias)) is not None:
                    blocked.add(video)
    # Exclude all OVO media names when the known external benchmark root is available.
    # Its final question labels and data split are never loaded.
    if args.ovo_media:
        ovo_media = Path(args.ovo_media)
        if not ovo_media.is_dir():
            raise ValueError(f"OVO source-media root missing: {ovo_media}")
        for path in ovo_media.rglob("*"):
            if path.suffix.lower() in EXTENSIONS and (video := source_id(path)) is not None:
                blocked.add(video)
    candidates = populations(annotation, blocked, args)
    paths, audit = match_media(root, candidates, [Path(x) for x in args.media], args)
    reasons = Counter()

    def align(row):
        media = paths.get(row["source_group"])
        if media is None:
            return "missing", None
        duration = probe_duration(media)
        if duration is None:
            return "unreadable", None
        if abs(duration - row["duration"]) > max(1., .015 * row["duration"]):
            return "duration_mismatch", None
        query = min(duration, row["query_time"])
        if row["evidence"][1] > query - args.recent - 1:
            return "evidence_in_recent_window", None
        return "aligned", {**row, "video": str(media.resolve()), "media_duration": duration,
                           "query_time": query, "observation": {"start": 0., "end": query}}

    all_rows = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for split, limit in (("train", args.train_limit), ("dev", args.dev_limit)):
            queue = sorted(candidates[split].values(), key=lambda row: digest([args.seed, row["id"]]))
            rows = []
            for label, aligned in pool.map(align, queue):
                reasons[f"{split}/{label}"] += 1
                if aligned:
                    rows.append(aligned)
            if limit:
                rows = rows[:limit]
            all_rows[split] = rows
    all_rows["ovo"] = ovo
    if {r["source_group"] for r in all_rows["train"]} & {r["source_group"] for r in all_rows["dev"]}:
        raise ValueError("ActivityNet train and dev sources overlap")
    counts = {name: len(rows) for name, rows in all_rows.items()}
    for name, rows in all_rows.items():
        write_rows(root / f"{name}.jsonl", rows)
    report = {"candidates": {k: len(v) for k, v in candidates.items()},
              "selected": counts, "alignment": dict(reasons), "blocked_ovo_source_ids": len(blocked),
              "media": audit, "duration_rule_seconds": "max(1.0, 1.5% of the source annotation duration)",
              "split_rule": "80/20 seeded source-ID split of official train; official val sources reserved",
              "annotation_sha256": sha256(annotation)}
    write_json(root / "data-audit.json", report)
    print("DATA", {**counts, **reasons}, flush=True)
    return counts
