"""Freeze development rows and resolve selected StreamingBench clips on disk."""
from __future__ import annotations

import ast
import csv
import json
import math
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from trace_vl.core_vlm.control import select_records
from trace_vl.core_vlm.data import read_split
from trace_vl.core_vlm.prepare import aliases, digest
from trace_vl.v3.run_io import atomic_text

REAL_TASK = "Real-Time Visual Understanding"
MEDIA = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


def _sample_number(path):
    for part in reversed(Path(path).parts):
        name = Path(part).stem.replace(" ", "_")
        value = re.fullmatch(r"(?:Real-Time_Visual_Understanding_)?sample_(\d+)"
                             r"(?:_real|_Real-Time_Visual_Understanding)?", name)
        if value:
            return int(value[1])
    return None


def _timestamp(text):
    terms = text.strip().split(":")
    if not 1 <= len(terms) <= 3 or any(not re.fullmatch(r"\d+(?:\.\d+)?", term) for term in terms):
        raise ValueError(f"invalid StreamingBench timestamp: {text!r}")
    return sum(float(v) * 60**i for i, v in enumerate(reversed(terms)))


def _streaming_rows(annotation):
    prefix = re.compile(r"Real-Time Visual Understanding_sample_(\d+)_\d+")
    with annotation.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            identity = raw["question_id"]
            match = prefix.fullmatch(identity)
            if not match:
                raise ValueError(f"unrecognized StreamingBench question ID: {identity}")
            options = ast.literal_eval(raw["options"])
            if not isinstance(options, list) or not 2 <= len(options) <= 26 or any(
                not isinstance(o, str) or not o.strip() for o in options
            ):
                raise ValueError(f"invalid options for {identity}")
            options = [re.sub(rf"^{chr(65+i)}[.)]\s*", "", o.strip()) for i, o in enumerate(options)]
            label = raw["answer"].strip()
            if len(label) != 1 or not "A" <= label < chr(65 + len(options)):
                raise ValueError(f"invalid correct option for {identity}")
            sample = int(match[1])
            query = _timestamp(raw["time_stamp"])
            yield {"id": identity, "question": raw["question"], "options": options,
                   "answer": ord(label) - 65, "query_time": query, "task": raw["task_type"],
                   "sample": sample, "source_group": f"streamingbench-real:{sample:04d}",
                   "video": f"sample_{sample}.mp4", "split": "val", "source": "streamingbench",
                   "protocol": "causal-prefix", "observation": {"start": 0., "end": query}}


def streaming_population(root, out, limit):
    root, out = Path(root), Path(out)
    annotations = sorted(root.rglob("Real_Time_Visual_Understanding.csv"))
    if len(annotations) != 1:
        raise ValueError(f"expected one StreamingBench Real_Time_Visual_Understanding.csv under {root}")
    rows = list(_streaming_rows(annotations[0]))
    # A time-zero query has no completed two-frame observation. Freeze this
    # exclusion before opening a model or inspecting predictions.
    excluded = [r["id"] for r in rows if r["query_time"] <= 0]
    rows = select_records([r for r in rows if r["query_time"] > 0], limit)
    needed = {r["sample"] for r in rows}
    media, source_names = {}, {}
    extracted = defaultdict(list)
    for item in root.rglob("*"):
        relative = item.relative_to(root)
        if (item.suffix.lower() not in MEDIA or "__MACOSX" in relative.parts
                or item.name.startswith("._") or not item.is_file()):
            continue
        is_realtime = (REAL_TASK in str(relative).replace("_", " ") or "real" in relative.parts[:-1]
                       or item.stem.endswith("_real"))
        sample = _sample_number(relative)
        if is_realtime and sample in needed:
            extracted[sample].append(item.resolve())
    archives = defaultdict(list)
    archives_to_open = sorted(root.rglob(f"{REAL_TASK}*.zip")) if needed - extracted.keys() else []
    for archive_path in archives_to_open:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                if (member.is_dir() or path.suffix.lower() not in MEDIA
                        or "__MACOSX" in path.parts or path.name.startswith("._")):
                    continue
                sample = _sample_number(path)
                if sample in needed:
                    archives[sample].append((archive_path, member.filename))
    out.mkdir(parents=True, exist_ok=True)
    for sample in sorted(needed):
        matches = sorted(set(extracted[sample]))
        if len(matches) > 1:
            raise ValueError(f"several extracted StreamingBench videos for sample_{sample}: {matches}")
        if matches:
            media[sample] = matches[0]
            source_names[sample] = matches[0].name
            continue
        sources = archives[sample]
        if len(sources) != 1:
            raise ValueError(f"expected one StreamingBench ZIP member for sample_{sample}: {sources}")
        archive_path, member = sources[0]
        path = out / f"sample_{sample}{Path(member).suffix.lower()}"
        if not path.is_file():
            temporary = path.with_suffix(".partial")
            with zipfile.ZipFile(archive_path) as archive, archive.open(member) as handle, temporary.open("wb") as dest:
                shutil.copyfileobj(handle, dest)
            temporary.replace(path)
        media[sample] = path.resolve()
        source_names[sample] = Path(member).name
    for row in rows:
        sample = row.pop("sample")
        row["video"] = str(media[sample])
        row["source_aliases"] = sorted(aliases(source_names[sample]))
    return rows, {"annotation": str(annotations[0]), "time_zero_exclusions": excluded,
                  "question_count": len(rows), "video_count": len(media)}


def train_population(manifest, external, clips, seed):
    blocked = set()
    for rows in external.values():
        for row in rows:
            blocked.update(set(row.get("source_aliases", ())) | aliases(row["video"]))
    unique = {}
    for row in read_split(Path(manifest), "train"):
        if row.get("mixture_family") not in (None, "general"):
            continue
        keys = set(row.get("source_aliases", ())) | aliases(row["video"])
        if keys & blocked:
            continue
        group = row.get("source_group", row["video"])
        if group not in unique or row["id"] < unique[group]["id"]:
            unique[group] = row
    groups = sorted(unique, key=lambda g: digest([seed, g]))
    if len(groups) < clips:
        raise ValueError(f"need {clips} training source videos; found {len(groups)}")
    selected = []
    for group in groups[:clips]:
        row = unique[group]
        if not Path(row["video"]).is_file():
            raise ValueError(f"training video is missing: {row['video']}")
        # Use contiguous recent observations ending at the clip end. The QA
        # label and question never enter visual distillation.
        selected.append({**row, "protocol": "offline", "query_time": None,
                         "observation": {"start": 0., "end": "video_end"}})
    return selected


def write_rows(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def validate_population(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate development sample")
    for row in rows:
        if (not math.isfinite(row["query_time"]) or row["query_time"] <= 0
                or not Path(row["video"]).is_file() or "options" not in row):
            raise ValueError(f"invalid development question or missing media: {row['id']}")
