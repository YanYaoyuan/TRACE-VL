"""Stage/hash outside-development media and audit header availability; no QA/model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from trace_vl.reservoir_capacity.supervisor import atomic_json, budget, run_jobs

CSV_SHA = "0121417a71e1dcf367222ae97e9512e3e99725d72e60d50760a050816ccd58da"
DEV_SHA = "33457ec0ea3319fa97d2a9b289b2feadde5c19ed0574afb5ce87dbd94497a757"
SECONDS = 3600
STRICT = {"Action Perception", "Object Perception"}
BROAD = STRICT | {"Attribute Perception", "Spatial Understanding"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def timestamp(text):
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(not re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
        raise ValueError("invalid annotation timestamp")
    return sum(float(p) * 60**i for i, p in enumerate(reversed(parts)))


def sample_number(name):
    for part in reversed(Path(name).parts):
        match = re.fullmatch(r"(?:Real-Time_Visual_Understanding_)?sample_(\d+)"
                             r"(?:_real|_Real-Time_Visual_Understanding)?", Path(part).stem.replace(" ", "_"))
        if match:
            return int(match[1])
    return None


def plan(args):
    annotation = args.nas / "StreamingBench/Real_Time_Visual_Understanding.csv"
    if sha(annotation) != CSV_SHA or sha(args.development) != DEV_SHA:
        raise ValueError("frozen annotation/development manifest changed")
    old = {}
    with args.development.open() as handle:
        for line in handle:
            row = json.loads(line)
            sample = int(row["source_group"].rsplit(":", 1)[1])
            path = Path(row["video"])
            if not path.is_absolute():
                path = args.development.parent / path
            path = path.resolve(strict=True)
            if sample in old and old[sample] != str(path):
                raise ValueError("old source has inconsistent media paths")
            old[sample] = str(path)
    if set(old) != set(range(1, 251)):
        raise ValueError("expected exactly the original development sources1-250")
    queries = defaultdict(list)
    with annotation.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            match = re.fullmatch(r"Real-Time Visual Understanding_sample_(\d+)_\d+", row["question_id"])
            if not match:
                raise ValueError("unrecognized source ID")
            sample = int(match[1])
            if sample not in old:
                # Deliberately do not access question/options/answer/frames_required.
                queries[sample].append({"id": row["question_id"], "task": row["task_type"],
                                        "cutoff": timestamp(row["time_stamp"]), "clue": row["temporal_clue_type"]})
    if set(queries) != set(range(251, 501)) or sum(map(len, queries.values())) != 1250:
        raise ValueError("expected the complete250 outside-development candidates/1250 metadata rows")
    members, archives = defaultdict(list), {}
    for lo in range(251, 501, 50):
        path = args.nas / f"Real-Time Visual Understanding_{lo}-{lo+49}.zip"
        stat = path.stat()
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                name = Path(item.filename)
                if (item.is_dir() or name.suffix.lower() != ".mp4" or "__MACOSX" in name.parts
                        or name.name.startswith("._")):
                    continue
                sample = sample_number(item.filename)
                if sample in queries:
                    members[sample].append({"archive": str(path), "member": item.filename,
                                            "bytes": item.file_size, "crc32": item.CRC})
        archives[str(path)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if any(len(members[s]) != 1 for s in queries):
        raise ValueError("each candidate requires exactly one canonical ZIP media member")
    candidates = [{"sample": sample, **members[sample][0], "queries": queries[sample]} for sample in sorted(queries)]
    total = sum(c["bytes"] for c in candidates)
    if total > 100_000_000_000 or shutil.disk_usage(args.out).free < total + 20 * 2**30:
        raise ValueError("staging exceeds100GB or leaves less than20GiB free")
    training = read(args.training)["videos"]
    hashes = sorted(set(training.values()))
    if len(hashes) != 20370 or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
        raise ValueError("expected the existing20,370 unique training media hashes")
    value = {"protocol": "streaming-candidate-byte-header-audit-v1", "csv_sha256": CSV_SHA,
             "development_sha256": DEV_SHA, "training_manifest_sha256": sha(args.training),
             "training_hashes": hashes, "candidates": candidates, "old_media": old, "archives": archives,
             "staged_bytes_planned": total, "ceiling_seconds": SECONDS,
             "code_sha256": sha(__file__), "supervisor_sha256": sha(Path(sys.modules[budget.__module__].__file__)),
             "protocol_sha256": sha(Path(__file__).parents[1] / "docs/complementary_memory/streaming_candidate_audit.md"),
             "ffprobe_version": subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0],
             "scope": "Byte disjointness and header availability only; no QA, complete PTS or underlying-source independence."}
    value["id"] = identity(value)
    atomic_json(args.out / "inputs.json", value)
    return value


def load(out):
    value = read(out / "inputs.json")
    if value["id"] != identity({k: v for k, v in value.items() if k != "id"}) or value["code_sha256"] != sha(__file__):
        raise ValueError("audit manifest or source changed")
    if value["supervisor_sha256"] != sha(Path(sys.modules[budget.__module__].__file__)):
        raise ValueError("supervisor source changed")
    if value["protocol_sha256"] != sha(Path(__file__).parents[1] / "docs/complementary_memory/streaming_candidate_audit.md"):
        raise ValueError("audit protocol changed")
    for name, expected in value["archives"].items():
        stat = Path(name).stat()
        if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError("source archive changed")
    return value


def probe(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=codec_name,width,height,avg_frame_rate,start_time,duration,nb_frames:format=duration",
               "-of", "json", str(path)]
    value = json.loads(subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=45))
    if len(value.get("streams", [])) != 1:
        raise ValueError("one primary video stream required")
    stream = value["streams"][0]
    duration = stream.get("duration", value.get("format", {}).get("duration"))
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("positive finite header duration required")
    return {"valid": True, "duration_seconds": duration, "duration_scope": "video-stream header if available, else container header; not verified PTS",
            "stream": stream, "container": value.get("format", {})}


def stage_one(candidate, out, probe_fn=probe):
    sample = candidate["sample"]
    if type(sample) is not int or not 251 <= sample <= 500:
        raise ValueError("candidate sample outside frozen range")
    destination = out / "media" / f"sample_{sample}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".mp4.partial")
    if destination.exists() or temp.exists():
        raise RuntimeError("no same-attempt overwrite or deadline-reset resume")
    value, size = hashlib.sha256(), 0
    try:
        with zipfile.ZipFile(candidate["archive"]) as archive:
            member = archive.getinfo(candidate["member"])
            if member.file_size != candidate["bytes"] or member.CRC != candidate["crc32"]:
                raise ValueError("ZIP member metadata changed")
            with archive.open(member) as source, temp.open("xb") as target:
                for chunk in iter(lambda: source.read(8 * 2**20), b""):
                    size += len(chunk)
                    if size > candidate["bytes"]:
                        raise ValueError("ZIP payload exceeded frozen byte budget")
                    value.update(chunk)
                    target.write(chunk)
        if size != candidate["bytes"]:
            raise ValueError("incomplete ZIP payload")
        temp.rename(destination)
        try:
            metadata = probe_fn(destination)
        except (subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
            metadata = {"valid": False, "error_type": type(exc).__name__,
                        "scope": "Header failure is a data finding, not an eligibility pass."}
    finally:
        temp.unlink(missing_ok=True)
    return {"sample": sample, "bytes": size, "sha256": value.hexdigest(), "staged_path": str(destination),
            "zip_crc_verified": True, "header": metadata}


def worker(args):
    value = load(args.out)
    if args.role == "old":
        for sample, name in value["old_media"].items():
            path = Path(name)
            atomic_json(args.out / "old_hashes" / f"{sample}.json",
                        {"sample": int(sample), "sha256": sha(path), "bytes": path.stat().st_size})
        return
    for c in value["candidates"][args.shard::2]:
        row = stage_one(c, args.out)
        atomic_json(args.out / "candidates" / f"{c['sample']}.json", row)
        print(json.dumps({"sample": c["sample"], "bytes": row["bytes"], "completed": True}), flush=True)


def summarize(out):
    value = load(out)
    rows = [read(out / "candidates" / f"{c['sample']}.json") for c in value["candidates"]]
    old = [read(out / "old_hashes" / f"{s}.json") for s in value["old_media"]]
    if len(rows) != 250 or len(old) != 250 or sum(r["bytes"] for r in rows) != value["staged_bytes_planned"]:
        raise ValueError("complete250 new and250 old media audit required")
    for candidate, row in zip(value["candidates"], rows, strict=True):
        if (row["sample"] != candidate["sample"] or row["bytes"] != candidate["bytes"]
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or not row["zip_crc_verified"]
                or Path(row["staged_path"]).stat().st_size != row["bytes"]):
            raise ValueError("candidate record/media identity or byte count diverged")
    if any(row["sample"] != int(sample) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
           for sample, row in zip(value["old_media"], old, strict=True)):
        raise ValueError("old development hash record identity diverged")
    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row["sample"])
    training, development = set(value["training_hashes"]), {r["sha256"] for r in old}
    overlaps = {"training": sorted(r["sample"] for r in rows if r["sha256"] in training),
                "old_development": sorted(r["sample"] for r in rows if r["sha256"] in development)}
    # Remove entire duplicate groups for eligibility counts; do not pick a representative after seeing labels.
    clean = {r["sample"] for r in rows if len(groups[r["sha256"]]) == 1
             and r["sha256"] not in training | development}
    by_sample = {r["sample"]: r for r in rows}
    counts = {}
    for name, tasks in (("strict_action_object", STRICT), ("broader_perception", BROAD)):
        selected = [(c["sample"], q) for c in value["candidates"] if c["sample"] in clean for q in c["queries"]
                    if by_sample[c["sample"]]["header"]["valid"]
                    and q["task"] in tasks and q["clue"] == "Concurrent" and q["cutoff"] >= 60
                    and q["cutoff"] + 300 <= by_sample[c["sample"]]["header"]["duration_seconds"]]
        counts[name] = {"sources_header_eligible": len({s for s, _ in selected}), "metadata_rows": len(selected),
                        "scope": "Header-only upper bound; full PTS and evidence visibility still unverified."}
    report = {"completed": True, "input_id": value["id"], "candidates": len(rows), "old_media_hashed": len(old),
              "staged_bytes": sum(r["bytes"] for r in rows), "byte_overlap": overlaps,
              "duplicate_candidate_groups": [v for v in groups.values() if len(v) > 1],
              "invalid_header_samples": [r["sample"] for r in rows if not r["header"]["valid"]],
              "byte_disjoint_unique_candidates": len(clean), "header_eligibility": counts,
              "quality_evaluation": False, "model_calls": 0, "full_pts_verified": False,
              "underlying_source_independence_verified": False}
    atomic_json(out / "report.json", report)
    return report


def run(args):
    import fcntl

    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (args.out / ".attempt").open("x") as handle:
        handle.write("One bounded attempt; completed staged media remain on /data.\n")
    status = {"stage": "running", "started_unix": time.time(), "ceiling_seconds": SECONDS}
    atomic_json(args.out / "run-status.json", status)
    try:
        with budget(SECONDS) as deadline:
            value = plan(args)
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--out", str(args.out)]
            jobs = [("old-hashes", [*command, "--role", "old"], [])]
            jobs += [(f"stage-{s}", [*command, "--role", "candidates", "--shard", str(s)], []) for s in range(2)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            run_jobs(jobs, env, args.out / "logs", deadline, args.out / "jobs.json")
            report = summarize(args.out)
            status.update(stage="completed", input_id=value["id"], report_sha256=sha(args.out / "report.json"))
    except BaseException as exc:
        status.update(stage="held_or_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["finished_unix"] = time.time()
        atomic_json(args.out / "run-status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "run", "worker"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nas", type=Path, default=Path("/nas-data/vlm/data/StreamingBench"))
    parser.add_argument("--development", type=Path, default=Path("/nas-data/vlm/yyy/tracevl-runs/streaming-vision-a566104-001/streamingbench.jsonl"))
    parser.add_argument("--training", type=Path, default=Path("/data/yyy/tracevl-runs/core-pool-2efc7cc-001/train/plain/run.json"))
    parser.add_argument("--role", choices=("old", "candidates"), default="candidates")
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    {"plan": plan, "run": run, "worker": worker}[args.stage](args)


if __name__ == "__main__":
    main()
