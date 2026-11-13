"""Prepare immutable, video-disjoint LLaVA QA manifests from existing downloads.

No downloads, no model calls, no benchmark answer use. Archive scans are scoped
to explicitly selected source directories. Full-frame decoding and exact-byte
hashes are verified; filename/source aliases are not perceptual deduplication.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from trace_vl.event_state.data import benchmark_inventory
from trace_vl.event_state.media_inventory import inventory_archives, signature
from trace_vl.v3.ovo_transfer import _path_match, ensure_media, find_archives
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .data import sha256
from .qa import question_and_answer

VERIFICATION = "full-ffmpeg-decode-and-PTS-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    atomic_text(Path(path), json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def aliases(value):
    """Explicit source IDs and known filename wrappers; no arbitrary stem slicing."""
    value = str(value).replace("\\", "/")
    youtube = re.search(r"(?:[?&]v=|youtu\.be/)([\w-]{11})(?:[?&#/]|$)", value)
    name = Path(value.split("?")[0]).name
    stem = Path(name).stem if Path(name).suffix.lower() in {".mp4", ".mkv", ".avi", ".webm", ".mov"} else name
    result = {stem} if stem else set()
    wrapped = re.fullmatch(r"(?:v_|ytb_)([\w-]{11})", stem)
    scene = re.fullmatch(r"([\w-]{11})-Scene-\d+", stem)
    for match in (youtube, wrapped, scene):
        if match:
            result.add(match[1])
    return result


def source_group(relative, explicit=None):
    keys = aliases(relative)
    if explicit:
        keys.update(aliases(explicit))
    # Known YouTube wrappers identify their parent video, including scene cuts.
    youtube = [x for x in keys if re.fullmatch(r"[\w-]{11}", x)]
    return "source:" + min(youtube or keys), keys


def split_for(group, seed, val_fraction):
    value = int(digest([seed, group])[:16], 16) / 2**64
    return "val" if value < val_fraction else "train"


def parse_pair(question, answer, kind):
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("question/answer must be strings")
    question = question.replace("<image>", "").replace("<video>", "").strip()
    answer = answer.strip()
    if kind == "oe":
        row = {"question": question, "answer_text": answer}
    elif kind == "mc":
        lines, options = [], []
        for line in question.splitlines():
            match = re.fullmatch(r"\s*([A-Z])[.)]\s*(.+?)\s*", line)
            if match:
                if ord(match[1]) - 65 != len(options):
                    raise ValueError("nonconsecutive MC options")
                options.append(match[2])
            elif not options:
                lines.append(line)
            elif line.strip() and not re.match(r"(?:Answer|Please (?:answer|select)|Select|Choose)\b", line.strip(), re.I):
                raise ValueError("ambiguous text after MC options")
        gold = re.fullmatch(r"([A-Z])[.)]?", answer)
        if gold is None:
            raise ValueError("MC answer must be a single option letter")
        row = {"question": "\n".join(lines).strip(), "options": options, "answer": ord(gold[1]) - 65}
    else:
        raise ValueError("unknown QA kind")
    question_and_answer(row)
    return row


def annotation_rows(path, source, kind, blocked, cfg):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError(f"expected LLaVA JSON list: {path}")
    rows, rejected = [], Counter()
    for index, item in enumerate(payload):
        try:
            relative = item["video"]
            if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                    or ".." in Path(relative).parts or "\\" in relative):
                raise ValueError("unsafe video path")
            group, keys = source_group(relative, item.get("original_video_id") or item.get("youtube_id"))
            if any(p.casefold() in relative.casefold() for p in cfg["exclude_source_patterns"]):
                rejected["excluded_source_family"] += 1
                continue
            if keys & blocked:
                rejected["benchmark_source_overlap"] += 1
                continue
            dialogue = item["conversations"]
            # Multi-turn gold history can answer the next question without video.
            # Use only the first explicit pair, documenting discarded later turns.
            if not isinstance(dialogue, list) or len(dialogue) < 2 or len(dialogue) % 2:
                raise ValueError("malformed conversation")
            q, a = dialogue[:2]
            if q["from"] != "human" or a["from"] != "gpt":
                raise ValueError("nonalternating first pair")
            pair = parse_pair(q["value"], a["value"], kind)
            prompt, answer = question_and_answer(pair)
            if len(prompt) > cfg["max_question_chars"] or len(answer) > cfg["max_answer_chars"]:
                rejected["text_length_limit"] += 1
                continue
            rejected["later_dialogue_turns_not_used"] += len(dialogue) // 2 - 1
            rows.append({**pair, "id": "qa-" + digest([group, pair]), "source_group": group,
                         "source": source, "kind": kind, "relative_video": relative,
                         "source_aliases": sorted(keys), "annotation_file": str(Path(path).resolve()),
                         "annotation_row": index, "upstream_id": str(item.get("id", index)),
                         "protocol": "offline", "observation": {"start": 0., "end": "video_end"}})
        except (ValueError, TypeError, KeyError, IndexError) as error:
            rejected["malformed"] += 1
            if cfg.get("fail_on_malformed", True):
                raise ValueError(f"{path} row {index}: {error}") from error
    return rows, dict(rejected)


def unique_records(rows, conflicts=None):
    """Deduplicate within a clip; quarantine all conflicting copies if requested.

    Parent-video grouping is for split ownership. Different scene clips from
    the same parent may legitimately have different answers to the same question.
    """
    unique, questions, bad = {}, defaultdict(list), set()
    for row in rows:
        clip = row.get("media_sha256", row.get("relative_video", row["source_group"]))
        key = digest([row["source_group"], clip, row["question"], row.get("options")])
        questions[key].append(row)
    for key, copies in questions.items():
        if len({r.get("answer_text", r.get("answer")) for r in copies}) > 1:
            if conflicts is None:
                raise ValueError("conflicting answers for one source-video question")
            conflicts.append({"question_key": key, "copies": copies})
            bad.update(r["id"] for r in copies)
    for row in rows:
        if row["id"] not in bad:
            unique.setdefault(row["id"], row)
    return sorted(unique.values(), key=lambda x: x["id"])


def alias_groups(rows):
    """Group transitive explicit aliases before sampling, including scene cuts."""
    parent, seen = {r["source_group"]: r["source_group"] for r in rows}, {}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row in rows:
        for alias in row["source_aliases"]:
            key = root(row["source_group"])
            if alias in seen:
                other = root(seen[alias])
                parent[max(key, other)] = min(key, other)
            seen[alias] = key
    return [with_group(row, root(row["source_group"])) for row in rows]


def with_group(row, group):
    pair = {k: row[k] for k in ("question", "options", "answer", "answer_text") if k in row}
    clip = row.get("media_sha256", row.get("relative_video", group))
    return {**row, "source_group": group, "id": "qa-" + digest([group, clip, pair])}


def choose(rows, cfg):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    groups = sorted(grouped, key=lambda k: digest([cfg["seed"], "selection", k]))[:cfg["max_videos"]]
    selected = []
    for group in groups:
        # Deterministically interleave MC/OE to avoid a file-order bias.
        kinds = {k: sorted([r for r in grouped[group] if r["kind"] == k], key=lambda r: r["id"])
                 for k in ("mc", "oe")}
        interleaved = [values[i] for i in range(max(map(len, kinds.values())))
                       for values in kinds.values() if i < len(values)]
        selected.extend(interleaved[:cfg["max_questions_per_video"]])
    return selected


def verify_video(path, minimum_frames):
    """Full decode with ffmpeg error checking, plus actual PTS and file SHA256."""
    path = Path(path)
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
                            "-show_frames", "-show_entries", "stream=width,height:frame=best_effort_timestamp_time",
                            "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=600)
    meta = json.loads(probe.stdout)
    pts = [float(f["best_effort_timestamp_time"]) for f in meta["frames"]]
    if (len(pts) < minimum_frames or not all(math.isfinite(t) for t in pts)
            or any(b <= a for a, b in zip(pts, pts[1:]))):
        raise ValueError("insufficient frames or nonmonotonic PTS")
    result = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1",
                             "-noautorotate", "-i", str(path), "-map", "0:v:0", "-an", "-vsync", "0",
                             "-threads", "1", "-progress", "pipe:1", "-f", "null", "-"],
                            capture_output=True, text=True, check=True, timeout=600)
    counts = re.findall(r"^frame=(\d+)$", result.stdout, flags=re.M)
    if not counts or int(counts[-1]) != len(pts):
        raise ValueError("full decoder/probe frame counts disagree")
    identity = sha256(path)
    if before != (path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError("media changed during verification")
    return {"sha256": identity, "frames": len(pts), "duration": pts[-1] - pts[0],
            "width": meta["streams"][0]["width"], "height": meta["streams"][0]["height"],
            "bytes": before[0], "mtime_ns": before[1], "verification": VERIFICATION}


def exact_content_groups(rows, verification):
    """Union source groups sharing file bytes before choosing their split."""
    parent = {r["source_group"]: r["source_group"] for r in rows}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    hashes = {}
    for row in rows:
        key, fingerprint = root(row["source_group"]), verification[row["video"]]["sha256"]
        if fingerprint in hashes:
            other = root(hashes[fingerprint])
            parent[max(key, other)] = min(key, other)
        hashes[fingerprint] = key
    return [with_group(row, root(row["source_group"])) for row in rows]


def implementation_identity():
    # All data-preparation dependencies, not just the CLI wrapper.
    package = Path(__file__).resolve().parents[1]
    modules = ["core_vlm/prepare.py", "core_vlm/data.py", "core_vlm/qa.py",
               "event_state/data.py", "event_state/media_inventory.py", "event_state/common.py",
               "v3/ovo_transfer.py", "v3/archive_scan.py", "v3/run_io.py"]
    return {name: sha256(package / name) for name in modules}


def check_space(out, reserve_gb):
    free = shutil.disk_usage(out).free
    if free < reserve_gb * 2**30:
        raise ValueError(f"{out}: {free / 2**30:.1f} GiB free, reserve is {reserve_gb} GiB")


def publish_manifest(out, manifest, completion):
    """Publish completion last; refuse to repair a changed committed dataset."""
    target, record = out / "manifest.jsonl", out / "complete.json"
    if record.exists():
        if json.loads(record.read_text()) != completion:
            raise ValueError("completed dataset identity changed; use a new output directory")
        if not target.is_file() or sha256(target) != completion["manifest_sha256"]:
            raise ValueError("completed manifest is missing or changed; preserve the run for inspection")
        return
    atomic_text(target, manifest)
    ensure_run_manifest(record, completion)


def prepare(config, out):
    cfg = json.loads(Path(config).read_text())
    integer_keys = ("max_videos", "max_questions_per_video", "minimum_frames", "workers", "archive_workers",
                    "max_question_chars", "max_answer_chars")
    if (any(type(cfg[k]) is not int or cfg[k] < 1 for k in integer_keys)
            or not 0 < cfg["val_fraction"] < 1 or type(cfg["seed"]) is not int
            or cfg["workers"] > 64 or cfg["archive_workers"] > 32):
        raise ValueError("invalid data limits")
    if (set(cfg["min_videos"]) != {"train", "val"}
            or any(type(n) is not int or n < 1 for n in cfg["min_videos"].values())
            or not cfg["sources"] or len(set(cfg["sources"])) != len(cfg["sources"])):
        raise ValueError("explicit nonempty sources and positive train/val minimums required")
    reserve = cfg.setdefault("reserve_gb", 30)
    if not isinstance(reserve, (float, int)) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("invalid reserve_gb")
    fraction = cfg.setdefault("max_decode_error_fraction", 0.)
    if not isinstance(fraction, (float, int)) or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("invalid max_decode_error_fraction")
    if not cfg["benchmark_roots"]:
        raise ValueError("explicit benchmark exclusion inventories are required")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    check_space(out, reserve)
    with (out / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(cfg, out)


def _prepare(cfg, out):
    begin = time.monotonic()
    blocked, benchmarks = benchmark_inventory(cfg["benchmark_roots"], required=True)
    blocked = set().union(*(aliases(k) for k in blocked))
    llava = Path(cfg["llava_root"]).resolve()
    rows, annotations, rejects, source_dirs, archives = [], {}, {}, {}, {}
    for source in cfg["sources"]:
        if Path(source).name != source or source in {".", ".."}:
            raise ValueError("source must be a LLaVA subset directory name")
        directory = llava / source
        source_dirs[source] = directory
        if not directory.is_dir():
            raise ValueError(f"source directory missing: {directory}")
        matched = []
        for kind, pattern in (("oe", "*_oe_qa*.json"), ("mc", "*_mc_*.json")):
            for path in sorted(directory.glob(pattern)):
                if path.name.endswith(".trace-vl.json"):
                    continue
                annotations[str(path)] = sha256(path)
                parsed, rejected = annotation_rows(path, source, kind, blocked, cfg)
                rows.extend(parsed)
                rejects[str(path)] = rejected
                matched.append(path)
        if not matched:
            raise ValueError(f"no supported QA annotations in {directory}")
        catalog, skipped = find_archives([directory])
        if skipped:
            raise ValueError(f"incomplete/unsupported archives in {directory}: {skipped}")
        archives[source] = {a["name"]: signature(a) for a in catalog}
    identity = {"protocol": "core-llava-prepare-v1", "config": cfg, "annotations": annotations,
                "benchmarks": benchmarks, "archives": archives, "implementation": implementation_identity()}
    identity["id"] = digest(identity)
    ensure_run_manifest(out / "run.json", identity)
    conflicts = []
    unique = unique_records(alias_groups(rows), conflicts)
    selected = choose(unique, cfg)
    if not selected:
        raise ValueError("no QA left after exclusions")
    write_json(out / "selection.json", {"run": identity["id"], "records": selected, "rejected": rejects,
                                      "rule": "fixed hash order before extraction; no post-decode replacement"})
    media, resolved = {}, {}
    for source, directory in source_dirs.items():
        relative = sorted({r["relative_video"] for r in selected if r["source"] == source})
        if not relative:
            continue
        overlay = out / "sources" / source
        originals = overlay / "originals"
        missing = []
        for video in relative:
            roots = [llava, llava / "media", directory, *map(Path, cfg.get("media_roots", []))]
            candidates = {p.resolve() for p in [root / video for root in roots]
                          if p.is_file()}
            if len(candidates) > 1 and len({sha256(p) for p in candidates}) > 1:
                raise ValueError(f"different originals match {source}/{video}")
            if candidates:
                target = originals / video
                target.parent.mkdir(parents=True, exist_ok=True)
                chosen = min(candidates)
                if target.is_symlink():
                    if target.resolve() != chosen:
                        raise ValueError("original media mapping changed")
                elif target.exists():
                    raise ValueError("unexpected file in original-media overlay")
                else:
                    target.symlink_to(chosen)
            elif not (overlay / "media-extracted" / video).is_file():
                missing.append(video)
        catalog = []
        if missing:
            catalog, members, inventory = inventory_archives([directory], out / "archive-index", cfg["archive_workers"])
            write_json(overlay / "archive-inventory.json", inventory)
            if inventory["errors"] or inventory["skipped"]:
                raise ValueError(f"archive inventory failed for {source}; see {overlay}")
            needed = set()
            for video in missing:
                candidates = members.get(Path(video).name, [])
                exact = [c for c in candidates if _path_match(c["member"], video) == "path"]
                candidates = exact or candidates
                if len(candidates) > 1:
                    raise ValueError(f"ambiguous archive members for {source}/{video}: {candidates}")
                if candidates:
                    needed.add(candidates[0]["archive"])
            catalog = [a for a in catalog if a["name"] in needed]
        check_space(out, cfg["reserve_gb"])
        print(f"MEDIA {source}: {len(relative)} selected paths; {len(catalog)} relevant archives", flush=True)
        result = ensure_media(relative, originals, [directory], overlay,
                              archive_workers=cfg["archive_workers"], archive_catalog=catalog)
        media[source] = result
        for video in relative:
            path = overlay / "media" / video
            if path.is_file():
                resolved[(source, video)] = str(path)
    write_json(out / "media.json", media)
    unavailable = [(r["source"], r["relative_video"]) for r in selected
                   if (r["source"], r["relative_video"]) not in resolved]
    if unavailable:
        write_json(out / "missing.json", sorted(set(unavailable)))
        raise ValueError(f"{len(set(unavailable))} selected videos unavailable; see missing.json, no training started")
    selected = [{**r, "video": resolved[(r["source"], r["relative_video"])]} for r in selected]
    verification, errors = {}, {}
    videos = sorted({r["video"] for r in selected})
    cache = out / "verified"
    cache.mkdir(exist_ok=True)

    def verify(path):
        target = cache / (digest(path) + ".json")
        # Rehash on resume: a same-size/same-mtime replacement must not be reused.
        fingerprint = sha256(path)
        if target.is_file():
            saved = json.loads(target.read_text())
            if (saved.get("verification") == VERIFICATION and saved.get("sha256") == fingerprint
                    and saved.get("frames", 0) >= cfg["minimum_frames"]):
                return saved
        value = verify_video(path, cfg["minimum_frames"])
        write_json(target, value)
        return value

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(verify, p): p for p in videos}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                verification[path] = future.result()
            except Exception as error:
                errors[path] = str(error)
            if i == 1 or i % 25 == 0 or i == len(videos):
                print(f"DECODE {i}/{len(videos)} errors={len(errors)}", flush=True)
    write_json(out / "verification.json", {"videos": verification, "errors": errors})
    if len(errors) / len(videos) > cfg["max_decode_error_fraction"]:
        raise ValueError("selected media decode failure rate exceeds declared maximum; "
                         "see verification.json, no replacement sampling")
    # Apply the declared quality rule to both splits; never refill a failed clip
    # with a more convenient one and never inspect model scores for selection.
    selected = [r for r in selected if r["video"] not in errors]
    selected = [{**r, "media_sha256": verification[r["video"]]["sha256"]} for r in selected]
    selected = unique_records(exact_content_groups(selected, verification), conflicts)
    write_json(out / "qa-conflicts.json", {"policy": "exclude every conflicting copy; no answer is chosen",
                                         "conflicts": conflicts})
    for row in selected:
        row["split"] = split_for(row["source_group"], cfg["seed"], cfg["val_fraction"])
    counts = {split: {"videos": len({r["source_group"] for r in selected if r["split"] == split}),
                      "questions": sum(r["split"] == split for r in selected),
                      "by_kind": dict(Counter(r["kind"] for r in selected if r["split"] == split))}
              for split in ("train", "val")}
    problems = [f"{split} videos {counts[split]['videos']} < {minimum}"
                for split, minimum in cfg["min_videos"].items() if counts[split]["videos"] < minimum]
    # An immutable completed manifest cannot silently change after source media edits.
    manifest = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(selected, key=lambda r: r["id"]))
    completion = {"id": digest([identity["id"], {p: v["sha256"] for p, v in verification.items()}, manifest]),
                  "run": identity["id"], "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}
    report = {"prepared": not problems, "run": identity["id"], "counts": counts, "problems": problems,
              "rejected": rejects, "selected_source_groups": len({r["source_group"] for r in selected}),
              "parsed_questions": len(rows), "unique_questions_before_selection": len(unique),
              "conflicting_questions_excluded": len(conflicts),
              "verified_media_paths": len(verification), "seconds": time.monotonic() - begin,
              "decode_failures_excluded": len(errors), "decode_failure_fraction": len(errors) / len(videos),
              "exclusion_scope": "normalized annotation/media IDs and known source aliases; exact-byte duplicates "
              "are grouped within selected data; no cross-name/transcoded perceptual benchmark audit",
              "dialogue": "first QA pair only; previous gold answers are never put into the prompt",
              "manifest": str(out / "manifest.jsonl")}
    write_json(out / "report.json", report)
    if problems:
        raise ValueError(f"insufficient split counts: {problems}; no training started")
    check_space(out, cfg["reserve_gb"])
    publish_manifest(out, manifest, completion)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    prepare(args.config, args.out)


if __name__ == "__main__":
    main()
