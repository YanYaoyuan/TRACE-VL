"""Read-only data audit for the A100 host (issue #4 dataset inventory): what is on disk, what is usable now,
what the existing converters can consume, and where training/evaluation sources overlap.

Nothing is extracted, downloaded, converted in place or trained. Converter dry-runs write only into the report
directory. Every walk is bounded (`max_files`) and media probing is sampled (`probe_limit`), so the runtime is
predictable on NAS storage.

Sections:
  manifests   current Stage 1 train/val manifests: counts, unique videos, durations, overlap, missing media
  llava       per LLaVA-Video-178K subset: archives (name/size), annotation files (rows, MC/OE/caption kind,
              unique videos), fraction of referenced videos already extracted under the media root, and MC
              questions usable now that are not in the current manifests
  datasets    per directory under the data root: extension counts, archives, annotation-like files, top level
  converters  dry-runs of the installed converters (OVO-Bench, Video-MME, ShanghaiTech) into the report dir
  overlap     NExT-QA ids in LLaVA's NExT-QA-derived subsets vs the standalone NExTQA splits; monitoring datasets
              (UCF-Crime, XD-Violence, CCTV-Anomaly, CUVA, ShanghaiTech, mmFall) by normalized file name and
              by (size) collisions
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v", ".flv", ".mpg", ".mpeg", ".ts"}
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".rar", ".tar.zst", ".gz")
ANNOTATION_EXT = {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".txt", ".npy", ".mat", ".pkl", ".xlsx"}


def _is_archive(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(s) for s in ARCHIVE_SUFFIXES) or bool(re.search(r"\.(tar\.gz|zip|7z)\.\d+$|\.part\d+$", low))


def walk(root: Path, max_files: int = 2_000_000):
    """(path, size) of regular files under root, symlinks not followed, at most max_files; truncated flag."""
    out, truncated = [], False
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            path = Path(directory) / name
            try:
                st = path.lstat()
            except OSError:
                continue
            if not path.is_symlink():
                out.append((path, st.st_size))
            if len(out) >= max_files:
                return out, True
    return out, truncated


def video_meta(path: Path) -> dict | None:
    """Container metadata via OpenCV (no decoding): duration, fps, frames, size; None if unreadable."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frames <= 0:
            return None
        return {"duration": frames / fps, "fps": fps, "frames": frames, "width": w, "height": h}
    finally:
        cap.release()


def _quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]  # noqa: E731
    return {"n": len(v), "min": v[0], "p10": pick(0.1), "median": pick(0.5), "p90": pick(0.9), "max": v[-1],
            "total_hours": sum(v) / 3600}  # fmt: skip


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


# ---------------------------------------------------------------------------- manifests
def audit_manifests(data_dir: Path, media_root: Path, probe_limit: int = 2000, seed: int = 7) -> dict:
    data_dir, media_root = Path(data_dir), Path(media_root)
    splits = {}
    videos_by_split = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        rows = _read_jsonl(path)
        videos = {r["video"] for r in rows if r.get("video")}
        videos_by_split[path.stem] = videos
        splits[path.stem] = {
            "questions": len(rows), "unique_ids": len({r.get("id") for r in rows}), "unique_videos": len(videos),
            "splits": dict(Counter(r.get("split") for r in rows)), "tasks": dict(Counter(r.get("task") for r in rows)),
            "sources": dict(Counter(r.get("source") for r in rows)),
            "options": dict(Counter(len(r.get("options") or []) for r in rows)),
            "missing_media": sum(not (media_root / v).is_file() for v in videos),
        }  # fmt: skip
    rng = random.Random(seed)
    for name, videos in videos_by_split.items():
        sample = sorted(videos)
        if len(sample) > probe_limit:
            sample = rng.sample(sample, probe_limit)
        metas = [video_meta(media_root / v) for v in sample if (media_root / v).is_file()]
        splits[name]["duration_seconds"] = _quantiles([m["duration"] for m in metas if m])
        splits[name]["unreadable_media"] = sum(m is None for m in metas)
        splits[name]["duration_probe"] = f"{len(sample)} of {len(videos)} videos"
    names = sorted(videos_by_split)
    overlap = {f"{a}&{b}": len(videos_by_split[a] & videos_by_split[b]) for i, a in enumerate(names) for b in names[i + 1 :]}
    return {"data_dir": str(data_dir), "media_root": str(media_root), "files": splits, "video_overlap": overlap,
            "referenced_videos": sorted(set().union(*videos_by_split.values())) if videos_by_split else []}  # fmt: skip


def audit_media_root(media_root: Path, referenced: set[str], probe_limit: int, max_files: int, seed: int = 7) -> dict:
    files, truncated = walk(Path(media_root), max_files)
    videos = [(p, s) for p, s in files if p.suffix.lower() in VIDEO_EXT]
    rel = {str(p.relative_to(media_root)): s for p, s in videos}
    unused = sorted(set(rel) - referenced)
    rng = random.Random(seed)
    sample = rng.sample(unused, min(probe_limit, len(unused)))
    metas = [video_meta(Path(media_root) / v) for v in sample]
    top = Counter(Path(v).parts[0] if len(Path(v).parts) > 1 else "." for v in rel)
    return {"video_files": len(rel), "video_bytes": sum(rel.values()), "referenced_by_manifests": len(set(rel) & referenced),
            "unused_video_files": len(unused), "unused_video_bytes": sum(rel[v] for v in unused),
            "unused_duration_seconds_sampled": _quantiles([m["duration"] for m in metas if m]),
            "unused_probe": f"{len(sample)} of {len(unused)}", "top_level_dirs": dict(top.most_common(20)),
            "walk_truncated": truncated}  # fmt: skip


# ---------------------------------------------------------------------------- LLaVA-Video-178K
def _annotation_kind(name: str) -> str:
    low = name.lower()
    return "mc" if "_mc_" in low or low.endswith("_mc.json") else "oe" if "_oe_" in low else "caption" if "cap" in low else "other"


def audit_llava(llava_root: Path, media_root: Path, manifest_videos: set[str], max_files: int) -> dict:
    llava_root, media_root = Path(llava_root), Path(media_root)
    out = {}
    for subset in sorted(d for d in llava_root.iterdir() if d.is_dir() and not d.name.startswith(".")):
        if subset.resolve() == media_root.resolve():
            continue
        files, truncated = walk(subset, max_files)
        archives = [{"name": str(p.relative_to(subset)), "gb": round(s / 1e9, 3)} for p, s in files if _is_archive(p.name)]
        annotations = []
        for p, s in files:
            if p.suffix.lower() != ".json" or s > 2e9:
                continue
            try:
                items = json.loads(p.read_text())
            except (ValueError, UnicodeDecodeError):
                annotations.append({"name": p.name, "error": "unparseable"})
                continue
            if not isinstance(items, list):
                continue
            videos = {i.get("video") for i in items if isinstance(i, dict) and i.get("video")}
            turns = sum(len(i.get("conversations", [])) // 2 for i in items if isinstance(i, dict))
            extracted = sum((media_root / v).is_file() for v in videos)
            unused = {v for v in videos if (media_root / v).is_file()} - manifest_videos
            annotations.append({"name": p.name, "kind": _annotation_kind(p.name), "items": len(items), "qa_turns": turns,
                                "unique_videos": len(videos), "videos_extracted": extracted,
                                "extracted_videos_not_in_manifests": len(unused)})  # fmt: skip
        out[subset.name] = {"files": len(files), "gb": round(sum(s for _, s in files) / 1e9, 2), "archives": archives,
                            "annotations": annotations, "walk_truncated": truncated}  # fmt: skip
    return out


# ---------------------------------------------------------------------------- generic dataset inventory
def inventory(root: Path, max_files: int) -> dict:
    out = {}
    for ds in sorted(d for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith(".")):
        files, truncated = walk(ds, max_files)
        ext = Counter(("".join(p.suffixes[-2:]).lower() if p.name.lower().endswith(".tar.gz") else p.suffix.lower()) or "<none>"
                      for p, _ in files)  # fmt: skip
        by_ext = defaultdict(int)
        for p, s in files:
            by_ext[p.suffix.lower() or "<none>"] += s
        out[ds.name] = {
            "files": len(files), "gb": round(sum(s for _, s in files) / 1e9, 2), "walk_truncated": truncated,
            "extensions": dict(ext.most_common(15)), "gb_by_extension": {k: round(v / 1e9, 2) for k, v in
                                                                         sorted(by_ext.items(), key=lambda x: -x[1])[:10]},
            "video_files": sum(p.suffix.lower() in VIDEO_EXT for p, _ in files),
            "archives": [{"name": str(p.relative_to(ds)), "gb": round(s / 1e9, 3)} for p, s in files if _is_archive(p.name)][:100],
            "annotation_files": [{"name": str(p.relative_to(ds)), "kb": round(s / 1e3, 1)} for p, s in files
                                 if p.suffix.lower() in ANNOTATION_EXT and s < 5e8][:80],
            "top_level": sorted(x.name + ("/" if x.is_dir() else "") for x in ds.iterdir())[:60],
        }  # fmt: skip
    return out


# ---------------------------------------------------------------------------- converter readiness
def converter_dry_runs(data_root: Path, report_dir: Path) -> dict:
    """Try the installed converters on the installed layouts; outputs go to report_dir/converted only."""
    from .anomaly import convert_anomaly
    from .data import convert_ovo, convert_videomme

    data_root, out = Path(data_root), Path(report_dir) / "converted"
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    ovo = Path(data_root) / "OVO_Bench"
    candidates = sorted(ovo.rglob("ovo_bench*.json")) if ovo.is_dir() else []
    attempts = []
    for ann in candidates[:4]:
        for root in [ann.parent, ovo, ovo / "src_videos"]:
            try:
                r = convert_ovo(str(ann), str(out / "ovo.jsonl"), str(root), forward=True)
                attempts.append({"annotation": str(ann), "video_root": str(root), **r})
            except Exception as error:  # noqa: BLE001 - report, do not stop the audit
                attempts.append({"annotation": str(ann), "video_root": str(root), "error": repr(error)[:300]})
    result["ovo_bench"] = {"annotations_found": [str(c) for c in candidates], "attempts": attempts}
    vmme = data_root / "Video-MME"
    parquets = sorted(vmme.rglob("*.parquet")) if vmme.is_dir() else []
    video_dirs = sorted({str(p.parent) for p in vmme.rglob("*.mp4")})[:5] if vmme.is_dir() else []
    attempts = []
    for pq in parquets[:2]:
        for root in video_dirs[:3] or [str(vmme)]:
            try:
                attempts.append({"parquet": str(pq), "video_root": root,
                                 **convert_videomme(str(pq), str(out / "videomme.jsonl"), root)})  # fmt: skip
            except Exception as error:  # noqa: BLE001
                attempts.append({"parquet": str(pq), "video_root": root, "error": repr(error)[:300]})
    result["video_mme"] = {"parquets": [str(p) for p in parquets], "video_dirs_sample": video_dirs, "attempts": attempts}
    sht = data_root / "ShanghaiTech"
    roots = [p.parent.parent for p in sht.rglob("test_frame_mask")] if sht.is_dir() else []
    attempts = []
    for root in roots[:2]:
        try:
            attempts.append({"root": str(root), **convert_anomaly("shanghaitech", str(root), str(out / "sht.jsonl"))})
        except Exception as error:  # noqa: BLE001
            attempts.append({"root": str(root), "error": repr(error)[:300]})
    result["shanghaitech"] = {"roots_with_test_frame_mask": [str(r) for r in roots], "attempts": attempts}
    result["not_implemented"] = {
        "StreamingBench": "no converter yet", "NExTQA": "no standalone converter yet (LLaVA's NExT-QA subsets are MC)",
        "ucf-crime / XD-Violence / CCTV-Anomaly / CUVA / mmFall": "no converter yet (alert data)",
        "PhysicalAI-SmartSpaces / finevideo / JoyAI-VL-Interaction": "no converter yet",
    }  # fmt: skip
    return result


# ---------------------------------------------------------------------------- overlap
NEXT_ID = re.compile(r"(\d{6,})")


def _nextqa_ids_from_llava(llava_root: Path) -> dict:
    out = {}
    for subset in sorted(Path(llava_root).glob("*nextqa*")):
        ids = set()
        for p in subset.rglob("*.json"):
            try:
                items = json.loads(p.read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            for i in items if isinstance(items, list) else []:
                m = NEXT_ID.search(Path(str(i.get("video", ""))).stem) if isinstance(i, dict) else None
                if m:
                    ids.add(m.group(1))
        out[subset.name] = ids
    return out


def _nextqa_standalone_ids(root: Path) -> dict:
    """video ids per split file (csv/json with a 'video' or 'video_id' field)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() == ".csv":
            try:
                with p.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            key = next((k for k in ("video", "video_id", "vid") if rows and k in rows[0]), None)
            if key:
                out[str(p.relative_to(root))] = {str(r[key]).strip() for r in rows if r.get(key)}
    return out


def _norm_name(path: Path) -> str:
    stem = path.name.lower()
    for ext in VIDEO_EXT:
        stem = stem.removesuffix(ext)
    return re.sub(r"(_x264|_x265|_\d+fps)$", "", stem)


def overlap_report(llava_root: Path, data_root: Path, max_files: int) -> dict:
    report = {}
    llava_ids = _nextqa_ids_from_llava(llava_root)
    standalone = _nextqa_standalone_ids(Path(data_root) / "NExTQA") if (Path(data_root) / "NExTQA").is_dir() else {}
    report["nextqa"] = {
        "llava_subsets": {k: len(v) for k, v in llava_ids.items()},
        "standalone_split_files": {k: len(v) for k, v in standalone.items()},
        "overlap": {f"{a} & {b}": len(ia & ib) for a, ia in llava_ids.items() for b, ib in standalone.items()},
        "note": "ids = 6+ digit video numbers; overlap with a standalone test/val split means LLaVA NExT-QA "
                "training data must not be mixed with standalone NExT-QA evaluation on those ids",
    }  # fmt: skip
    names, sizes = {}, {}
    for ds in ("ucf-crime", "XD-Violence", "CCTV-Anomaly", "CUVA", "ShanghaiTech", "mmFall"):
        root = Path(data_root) / ds
        if not root.is_dir():
            continue
        files, _ = walk(root, max_files)
        vids = [(p, s) for p, s in files if p.suffix.lower() in VIDEO_EXT]
        names[ds] = {_norm_name(p) for p, _ in vids}
        sizes[ds] = {s for _, s in vids if s > 0}
    keys = sorted(names)
    report["monitoring"] = {
        "video_files": {k: len(names[k]) for k in keys},
        "name_overlap": {f"{a} & {b}": len(names[a] & names[b]) for i, a in enumerate(keys) for b in keys[i + 1 :]},
        "exact_size_collisions": {f"{a} & {b}": len(sizes[a] & sizes[b]) for i, a in enumerate(keys) for b in keys[i + 1 :]},
        "examples": {f"{a} & {b}": sorted(names[a] & names[b])[:10] for i, a in enumerate(keys) for b in keys[i + 1 :]
                     if names[a] & names[b]},
        "note": "videos inside archives are not seen; equal byte sizes are candidates for content checks, not proof",
    }  # fmt: skip
    return report


def run_audit(data_dir: Path, llava_root: Path, media_root: Path, data_root: Path, report_dir: Path,
              probe_limit: int = 2000, max_files: int = 2_000_000) -> dict:  # fmt: skip
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report, errors = {}, {}

    def section(name, fn):
        try:
            report[name] = fn()
        except Exception as error:  # noqa: BLE001 - one failing section must not hide the others
            errors[name] = repr(error)[:500]
        (report_dir / "audit.json").write_text(json.dumps({**report, "errors": errors}, indent=2, default=str) + "\n")

    section("manifests", lambda: audit_manifests(data_dir, media_root, probe_limit))
    referenced = set(report.get("manifests", {}).pop("referenced_videos", []))
    section("media_root", lambda: audit_media_root(media_root, referenced, probe_limit, max_files))
    section("llava", lambda: audit_llava(llava_root, media_root, referenced, max_files))
    section("datasets", lambda: inventory(data_root, max_files))
    section("converters", lambda: converter_dry_runs(data_root, report_dir))
    section("overlap", lambda: overlap_report(llava_root, data_root, max_files))
    report["errors"] = errors
    (report_dir / "audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def digest(report: dict) -> str:
    """Short text summary to paste into issue #4 (the full audit.json stays the reference)."""
    lines = ["DATA AUDIT DIGEST (read-only; nothing extracted or trained)"]
    for name, f in (report.get("manifests") or {}).get("files", {}).items():
        d = f.get("duration_seconds") or {}
        lines.append(f"manifest {name}: {f['questions']} questions, {f['unique_videos']} videos, missing {f['missing_media']}, "
                     f"unreadable {f.get('unreadable_media')}, duration median {d.get('median', 0):.1f}s "
                     f"[{d.get('min', 0):.1f}, {d.get('max', 0):.1f}] hours {d.get('total_hours', 0):.2f} ({f.get('duration_probe')})")
    m = report.get("manifests") or {}
    if m.get("video_overlap"):
        lines.append(f"manifest video overlap: {m['video_overlap']}")
    mr = report.get("media_root") or {}
    if mr:
        d = mr.get("unused_duration_seconds_sampled") or {}
        lines.append(f"media root: {mr['video_files']} videos ({mr['video_bytes'] / 1e9:.1f} GB), referenced {mr['referenced_by_manifests']}, "
                     f"UNUSED {mr['unused_video_files']} ({mr['unused_video_bytes'] / 1e9:.1f} GB), unused duration median "
                     f"{d.get('median', 0):.1f}s p90 {d.get('p90', 0):.1f}s ({mr.get('unused_probe')})")
    for subset, s in (report.get("llava") or {}).items():
        ann = s.get("annotations", [])
        mc = [a for a in ann if a.get("kind") == "mc"]
        lines.append(f"llava {subset}: {s['gb']} GB, {len(s['archives'])} archives, annotations "
                     + ", ".join(f"{a['name']}({a.get('kind')}: {a.get('qa_turns')} turns, {a.get('unique_videos')} videos, "
                                 f"extracted {a.get('videos_extracted')}, extracted&unused {a.get('extracted_videos_not_in_manifests')})"
                                 for a in ann if "error" not in a)
                     + (f" | MC files {len(mc)}" if mc else ""))
    for ds, s in (report.get("datasets") or {}).items():
        lines.append(f"data {ds}: {s['gb']} GB, {s['files']} files, {s['video_files']} videos, {len(s['archives'])} archives, "
                     f"top {s['extensions']}{' TRUNCATED' if s['walk_truncated'] else ''}")
    for name, c in (report.get("converters") or {}).items():
        if name == "not_implemented":
            lines.append(f"converters not implemented: {list(c)}")
            continue
        ok = [a for a in c.get("attempts", []) if "error" not in a]
        best = max(ok, key=lambda a: a.get("records", 0), default=None)
        lines.append(f"converter {name}: {len(ok)}/{len(c.get('attempts', []))} attempts ok; best "
                     + (json.dumps({k: v for k, v in best.items() if k not in ('annotation', 'parquet')}) if best else "none"))
    ov = report.get("overlap") or {}
    if ov.get("nextqa"):
        lines.append(f"nextqa: llava {ov['nextqa']['llava_subsets']}, standalone {ov['nextqa']['standalone_split_files']}, "
                     f"overlap {ov['nextqa']['overlap']}")
    if ov.get("monitoring"):
        lines.append(f"monitoring: files {ov['monitoring']['video_files']}, name overlap {ov['monitoring']['name_overlap']}, "
                     f"size collisions {ov['monitoring']['exact_size_collisions']}")
    if report.get("errors"):
        lines.append(f"SECTION ERRORS: {report['errors']}")
    return "\n".join(lines) + "\n"
