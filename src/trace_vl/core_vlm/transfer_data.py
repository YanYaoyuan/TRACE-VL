"""Frozen external development populations; never train on benchmark questions."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trace_vl.v3.backbone import frame_size
from trace_vl.v3.ovo_transfer import BACKWARD, REALTIME, ensure_media, video_half
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .data import read_split, sha256
from .prepare import aliases, check_space, digest, verify_video, write_json

NEXTQA_VAL_SHA256 = "43198bdef8436b8d64a9b75d846b0987c10cbf94ebf4be325c4a4e54634d66b8"
NEXT_TASKS = {"TN", "TP", "TC", "CW", "CH", "DC", "DL", "DO"}


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def ovo_rows(directory):
    directory = Path(directory)
    inventory = json.loads((directory / "inventory.json").read_text())
    path = directory / "questions.jsonl"
    if sha256(path) != inventory["questions_sha256"]:
        raise ValueError("OVO frozen questions hash differs")
    split = inventory["hash_split"]
    root = Path(inventory["sources"]["video_root"])
    rows = jsonl(path)
    result = []
    for r in rows:
        if (video_half(r["video"], split["seed"], split["fraction"]) != "diagnostic"
                or r["task"] not in (*BACKWARD, *REALTIME)):
            raise ValueError("OVO question outside diagnostic MC population")
        result.append({**r, "source_group": "ovo:" + r["video"], "source": "ovo",
                       "source_aliases": sorted(aliases(r["video"])),
                       "video": str((root / r["video"]).resolve()), "split": "val",
                       "query_time": float(r["time"]), "protocol": "causal-prefix",
                       "observation": {"start": 0., "end": float(r["time"])}})
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("duplicate OVO questions")
    return result


def nextqa_rows(path, *, videos=200, seed=17):
    # Official NExT-QA, NOT NExT-GQA's different subset/schema or test.csv.
    with Path(path).open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    groups = sorted({r["video"] for r in raw}, key=lambda v: digest([seed, v]))
    if len(groups) < videos:
        raise ValueError("insufficient NExT-QA validation videos")
    selected = set(groups[:videos])
    rows = []
    for r in raw:
        if r["video"] not in selected:
            continue
        if not r["video"].isdigit() or r["type"] not in NEXT_TASKS or not 0 <= int(r["answer"]) < 5:
            raise ValueError("invalid official NExT-QA MC row")
        rows.append({"id": f"nextqa-{r['video']}-{r['qid']}", "source": "nextqa",
                     "source_group": "nextqa:" + r["video"], "source_aliases": [r["video"]],
                     "video": r["video"] + ".mp4", "split": "val", "task": r["type"],
                     "question": r["question"], "options": [r[f"a{i}"] for i in range(5)],
                     "answer": int(r["answer"]), "protocol": "offline",
                     "observation": {"start": 0., "end": "video_end"}})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate NExT-QA questions")
    return sorted(rows, key=lambda r: r["id"])


def check_overlap(rows, training, hashes, train_hashes):
    blocked = set()
    for r in training:
        blocked.update(r.get("source_aliases", []))
        blocked.update(aliases(r.get("relative_video", r["video"])))
    overlapping = [r["id"] for r in rows if blocked & set(r["source_aliases"])
                   or hashes[r["video"]] in train_hashes]
    if overlapping:
        raise ValueError(f"benchmark overlaps actual training: {overlapping[:10]}")


def prepare(cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    check_space(out, 30)
    reuse = Path(cfg['reuse_preparation']) if cfg.get('reuse_preparation') else None
    if reuse:
        previous = json.loads((reuse / 'inputs.json').read_text())['config']
        operational = {'data_out', 'reuse_preparation', 'workers', 'archive_workers'}
        if ({k: v for k, v in cfg.items() if k not in operational}
                != {k: v for k, v in previous.items() if k not in operational}):
            raise ValueError('reused preparation has different experiment/data settings')
    base = Path(cfg["baseline_run"])
    baseline_cfg = json.loads((base / "experiment-config.json").read_text())
    manifest = Path(baseline_cfg["data_out"]) / "manifest.jsonl"
    training_run = json.loads((base / "train/grid/run.json").read_text())
    if sha256(manifest) != training_run["manifest_sha256"]:
        raise ValueError("original training manifest changed")
    training = read_split(manifest, "train")
    training_ids = set(training_run["ids"])
    if training_ids != {r["id"] for r in training}:
        raise ValueError("unexpected original training subset")
    next_path = Path(cfg["nextqa_annotation"])
    if sha256(next_path) != NEXTQA_VAL_SHA256:
        raise ValueError("use the pinned official NExT-QA val.csv; hash differs")
    ovo = ovo_rows(cfg["ovo_transfer"])
    nxt = nextqa_rows(next_path, videos=cfg["nextqa_videos"], seed=cfg["population_seed"])
    inputs = {"config": cfg, "training_manifest": sha256(manifest),
              "nextqa_annotation": sha256(next_path), "ovo_questions": digest(ovo),
              "nextqa_selection": digest(nxt), "implementation": sha256(__file__)}
    inputs["id"] = digest(inputs)
    ensure_run_manifest(out / "inputs.json", inputs)
    completion = out / "complete.json"
    if completion.is_file():
        saved = json.loads(completion.read_text())
        if saved["inputs"] != inputs["id"]:
            raise ValueError("prepared external inputs changed")
        for name, fingerprint in saved["manifests"].items():
            if sha256(out / f"{name}.jsonl") != fingerprint:
                raise ValueError("prepared external manifest changed")
        for name, fingerprint in saved.get('audit_files', {}).items():
            if sha256(out / name) != fingerprint:
                raise ValueError('prepared external inventory/exclusions changed')
        for path, fingerprint in saved["videos"].items():
            if sha256(path) != fingerprint:
                raise ValueError("prepared external media changed")
        print("REUSED verified external preparation; no archive rescan", flush=True)
        return {name: out / f"{name}.jsonl" for name in saved["manifests"]}
    # Reuse already extracted NExT-QA media from any explicitly scoped root.
    wanted = {r["video"] for r in nxt}
    found = defaultdict(set)
    roots = [Path(p) for p in cfg["nextqa_roots"] if Path(p).is_dir()]
    if reuse:
        roots.append(reuse / 'nextqa-media/media')
    for root in roots:
        for p in root.rglob("*.mp4"):
            if p.name in wanted:
                found[p.name].add(p.resolve())
    originals = out / "nextqa-originals"
    originals.mkdir(exist_ok=True)
    for name, candidates in found.items():
        if len(candidates) > 1 and len({sha256(p) for p in candidates}) != 1:
            raise ValueError(f"ambiguous NExT-QA original: {name}")
        dest = originals / name
        chosen = min(candidates)
        if dest.is_symlink():
            if dest.resolve() != chosen:
                raise ValueError("NExT-QA media mapping changed")
        elif dest.exists():
            raise ValueError("unexpected original-media file")
        else:
            dest.symlink_to(chosen)
    last_progress = [0.]

    def progress(**status):
        now = time.monotonic()
        if now - last_progress[0] >= 20:
            last_progress[0] = now
            print("MEDIA PREPARE " + json.dumps(status, default=str), flush=True)

    media = ensure_media(sorted(wanted), originals, roots, out / "nextqa-media",
                         archive_workers=cfg["archive_workers"], progress=progress)
    write_json(out / "media.json", media)
    if any(media[k] for k in ("still_missing", "ambiguous", "extracted_but_unreadable", "archives_skipped")):
        raise ValueError(f"NExT-QA media incomplete: inspect {out / 'media.json'}; no GPU work")
    for r in nxt:
        r["video"] = str((out / "nextqa-media/media" / r["video"]).resolve())
    all_rows = ovo + nxt
    paths = sorted({r["video"] for r in all_rows})
    cache = out / "verified"
    cache.mkdir(exist_ok=True)

    def verify(path):
        key = cache / (digest(path) + ".json")
        fingerprint = sha256(path)
        candidates = [key] + ([reuse / 'verified' / key.name] if reuse else [])
        for candidate in candidates:
            if candidate.is_file():
                cached = json.loads(candidate.read_text())
                if cached.get('sha256') == fingerprint and 'minimum_frames_end' in cached:
                    if candidate != key:
                        write_json(key, cached)
                    return path, cached
        info = verify_video(path, 2 * cfg["steps"])
        write_json(key, info)
        return path, info

    print(f"VERIFY {len(paths)} external videos, {cfg['workers']} CPU workers", flush=True)
    with ThreadPoolExecutor(cfg["workers"]) as executor:
        verified = {}
        for i, (path, info) in enumerate(executor.map(verify, paths)):
            verified[path] = info
            if (i + 1) % 20 == 0:
                print(f"VERIFIED MEDIA {i + 1}/{len(paths)}", flush=True)
    exclusions = []
    for r in all_rows:
        info = verified[r["video"]]
        if r.get("query_time") is not None and r["query_time"] < info["minimum_frames_end"]:
            exclusions.append({'id': r['id'], 'source': r['source'], 'task': r['task'],
                               'video': r['video'], 'query_time': r['query_time'],
                               'required_distinct_frames': 2 * cfg['steps'],
                               'minimum_frames_end': info['minimum_frames_end'],
                               'reason': 'insufficient_distinct_causal_frames'})
            continue
        h, w = frame_size(info["height"], info["width"], cfg["pixels"], 32)
        if h // 32 < cfg["pool_grid"][0] or w // 32 < cfg["pool_grid"][1]:
            raise ValueError(f"visual grid too small for fixed pooling: {r['id']}")
    hashes = {p: v["sha256"] for p, v in verified.items()}
    check_overlap(all_rows, training, hashes, set(training_run["videos"].values()))
    excluded_ids = {r['id'] for r in exclusions}
    write_json(out / 'exclusions.json', {'rule': 'distinct-causal-frames-v1',
               'required_frames': 2 * cfg['steps'], 'excluded': exclusions,
               'note': 'Metadata-only selection before model scores; shared by every arm; no replacement questions.'})
    original_counts = {'ovo': len(ovo), 'nextqa': len(nxt)}
    ovo = [r for r in ovo if r['id'] not in excluded_ids]
    nxt = [r for r in nxt if r['id'] not in excluded_ids]
    print(f'CAUSAL INVENTORY: excluded {len(exclusions)} questions; OVO {len(ovo)}, NExT-QA {len(nxt)}; see exclusions.json', flush=True)
    # Exact duplicates are a single bootstrap cluster, even under different IDs.
    for r in all_rows:
        r["source_group"] = "sha256:" + hashes[r["video"]]
    inventory = {}
    for name, rows, main in (("ovo", ovo, BACKWARD), ("nextqa", nxt, ("TN", "TP", "TC"))):
        endpoint = [r for r in rows if r["task"] in main]
        groups = Counter(r["source_group"] for r in endpoint)
        minimum = cfg["minimums"][name]
        problems = []
        if len(endpoint) < minimum["questions"] or len(groups) < minimum["videos"]:
            problems.append("insufficient primary-endpoint questions/videos")
        if not groups or max(groups.values()) / len(endpoint) > .1:
            problems.append("one source exceeds 10% of primary endpoint")
        inventory[name] = {"questions": len(rows), "primary_questions": len(endpoint),
                           "primary_videos": len(groups), "by_task": dict(Counter(r["task"] for r in rows)),
                           "problems": problems}
        inventory[name].update(candidate_questions=original_counts[name],
                               excluded_by_task=dict(Counter(r['task'] for r in exclusions if r['source'] == name)))
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path = out / f"{name}.jsonl"
        if path.exists() and path.read_text() != text:
            raise ValueError("frozen external manifest changed")
        atomic_text(path, text)
    write_json(out / "inventory.json", inventory)
    if any(r["problems"] for r in inventory.values()):
        raise ValueError("external inventory insufficient; no training or evaluation started")
    write_json(out / "complete.json", {"inputs": inputs["id"], "manifests": {
        name: sha256(out / f"{name}.jsonl") for name in inventory},
        'audit_files': {name: sha256(out / name) for name in ('inventory.json', 'exclusions.json')},
        "videos": hashes, "training_overlap": 0,
        "limitation": "known source aliases + exact bytes; not perceptual or pretraining contamination audit"})
    return {name: out / f"{name}.jsonl" for name in inventory}
