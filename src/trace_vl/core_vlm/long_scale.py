"""Expand the audited 20K mixture and run the matched 4-GPU learning curve."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from trace_vl.v3.backbone import frame_size
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from . import batch_benchmark
from .data import read_split, sha256
from .experiment import run_jobs
from .mixed_data import frozen_external
from .mixed_experiment import run as run_training
from .prepare import aliases, digest, protected_holdouts, write_json
from .prepare import prepare as prepare_general

LAYOUTS = {"mb4-serial": (4, 1), "mb4-parallel": (4, 4), "mb8-parallel": (8, 4)}
ADDITIONS = 20_000
SOURCE_CAP = 8_000
UPDATES = 5_000
PER_RANK = 8


def read_json(path):
    return json.loads(path.read_text())


def completed_general(root):
    path = root / "manifest.jsonl"
    complete = read_json(root / "complete.json")
    if sha256(path) != complete["manifest_sha256"]:
        raise ValueError(f"prepared general QA manifest changed: {path}")
    return path


def added_questions(prior, old_paths, fresh, unused, count, seed):
    """Use new source groups first; then consume unused QA from verified training videos."""
    used_ids = {r["id"] for r in prior}
    pairs = {
        digest([old_paths[r["video"]], r["question"], r.get("options")]) for r in prior
    }
    old_bytes = set(old_paths.values())
    old_sources = {k for r in prior for k in (set(r.get("source_aliases", [])) | aliases(r["video"]))}
    additions = []
    for rows, new_sources in ((fresh, True), (unused, False)):
        groups = defaultdict(list)
        for r in rows:
            if "options" not in r or r["split"] != "train":
                continue
            if new_sources and (r["media_sha256"] in old_bytes
                                or bool((set(r["source_aliases"]) | aliases(r["video"])) & old_sources)):
                continue
            groups[r["source_group"]].append(r)
        order = sorted(groups, key=lambda g: digest([seed, g]))
        for r in groups.values():
            r.sort(key=lambda item: digest([seed, item["id"]]))
        for i in range(max(map(len, groups.values()), default=0)):
            for group in order:
                if i >= len(groups[group]):
                    continue
                row = groups[group][i]
                key = digest([row["media_sha256"], row["question"], row["options"]])
                if row["id"] in used_ids or key in pairs:
                    continue
                additions.append({**row, "mixture_family": "general"})
                pairs.add(key)
                used_ids.add(row["id"])
                if len(additions) == count:
                    return additions
    raise ValueError(f"only {len(additions)} additional verified QA available; target {count}")


def prepare(previous, out):
    cfg = read_json(previous / "config.json")
    source = previous / "mixture"
    complete = read_json(source / "complete.json")
    inputs = read_json(source / "inputs.json")
    if complete["inputs"] != inputs["id"] or digest({k: v for k, v in inputs.items() if k != "id"}) != inputs["id"]:
        raise ValueError("20K mixture provenance differs")
    for name, expected in complete["files"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"20K mixture file changed: {name}")
    old_manifest = source / "manifest.jsonl"
    rows = read_split(old_manifest, "train")
    if (Counter(r.get("mixture_family") for r in rows) != {"general": 10_000, "temporal": 9_997}
            or set(complete["videos"]) != {r["video"] for r in rows}):
        raise ValueError("expected the completed 19,997-question baseline")
    old_general = completed_general(previous / "general")
    if sha256(old_general) != inputs["general_manifest"]:
        raise ValueError("general data differs from the baseline selection")
    external_root = Path(cfg["external_data"])
    external, external_receipt = frozen_external(external_root)
    if external_receipt != inputs["external"]:
        raise ValueError("external development population differs from the 20K run")
    for key in ("steps", "pixels", "pool_grid", "seed"):
        if cfg[key] != inputs["config"][key]:
            raise ValueError(f"20K mixture protocol changed: {key}")
    if cfg["gpus"] != [0, 1, 2, 3]:
        raise ValueError("expected the four-GPU baseline configuration")

    saved = out / "mixture"
    saved.mkdir(exist_ok=True)
    training = {**cfg, "mixture_data": str(saved), "workers": 32, "updates": UPDATES,
                "curve_checkpoint": 2500, "save_every": 250,
                "preservation_weight": .5, "reviewed_pairs": None}
    identity = {"config": training, "external": external_receipt,
                "prior_complete": sha256(source / "complete.json"),
                "previous_general": sha256(old_general), "new_source_cap": SOURCE_CAP,
                "new_questions": ADDITIONS, "implementation": sha256(__file__)}
    identity["id"] = digest(identity)
    ensure_run_manifest(saved / "inputs.json", identity)
    if (saved / "complete.json").is_file():
        done = read_json(saved / "complete.json")
        if done["inputs"] != identity["id"] or any(
            sha256(saved / name) != h for name, h in done["files"].items()
        ):
            raise ValueError("expanded mixture manifest changed")
        return training

    config = read_json(previous / "general/run.json")["config"]
    config.update(max_videos=SOURCE_CAP, min_videos={"train": 2, "val": 1},
                  minimum_selected_questions=1, workers=32, archive_workers=8,
                  max_missing_fraction=.002, max_decode_error_fraction=.002,
                  previous_training_manifests=[str(old_manifest)],
                  protected_manifests=[*config.get("protected_manifests", []),
                                       str(old_general), *map(str, external.values())])
    write_json(out / "additional-general-config.json", config)
    print("DATA: recover and verify new LLaVA source videos from existing archives", flush=True)
    novel = out / ".additional-general"
    if not (novel / "complete.json").is_file():
        prepare_general(out / "additional-general-config.json", novel)
    write_json(out / "additional-general-report.json", read_json(novel / "report.json"))
    fresh = read_split(completed_general(novel), "train")
    # Also protect the unused old general pool from the frozen development bytes.
    heldout_bytes = set(external_receipt["videos"].values())
    blocked, heldout_general, _ = protected_holdouts(config["protected_manifests"])
    heldout_bytes.update(heldout_general)
    usable = set()
    for root in (novel, previous / "general"):
        for info in read_json(root / "verification.json")["videos"].values():
            height, width = frame_size(info["height"], info["width"], cfg["pixels"], 32)
            if (info["frames"] >= 2 * cfg["steps"] and height // 32 >= cfg["pool_grid"][0]
                    and width // 32 >= cfg["pool_grid"][1]):
                usable.add(info["sha256"])

    def eligible(row):
        return (row["media_sha256"] in usable and row["media_sha256"] not in heldout_bytes
                and not (set(row["source_aliases"]) | aliases(row["video"])) & blocked)

    extra = added_questions(rows, complete["videos"], filter(eligible, fresh),
                            filter(eligible, read_split(old_general, "train")), ADDITIONS, cfg["seed"])
    combined = sorted(rows + extra, key=lambda row: row["id"])
    hashes = dict(complete["videos"])
    for row in extra:
        if hashes.setdefault(row["video"], row["media_sha256"]) != row["media_sha256"]:
            raise ValueError(f"conflicting training video identity: {row['video']}")
    old_bytes = set(complete["videos"].values())
    new_bytes = set(hashes.values()) - old_bytes
    new_groups = {r["source_group"] for r in extra if r["media_sha256"] in new_bytes}
    if not new_groups:
        raise ValueError("no new source groups survived media and source exclusions")
    write_json(saved / "inventory.json", {
        "questions": len(combined), "new_questions": len(extra), "distinct_media": len(set(hashes.values())),
        "new_distinct_media": len(new_bytes), "new_source_groups": len(new_groups),
        "families": dict(Counter(r["mixture_family"] for r in combined)),
        "added_sources": dict(Counter(r["source"] for r in extra)),
        "source_overlap": 0, "exact_byte_overlap": 0,
        "note": "New source groups first, then unused training QA if needed. Source/byte audit only.",
    })
    write_json(saved / "media.json", {"original_mixture": str(source), "expanded_pool": str(novel),
               "expanded_pool_verification_sha256": sha256(novel / "verification.json")})
    write_json(saved / "annotation-audit.json", {"prior_audit_sha256": sha256(source / "annotation-audit.json"),
               "expanded_manifest_sha256": sha256(novel / "manifest.jsonl"),
               "new_ids_sha256": digest(sorted(r["id"] for r in extra))})
    atomic_text(saved / "manifest.jsonl", "".join(json.dumps(r) + "\n" for r in combined))
    write_json(saved / "complete.json", {"inputs": identity["id"], "videos": hashes,
               "files": {n: sha256(saved / n) for n in
                         ("manifest.jsonl", "inventory.json", "media.json", "annotation-audit.json")}})
    print("DATA READY", (saved / "inventory.json").read_text(), flush=True)
    return training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = prepare(args.previous.resolve(), out)
    write_json(out / "config.json", cfg)
    if args.prepare_only:
        return
    run_jobs([("gpu-check", [sys.executable, "-c",
              'import torch; assert torch.cuda.device_count() == 4; '
              'free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(4)]; '
              f'print(free); assert min(free) >= {cfg["minimum_free_gib"]}'], cfg["gpus"])],
             os.environ, out / "logs")
    plan = out / "long-scale-plan.json"
    batch_path = out / "batch/batch-report.json"
    if plan.is_file():
        batch = read_json(plan)["batch"]
    else:
        batch = batch_benchmark.run(out, out / "batch", steps=10, layouts=LAYOUTS, per_rank=PER_RANK)
    for name in ("microbatch_size", "decode_workers", "accumulation"):
        cfg[name] = batch[name]
    seconds = batch["candidate_seconds"][batch["layout"]]
    result = {"batch": batch, "updates_per_arm": UPDATES, "presentations_per_arm": UPDATES * PER_RANK * 4,
              "projected_two_arm_training_hours": UPDATES * seconds / 3600,
              "batch_report_sha256": sha256(batch_path)}
    result["id"] = digest(result)
    ensure_run_manifest(plan, result)
    write_json(out / "experiment-config.json", cfg)
    print("LONG TRAINING", json.dumps(result), flush=True)
    run_training(cfg, out)


if __name__ == "__main__":
    main()
