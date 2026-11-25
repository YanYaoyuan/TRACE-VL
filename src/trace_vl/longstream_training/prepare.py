"""Stage the existing timestamped ActivityNet queue; no outcome-based selection."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.prepare import aliases, digest, write_json
from trace_vl.v3.run_io import ensure_run_manifest

REFERENCE = Path("/nas-data/vlm/yyy/tracevl-runs/history-reader-3e5cb79-001")
PRIOR = Path("/data/yyy/tracevl-data/core-scale40k-54e161a/manifest.jsonl")


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def validate(row):
    q = float(row["query_time"])
    a, b = row["evidence"]
    if not (q >= 60 and 0 <= a < b <= q - 9 and q <= row["media_duration"]):
        raise ValueError("invalid long-prefix/old-evidence metadata")
    if row["protocol"] != "causal-prefix" or not row["question"].strip():
        raise ValueError("original historical question required")
    if any(k in row for k in ("target_time", "target_cutoff")):
        raise ValueError("hidden temporal address must not enter normal question")


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    prior = rows(args.prior)
    prior_hashes = {r["media_sha256"] for r in prior}
    prior_aliases = {a for r in prior for x in [r["video"], *r.get("source_aliases", [])] for a in aliases(x)}
    selected, excluded = {}, []
    for split, filename in [("train", "train.jsonl"), ("dev", "dev.jsonl")]:
        selected[split] = []
        for row in rows(args.reference / filename):
            try:
                validate(row)
            except ValueError as e:
                excluded.append({"id": row["id"], "split": split, "reason": str(e)})
                continue
            # Keep source folds inherited from the original annotation selection.
            # Exclude exact known source aliases already seen by PlainLoRA.
            if row["source_group"] in prior_aliases:
                excluded.append({"id": row["id"], "split": split, "reason": "known_prior_QA_training_alias"})
                continue
            selected[split].append(row)
    if not selected["train"] or not selected["dev"]:
        raise ValueError("both source-disjoint folds need eligible examples")
    if {r["source_group"] for r in selected["train"]} & {r["source_group"] for r in selected["dev"]}:
        raise ValueError("source folds overlap")
    ordered = [r for split in ("train", "dev") for r in selected[split]]
    spec = {
        "protocol": "matched-longstream-reader-pilot-v1",
        "reference": str(args.reference),
        "reference_manifests": {s: sha256(args.reference / f"{s}.jsonl") for s in ("train", "dev")},
        "prior_manifest_sha256": sha256(args.prior),
        "question_ids": {s: [r["id"] for r in selected[s]] for s in selected},
        "excluded": excluded,
        "minimum_prefix_seconds": 60,
        "recent_exclusion_seconds": 9,
        "selection_uses_model_outputs": False,
        "normal_reader_uses_evidence": False,
        "development_scope": "Previously used ActivityNet development pool; not official or unseen final evaluation.",
    }
    spec["id"] = digest(spec)
    ensure_run_manifest(args.out / "spec.json", spec)

    def stage(row):
        source = Path(row["video"]).resolve(strict=True)
        before = source.stat()
        checksum = sha256(source)
        if checksum in prior_hashes:
            raise ValueError("known prior QA training bytes; stop before changing frozen cohort")
        dest = args.out / "media" / f"{checksum}{source.suffix.lower()}"
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            if shutil.disk_usage(args.out).free < before.st_size + (30 << 30):
                raise ValueError("insufficient staging disk reserve")
            temp = dest.with_suffix(dest.suffix + ".partial")
            try:
                shutil.copyfile(source, temp)
                if sha256(temp) != checksum:
                    raise ValueError("media copy hash mismatch")
                temp.replace(dest)
            finally:
                temp.unlink(missing_ok=True)
        if sha256(dest) != checksum or source.stat() != before:
            raise ValueError("source/destination changed")
        return {**row, "video": str(dest), "media_sha256": checksum}

    with ThreadPoolExecutor(args.workers) as pool:
        staged = list(pool.map(stage, ordered))
    by_id = {r["id"]: r for r in staged}
    if {r["media_sha256"] for r in staged if r["split"] == "train"} & {
        r["media_sha256"] for r in staged if r["split"] == "dev"
    }:
        raise ValueError("train/dev byte-identical media")
    for split, group in selected.items():
        path = args.out / f"{split}.jsonl"
        payload = "".join(json.dumps(by_id[r["id"]], ensure_ascii=False) + "\n" for r in group)
        if path.exists() and path.read_text() != payload:
            raise ValueError("existing staged manifest changed")
        path.write_text(payload)
    write_json(
        args.out / "data-audit.json",
        {
            "completed": True,
            "input_id": spec["id"],
            "counts": {s: len(v) for s, v in selected.items()},
            "excluded": len(excluded),
            "unique_videos": len({r["media_sha256"] for r in staged}),
            "bytes": sum(Path(r["video"]).stat().st_size for r in staged),
            "manifest_sha256": {s: sha256(args.out / f"{s}.jsonl") for s in selected},
            "prefix_seconds": {
                s: [min(r["query_time"] for r in v), max(r["query_time"] for r in v)]
                for s, v in selected.items()
            },
            "supervision": "Original before/after generated QA from human-timestamped captions; evidence metadata for audit only.",
            "scope": spec["development_scope"],
        },
    )
    print(json.dumps({"completed": True, "counts": {s: len(v) for s, v in selected.items()}}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--prior", type=Path, default=PRIOR)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
