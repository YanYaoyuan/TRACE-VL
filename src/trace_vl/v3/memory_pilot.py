"""Stage-2 memory pilot (experiments.md §14): does LEARNED retention beat mass / FIFO at the same KV budget?

Splits are frozen at source-video level BEFORE streams are stitched; every video the Stage 1 checkpoint saw
(its train and validation manifests) is excluded, so dev/test streams contain only unseen videos.

Comparison design: ONE Stage-2 model (B4 representation, trained with the TKM policy) is evaluated under several
memory policies at the same budget; only the policy changes at evaluation. A second, separate question (does
Stage-2 training help at all?) compares the Stage-1 B4 checkpoint under the heuristic policies.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .data import read_jsonl, write_jsonl

SPLITS = ("train", "dev", "test")


def split_of(video: str, seed: int, fractions: tuple[float, float, float]) -> str:
    u = int(hashlib.sha256(f"stage2:{seed}:{video}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < fractions[0] else "dev" if u < fractions[0] + fractions[1] else "test"


def make_splits(records: list[str], exclude: list[str], out_dir: str, fractions=(0.8, 0.1, 0.1), seed: int = 7,
                max_videos: dict | None = None) -> dict:  # fmt: skip
    """MC records (convert-llava-mc output) -> train/dev/test record files by source video; videos listed in
    `exclude` manifests (Stage 1 train/val) are removed first. Writes <split>.jsonl and manifest.json."""
    out = Path(out_dir)
    excluded = {r["video"] for path in exclude for r in read_jsonl(path)}
    rows = [r for path in records for r in read_jsonl(path) if r.get("video") and r["video"] not in excluded
            and r.get("question") and r.get("options")]  # fmt: skip
    by_split = defaultdict(list)
    for r in rows:
        split = split_of(r["video"], seed, fractions)
        by_split[split].append({**r, "split": split})
    rng = random.Random(seed)
    for split, cap in (max_videos or {}).items():  # seeded per-split video cap (keeps all questions of kept videos)
        pool = sorted({r["video"] for r in by_split[split]})
        if cap is not None and len(pool) > cap:
            kept = set(rng.sample(pool, cap))
            by_split[split] = [r for r in by_split[split] if r["video"] in kept]
    videos = {s: {r["video"] for r in by_split[s]} for s in SPLITS}
    if videos["train"] & videos["dev"] or videos["train"] & videos["test"] or videos["dev"] & videos["test"]:
        raise AssertionError("splits share a source video")
    manifest = {"records": [str(p) for p in records], "excluded_manifests": [str(p) for p in exclude],
                "excluded_videos": len(excluded), "fractions": list(fractions), "seed": seed,
                "max_videos": max_videos or {}, "splits": {}}  # fmt: skip
    for s in SPLITS:
        write_jsonl(out / f"{s}.jsonl", sorted(by_split[s], key=lambda r: r["id"]))
        manifest["splits"][s] = {"questions": len(by_split[s]), "videos": len(videos[s]),
                                 "sha256": hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest()}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------- report
def _median(values: list[float]):
    v = sorted(values)
    return v[len(v) // 2] if v else None


def _rows(eval_dir: Path) -> list[dict]:
    return read_jsonl(Path(eval_dir) / "predictions.jsonl")


def _stream_bootstrap(per_stream: dict, samples: int = 4000, seed: int = 7) -> list[float]:
    """per_stream: stream -> [sum, count]; CI of sum/count (x100) resampling whole streams."""
    keys = sorted(per_stream)
    if not keys:
        return [None, None]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        pick = rng.choices(keys, k=len(keys))
        num, den = sum(per_stream[k][0] for k in pick), sum(per_stream[k][1] for k in pick)
        draws.append(100 * num / den if den else 0.0)
    draws.sort()
    return [round(draws[int(0.025 * samples)], 3), round(draws[int(0.975 * samples) - 1], 3)]


def _accuracy(rows: list[dict], keep) -> dict:
    sel = [r for r in rows if keep(r)]
    per = defaultdict(lambda: [0, 0])
    for r in sel:
        per[r["stream"]][0] += r["correct"]
        per[r["stream"]][1] += 1
    acc = 100 * sum(r["correct"] for r in sel) / len(sel) if sel else None
    return {"n": len(sel), "streams": len(per), "accuracy_pct": None if acc is None else round(acc, 3),
            "ci95_pct": _stream_bootstrap(per)}  # fmt: skip


def _paired(a: list[dict], b: list[dict], keep) -> dict:
    ra, rb = {r["id"]: r for r in a if keep(r)}, {r["id"]: r for r in b}
    per = defaultdict(lambda: [0, 0])
    for key, r in ra.items():
        per[r["stream"]][0] += r["correct"] - rb[key]["correct"]
        per[r["stream"]][1] += 1
    num, den = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"n": den, "delta_pp": round(100 * num / den, 3) if den else None, "ci95_pp": _stream_bootstrap(per)}


def memory_report(eval_root: str, window_seconds: float, primary: str = "s2_tkm",
                  heuristics=("s2_mass", "s2_fifo"), upper: str = "s2_full") -> dict:  # fmt: skip
    root = Path(eval_root)
    runs = {d.name: _rows(d) for d in sorted(root.iterdir()) if (d / "predictions.jsonl").is_file()}
    ids = {name: {r["id"] for r in rows} for name, rows in runs.items()}
    reference = ids.get(primary)
    for name, s in ids.items():
        if reference is not None and s != reference:
            raise ValueError(f"{name}: question set differs from {primary}")
    beyond = lambda r: r.get("evidence_age") is not None and r["evidence_age"] > window_seconds  # noqa: E731
    within = lambda r: r.get("evidence_age") is not None and r["evidence_age"] <= window_seconds  # noqa: E731
    summary = {}
    for name, rows in runs.items():
        n = max(1, len(rows))
        kept = [r["evidence_kept"] for r in rows if beyond(r) and r.get("evidence_kept") is not None]
        streams = {r["stream"]: r for r in rows}
        summary[name] = {
            "all": _accuracy(rows, lambda r: True), "beyond_window": _accuracy(rows, beyond),
            "within_window": _accuracy(rows, within),
            "evidence_kept_beyond_window": round(sum(kept) / len(kept), 4) if kept else None,
            "kv_tokens_mean": round(sum(r["kv_tokens"] for r in rows) / n, 1),
            "kv_mb_mean": round(sum(r.get("kv_mb", 0.0) for r in rows) / n, 3),
            "evicted_tokens_mean": round(sum(r.get("evicted_tokens", 0) for r in rows) / n, 1),
            "streamed_tokens_mean": round(sum(r.get("streamed_tokens", 0) for r in rows) / n, 1),
            "evidence_age_median": _median([r["evidence_age"] for r in rows if r.get("evidence_age") is not None]),
            "seconds_per_stream": round(sum(r.get("stream_seconds_elapsed", 0) for r in streams.values()) / max(1, len(streams)), 2),
        }  # fmt: skip
    pairs = {}
    if primary in runs:
        for other in [*heuristics, upper, f"{primary}_evict8"]:
            if other in runs:
                pairs[f"{primary}_minus_{other}"] = {"beyond_window": _paired(runs[primary], runs[other], beyond),
                                                     "all": _paired(runs[primary], runs[other], lambda r: True)}  # fmt: skip
    decision = None
    if primary in runs and all(h in runs for h in heuristics):
        gains = [pairs[f"{primary}_minus_{h}"]["beyond_window"] for h in heuristics]
        kv = [summary[x]["kv_tokens_mean"] for x in (primary, *heuristics)]
        fifo_kept = summary.get(heuristics[-1], {}).get("evidence_kept_beyond_window")
        pressure = (summary[primary]["evicted_tokens_mean"] > 0 and fifo_kept is not None and fifo_kept <= 0.8
                    and summary[primary]["beyond_window"]["n"] >= 100)  # fmt: skip
        equal_budget = max(kv) <= 1.02 * min(kv)
        supported = pressure and equal_budget and all(g["delta_pp"] is not None and g["delta_pp"] >= 3.0
                                                      and g["ci95_pp"][0] is not None and g["ci95_pp"][0] > 0 for g in gains)
        decision = {"memory_pressure": pressure, "equal_budget": equal_budget, "learned_retention_supported": supported,
                    "rule": "s2_tkm - each heuristic on beyond-window questions >= +3 pp with stream-bootstrap CI lower > 0, "
                            "with memory pressure (evictions, FIFO evidence kept <= 0.8, >= 100 beyond-window questions) "
                            "and equal retained KV (within 2%)",
                    "if_false": "retention is not a contribution: report heuristics; do not scale Stage 2 for retention"}  # fmt: skip
    return {"window_seconds": window_seconds, "runs": summary, "paired": pairs, "decision": decision}
