"""Evaluation-first temporal diagnostic (pilot-005 review, data_plan row D / L1): do transition tokens help where
temporal change matters, using EXISTING checkpoints only (no training)?

Design, fixed before any model is evaluated:
- Source: OVO-Bench (independent benchmark; never used for training or for any earlier decision).
- Split: OVO videos are hashed (seed 7) into a **diagnostic half**, used once now, and a **frozen final half**,
  untouched until the paper's final evaluation. The pilot validation set (repeatedly inspected) is not used.
- Categories by task semantics (OVO-Bench task definitions), never by model errors:
    temporal = ASI (action sequence), ACR (action recognition), FPD (future prediction),
               REC (repetition counting), SSR (sequential step)
    static   = OJR (object), ATR (attribute), OCR (text), STU (spatial)
  EPM / HLD / CRR (memory, hallucination, sufficiency) are outside this question and not evaluated here.
- Eligible questions: query time <= `max_query_seconds` (bounds compute; a property of the data, not of models).
  At most `per_task` questions per task, sampled with the fixed seed.
- Arms: native, transition, after, b4 from one run directory, identical budgets (46.9 tokens/step for the three
  compressed arms), whole videos streamed once.

Decision (predeclared; one evaluation, no re-selection, no retraining):
  CONTINUE the transition-token track only if, for transition - b4 on the temporal category,
    delta >= +2 pp AND its 95% video-bootstrap CI lower bound > 0 AND the interaction
    (delta_temporal - delta_static) has a CI lower bound > 0.
  Otherwise STOP transition-token work: B4 is the compression baseline, and the paper follows outcome C.
  after - b4 is reported with the same statistics (secondary, no gate).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .data import read_jsonl, write_jsonl

TEMPORAL = ("ASI", "ACR", "FPD", "REC", "SSR")
STATIC = ("OJR", "ATR", "OCR", "STU")


def task_code(record: dict) -> str:
    return str(record.get("task", "")).split("/")[-1]


def category(record: dict) -> str | None:
    code = task_code(record)
    return "temporal" if code in TEMPORAL else "static" if code in STATIC else None


def video_half(video: str, seed: int = 7, diagnostic_fraction: float = 0.5) -> str:
    digest = hashlib.sha256(f"{seed}:{video}".encode()).hexdigest()
    return "diagnostic" if int(digest[:8], 16) / 0xFFFFFFFF < diagnostic_fraction else "final"


def build_diagnostic(records_path: str, out_dir: str, per_task: int = 100, max_query_seconds: float = 180.0,
                     diagnostic_fraction: float = 0.5, seed: int = 7) -> dict:  # fmt: skip
    """diagnostic.jsonl (used now), final_frozen.jsonl (do not evaluate until the paper), manifest.json."""
    out = Path(out_dir)
    rows = [r for r in read_jsonl(records_path) if category(r) is not None]
    eligible = [r for r in rows if r.get("query_time") is not None and float(r["query_time"]) <= max_query_seconds]
    halves = defaultdict(list)
    for r in eligible:
        halves[video_half(r["video"], seed, diagnostic_fraction)].append(r)
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in sorted(halves["diagnostic"], key=lambda r: r["id"]):
        by_task[task_code(r)].append(r)
    diagnostic = []
    for code in sorted(by_task):
        items = by_task[code]
        diagnostic += items if len(items) <= per_task else rng.sample(items, per_task)
    final = sorted(halves["final"], key=lambda r: r["id"])
    if {r["video"] for r in diagnostic} & {r["video"] for r in final}:
        raise AssertionError("diagnostic and final halves share a video")
    for r in diagnostic + final:
        r["category"] = category(r)
    write_jsonl(out / "diagnostic.jsonl", sorted(diagnostic, key=lambda r: r["id"]))
    write_jsonl(out / "final_frozen.jsonl", final)

    def counts(items):
        c = defaultdict(int)
        for r in items:
            c[task_code(r)] += 1
        return dict(sorted(c.items()))

    manifest = {
        "source": str(records_path), "source_sha256": hashlib.sha256(Path(records_path).read_bytes()).hexdigest(),
        "per_task": per_task, "max_query_seconds": max_query_seconds, "diagnostic_fraction": diagnostic_fraction,
        "seed": seed, "temporal_tasks": TEMPORAL, "static_tasks": STATIC,
        "diagnostic": {"questions": len(diagnostic), "videos": len({r["video"] for r in diagnostic}), "per_task": counts(diagnostic),
                       "stream_seconds_max": max((float(r["query_time"]) for r in diagnostic), default=0.0)},
        "final_frozen": {"questions": len(final), "videos": len({r["video"] for r in final}), "per_task": counts(final),
                         "note": "do not evaluate before the paper's final evaluation"},
        "diagnostic_sha256": hashlib.sha256((out / "diagnostic.jsonl").read_bytes()).hexdigest(),
        "final_frozen_sha256": hashlib.sha256((out / "final_frozen.jsonl").read_bytes()).hexdigest(),
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------- report
def _load(eval_dir: Path) -> dict:
    return {r["id"]: r for p in sorted(Path(eval_dir).glob("predictions.*.jsonl")) for r in read_jsonl(p)}


def _units(a: dict, b: dict | None, ids: list[str]) -> dict:
    """Per video: [sum of correct(a) - correct(b) (or correct(a)), count]."""
    groups = defaultdict(lambda: [0, 0])
    for key in ids:
        r = a[key]
        value = int(r["prediction"] == r["answer"]) - (int(b[key]["prediction"] == b[key]["answer"]) if b else 0)
        g = groups[r["video_key"]]
        g[0] += value
        g[1] += 1
    return dict(groups)


def _bootstrap(stat, videos: list[str], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(stat(rng.choices(videos, k=len(videos))) for _ in range(samples))
    return [round(draws[int(0.025 * samples)], 3), round(draws[int(0.975 * samples) - 1], 3)]


def paired_by_category(preds: dict, a: str, b: str, records: dict, samples: int = 4000, seed: int = 7) -> dict:
    """a - b in pp per category, plus the interaction delta_temporal - delta_static, all by video bootstrap
    (videos resampled jointly across categories, so the interaction CI respects shared videos)."""
    ids = {c: [k for k in preds[a] if records[k]["category"] == c] for c in ("temporal", "static")}
    units = {c: _units(preds[a], preds[b], ids[c]) for c in ids}
    videos = sorted(set(units["temporal"]) | set(units["static"]))

    def delta(c, sample):
        num = sum(units[c][v][0] for v in sample if v in units[c])
        den = sum(units[c][v][1] for v in sample if v in units[c])
        return 100 * num / den if den else 0.0

    out = {}
    for c in ("temporal", "static"):
        out[c] = {"delta_pp": round(delta(c, videos), 3), "ci95_pp": _bootstrap(lambda s, c=c: delta(c, s), videos, samples, seed),
                  "questions": len(ids[c])}  # fmt: skip
    out["interaction"] = {
        "delta_pp": round(delta("temporal", videos) - delta("static", videos), 3),
        "ci95_pp": _bootstrap(lambda s: delta("temporal", s) - delta("static", s), videos, samples, seed),
    }
    return out


def temporal_report(eval_root: str, records_path: str, arms=("native", "transition", "after", "b4")) -> dict:
    records = {r["id"]: r for r in read_jsonl(records_path)}
    preds = {arm: _load(Path(eval_root) / arm) for arm in arms}
    for arm, p in preds.items():
        if set(p) != set(records):
            raise ValueError(f"{arm}: {len(set(p) & set(records))}/{len(records)} questions predicted; rerun the evaluation")
    accuracy = {}
    for arm, p in preds.items():
        accuracy[arm] = {}
        for c in ("temporal", "static"):
            ids = [k for k in p if records[k]["category"] == c]
            accuracy[arm][c] = sum(p[k]["prediction"] == p[k]["answer"] for k in ids) / max(1, len(ids))
        per_task = defaultdict(list)
        for k, r in p.items():
            per_task[task_code(records[k])].append(r["prediction"] == r["answer"])
        accuracy[arm]["per_task"] = {t: round(sum(v) / len(v), 4) for t, v in sorted(per_task.items())}
    pairs = {f"{a}_minus_{b}": paired_by_category(preds, a, b, records)
             for a, b in (("transition", "b4"), ("after", "b4"), ("transition", "after"), ("b4", "native"))}  # fmt: skip
    t = pairs["transition_minus_b4"]
    continue_track = (t["temporal"]["delta_pp"] >= 2.0 and t["temporal"]["ci95_pp"][0] > 0
                      and t["interaction"]["ci95_pp"][0] > 0)  # fmt: skip
    tokens = {arm: sum(r["lm_tokens_per_step"] for r in p.values()) / len(p) for arm, p in preds.items()}
    return {
        "questions": len(records), "videos": len({r["video"] for r in records.values()}),
        "accuracy": accuracy, "paired": pairs, "lm_tokens_per_step": tokens,
        "decision": {"continue_transition_track": continue_track,
                     "rule": "transition - b4 on temporal >= +2 pp, CI lower > 0, and interaction CI lower > 0",
                     "if_false": "stop transition-token work; B4 compression baseline; paper outcome C"},
    }  # fmt: skip
