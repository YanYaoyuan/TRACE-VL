"""Strict OVO multiple-choice scoring; NOT a full OVO/FAR evaluator.

Recompute correctness from the question manifest, require exact coverage, and
separate task-macro/category-macro metrics from question-weighted accuracy.
This does not certify that an inference implementation used the declared frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

REALTIME = ("OCR", "ACR", "ATR", "STU", "FPD", "OJR")
BACKWARD = ("EPM", "ASI", "HLD")
TASKS = REALTIME + BACKWARD
# Model identity intentionally differs between external models. Equal protocol is
# not equal pretraining, trainable parameters, FLOPs or a causal architecture test.
PROTOCOL_FIELDS = (
    "population_sha256", "annotation_sha256", "split", "frame_manifest_sha256",
    "pixel_policy", "history_policy", "timestamp_policy", "prompt_sha256",
    "answer_mode", "modality", "score_protocol",
)


def _index(rows, label):
    result = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"{label}: missing/duplicate ID {key!r}")
        result[key] = row
    if not result:
        raise ValueError(f"{label}: empty population")
    return result


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def score(questions: list[dict], predictions: list[dict], *, scope: str) -> dict:
    """Questions: id, video, task, options, answer; predictions: id, choice.

    Choice may be null for an unparseable answer (counts incorrect, never dropped).
    All nine task codes are required for the two-category macro. Optional frame
    timestamps are checked against question query_time when either is supplied.
    """
    if scope not in ("development", "heldout", "official"):
        raise ValueError("scope must be development, heldout or official")
    qs, ps = _index(questions, "questions"), _index(predictions, "predictions")
    if qs.keys() != ps.keys():
        raise ValueError(f"coverage mismatch: missing={len(qs.keys()-ps.keys())}, "
                         f"extra={len(ps.keys()-qs.keys())}")
    groups = defaultdict(list)
    correct = {}
    unparsed = 0
    for key, q in qs.items():
        p = ps[key]
        task = q.get("task", "").split("/")[-1]
        if task not in TASKS:
            raise ValueError(f"unsupported MC task {task!r}; FAR requires its own evaluator")
        if not isinstance(q.get("video"), str) or not q["video"]:
            raise ValueError(f"missing source video: {key}")
        opts, gold = q.get("options"), q.get("answer")
        if not isinstance(opts, list) or len(opts) < 2 or type(gold) is not int or not 0 <= gold < len(opts):
            raise ValueError(f"invalid answer/options: {key}")
        if "choice" not in p:
            raise ValueError(f"missing choice: {key}; use null for an unparseable answer")
        choice = p["choice"]
        if choice is not None and (type(choice) is not int or not 0 <= choice < len(opts)):
            raise ValueError(f"invalid choice: {key}")
        unparsed += choice is None
        c = int(choice == gold)
        if "correct" in p and (type(p["correct"]) not in (int, bool) or p["correct"] != c):
            raise ValueError(f"stored correctness disagrees with manifest: {key}")
        for field in ("video", "task"):
            if field in p and p[field] != q[field]:
                raise ValueError(f"prediction {field} disagrees with manifest: {key}")
        if "query_time" in q or "frame_times" in p:
            end, times = q.get("query_time"), p.get("frame_times")
            if not _finite(end) or end < 0 or not isinstance(times, list) or not times:
                raise ValueError(f"incomplete causal frame metadata: {key}")
            if any(not _finite(t) or t < 0 or t > end for t in times) or times != sorted(times):
                raise ValueError(f"invalid/future/nonmonotonic frame time: {key}")
        groups[task].append(c)
        correct[key] = c
    per_task = {t: {"n": len(groups[t]), "accuracy_percent": 100 * sum(groups[t]) / len(groups[t])}
                for t in TASKS if groups[t]}
    missing = [t for t in TASKS if not groups[t]]
    macros = {}
    for name, tasks in (("realtime_task_macro_percent", REALTIME),
                        ("backward_task_macro_percent", BACKWARD)):
        macros[name] = (sum(per_task[t]["accuracy_percent"] for t in tasks) / len(tasks)
                        if all(groups[t] for t in tasks) else None)
    avg = None if missing else sum(macros.values()) / 2
    return {"scope": scope, "questions": len(qs), "videos": len({q["video"] for q in qs.values()}),
            "unparseable_answers": unparsed, "per_task": per_task, **macros,
            "two_category_macro_percent": avg, "missing_tasks": missing,
            "question_weighted_percent": 100 * sum(correct.values()) / len(qs),
            "official_overall": None,
            "note": "MC only. The two-category macro excludes FAR; it is not the three-category OVO "
                    "overall. Matching an aggregation formula alone does not reproduce a paper protocol. "
                    "Scope is supplied by the caller, not certified by this scorer."}


def compare_protocols(left: dict, right: dict) -> dict:
    """Fail closed on missing declarations; return reasons instead of a false fair-comparison flag."""
    missing = [f"{side}.{key}" for side, obj in (("left", left), ("right", right))
               for key in PROTOCOL_FIELDS if key not in obj or obj[key] in (None, "", [], {})]
    mismatches = [key for key in PROTOCOL_FIELDS
                  if key in left and key in right and left[key] != right[key]]
    return {"same_declared_protocol": not missing and not mismatches,
            "missing": missing, "mismatches": mismatches,
            "note": "Metadata check only; verify actual frame logs. Does not imply matched compute/training."}


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scope", choices=("development", "heldout", "official"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(read_jsonl(args.questions), read_jsonl(args.predictions), scope=args.scope)
    result["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (("questions", args.questions), ("predictions", args.predictions))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
