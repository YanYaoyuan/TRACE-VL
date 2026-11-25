"""Expanded, fixed-cohort natural delayed development diagnostic; no training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream
from . import longstream_delayed as core
from .control import checkpoint_identity
from .data import sha256
from .experiment import run_jobs
from .prepare import digest, write_json
from .train import checked_checkpoint

DATA_ID = "6b84b8a3e7ecaf43fdc1d17d19aa9de09a07f5757f6aefbc616f84b8d0016a9f"
READY_SHA256 = "67a7cf5dfa14e0354295d5627983f36d476f3d844c5685000135fad6599697aa"
OLD_DEVELOPMENT_MANIFEST = Path("/nas-data/vlm/yyy/tracevl-runs/streaming-vision-a566104-001/streamingbench.jsonl")
TASKS = {"Event Understanding", "Action Perception", "Causal Reasoning", "Counting", "Object Perception"}
SCOPE = (
    "Fixed natural delayed development cohort, all34 metadata/availability-selected targets on28 "
    "already-used source groups. Primary26 past-event targets; secondary8 Prospective Reasoning "
    "targets separately reported. Every target has all5 delays0/15/60/180/300s of unchanged original "
    "video, explicitly anchored to original cutoff. No model-outcome selection. Frozen Plain LoRA2500, "
    "32 historical+4 recent slots per normal policy,128 read tokens, one pass per source. Privileged "
    "snapshot8 reference, recent8_now later-frame control. FP32 letter head/bf16 hidden primary; "
    "bf16 secondary. No training/new reserved sources/audio/subtitles/query-time vision/LM KV memory. "
    "Single deterministic memory seed per source, not multiple training seeds. Development diagnostic, "
    "not official benchmark or proof of natural past-event dependence from timestamps alone."
)


def validate_rows(rows, allowed):
    if (len(rows) != 34 or len({q["id"] for q in rows}) != 34
            or len({q["source_group"] for q in rows}) != 28
            or not {q["source_group"] for q in rows} <= allowed):
        raise ValueError("all34 targets/28 existing sources must survive")
    if any(not math.isfinite(q["query_time"]) or q["query_time"] < 60 or q.get("temporal_clue_type") != "Prior"
           or q.get("frames_required") != "multiple" or q.get("evidence") is not None for q in rows):
        raise ValueError("availability-only candidate metadata changed")
    if sum(q["task"] in TASKS for q in rows) != 26 or sum(q["task"] == "Prospective Reasoning" for q in rows) != 8:
        raise ValueError("prespecified semantic strata changed")


def choose_smoke(sources):
    longest = max(sources, key=lambda s: (s["planned_stream_encoder_calls"], s["source_group"]))
    multi = sorted((s for s in sources if len({q["target_id"] for q in s["queries"]}) > 1
                    and s["source_group"] != longest["source_group"]), key=lambda s: s["source_group"])
    if not multi:
        raise ValueError("smoke must cover distinct longest and multi-target sources")
    return sorted({longest["source_group"], multi[0]["source_group"]})


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.reference / "inputs.json").read_text())
    if base["id"] != core.REFERENCE_ID or digest({k: v for k, v in base.items() if k != "id"}) != core.REFERENCE_ID:
        raise ValueError("fixed natural model reference changed")
    ready = json.loads((args.data / "ready.json").read_text())
    if (not ready["completed"] or ready["id"] != DATA_ID or sha256(args.data / "ready.json") != READY_SHA256
            or digest({k: v for k, v in ready.items() if k != "id"}) != DATA_ID
            or sha256(args.data / "questions.jsonl") != ready["questions_sha256"]):
        raise ValueError("fully verified candidate data changed")
    spec = json.loads((args.data / "candidate-spec.json").read_text())
    if digest({k: v for k, v in spec.items() if k != "id"}) != spec["id"] or spec["id"] != ready["candidate_spec_id"]:
        raise ValueError("candidate selection spec changed")
    old = OLD_DEVELOPMENT_MANIFEST
    if sha256(old) != spec["old_development_manifest_sha256"]:
        raise ValueError("existing development boundary changed")
    allowed = {q["source_group"] for q in map(json.loads, old.read_text().splitlines())}
    rows = [json.loads(line) for line in (args.data / "questions.jsonl").read_text().splitlines()]
    validate_rows(rows, allowed)
    if sorted(q["id"] for q in rows) != spec["question_ids"]:
        raise ValueError("target selection changed")
    if checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]:
        raise ValueError("frozen checkpoint changed")
    if checkpoint_identity(Path(base["model"])) != base["model_identity"]:
        raise ValueError("base weights changed")
    train = json.loads((Path(base["plain"]).parent / "run.json").read_text())
    training_hashes = set(train["videos"].values())
    sources = []
    for s in sorted(ready["sources"], key=lambda s: s["source_group"]):
        if (sha256(s["video"]) != s["video_sha256"] or sha256(s["metadata"]) != s["metadata_sha256"]
                or s["video_sha256"] in training_hashes or not s["video"].startswith("/data/")):
            raise ValueError("source bytes/PTS/location or training disjointness changed")
        originals = sorted([q for q in rows if q["source_group"] == s["source_group"]], key=lambda q: (q["query_time"], q["id"]))
        if not 1 <= len(originals) <= 2:
            raise ValueError("protected reference target budget exceeded")
        queries = core.delayed_queries(originals, s["last_valid_source_time"])
        if len(queries) != len(originals) * 5:
            raise ValueError("all5 delays required for every target; do not silently drop")
        metadata = json.loads(Path(s["metadata"]).read_text())
        indices, times, dropped = longstream.schedule(metadata, max(q["query_time"] for q in queries))
        for q in queries:
            q["planned_encoder_calls"] = int(sum(pair[-1] <= q["query_time"] for pair in times))
            q["target_encoder_calls"] = int(sum(pair[-1] <= q["target_time"] for pair in times))
        sources.append({"source_group": s["source_group"], "video": s["video"], "video_sha256": s["video_sha256"],
                        "metadata": s["metadata"], "metadata_sha256": s["metadata_sha256"],
                        "queries": queries, "last_valid_source_time": s["last_valid_source_time"],
                        "sampled_frame_indices": indices.tolist(), "sampled_frame_times": times.tolist(),
                        "planned_stream_encoder_calls": len(times), "dropped_bad_pts_frames": dropped,
                        "memory_seed": int(digest(["natural-longprefix-writer-v1", s["source_group"]])[:8], 16),
                        "events": [list(event) for event in core.events_for(queries)]})
    inputs = {**{k: copy.deepcopy(base[k]) for k in ("model", "model_identity", "plain", "checkpoint", "pixels",
                                                   "capacity", "seed", "primary_score", "secondary_score")},
              "protocol": "natural-delayed-candidates-v1", "scope": SCOPE,
              "sources": sources, "questions": 170, "original_targets": 34,
              "candidate_data": str(args.data), "candidate_data_id": DATA_ID, "candidate_ready_sha256": READY_SHA256,
              "candidate_source_code_sha256": sha256(__file__), "delayed_source_code_sha256": sha256(core.__file__),
              "source_code_sha256": sha256(longstream.__file__), "cohorts": {str(d): {"questions": 34, "sources": 28} for d in core.DELAYS},
              "smoke_source_groups": choose_smoke(sources), "primary_tasks": sorted(TASKS),
              "primary_targets": 26, "secondary_prediction_targets": 8,
              "normal_history_slots": 32, "normal_recent_slots": 4,
              "protected_target_limit": 2, "protected_total_step_limit": 24,
              "natural_reference": str(args.reference), "natural_reference_input_id": core.REFERENCE_ID,
              "overlap_reference": str(args.overlap), "overlap_reference_input_id": "2b4ca77baee10a9d84f293f9bb0d6502e9577da00c573053afb1c31239c594c5"}
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)
    write_json(args.out / "data-audit.json", {"completed": True, "input_id": inputs["id"],
               "questions": 170, "original_targets": 34, "source_videos": 28,
               "primary_targets": 26, "secondary_prediction_targets": 8,
               "all_targets_have_all5_delays": True, "model_outcomes_used_for_eligibility": False,
               "new_reserved_source_groups_consumed": 0, "scope": SCOPE})


def worker(args):
    inputs = json.loads((args.out / "inputs.json").read_text())
    if inputs["candidate_source_code_sha256"] != sha256(__file__):
        raise ValueError("candidate protocol code changed")
    core.worker(args)


def report(args):
    inputs = json.loads((args.out / "inputs.json").read_text())
    if inputs["candidate_source_code_sha256"] != sha256(__file__):
        raise ValueError("candidate report source changed")
    core.report(args)
    path = args.out / ("smoke-ok.json" if args.smoke else "report.json")
    result = json.loads(path.read_text())
    selected = [s for s in inputs["sources"] if not args.smoke or s["source_group"] in inputs["smoke_source_groups"]]
    population = []
    old = json.loads((args.overlap / "inputs.json").read_text())
    if old["id"] != inputs["overlap_reference_input_id"]:
        raise ValueError("overlap reference changed")
    old_ids = {q["id"] for s in old["sources"] for q in s["queries"]}
    overlap_count = 0
    for s in selected:
        for q in s["queries"]:
            r = json.loads((args.out / "questions" / f"{digest(q['id'])}.json").read_text())
            population.append({**r, "task": q["task"]})
            if q["id"] in old_ids:
                prior = json.loads((args.overlap / "questions" / f"{digest(q['id'])}.json").read_text())
                if r["predictions"] != prior["predictions"]:
                    raise ValueError("shared original target/input predictions did not exactly replay")
                overlap_count += 1
    strata = {}
    for name, tasks in (("primary_past", TASKS), ("secondary_prospective", {"Prospective Reasoning"})):
        selected_rows = [r for r in population if r["task"] in tasks]
        strata[name] = {"targets": len({r["target_id"] for r in selected_rows}),
                        "sources": len({r["source_group"] for r in selected_rows}),
                        "by_delay": {str(d): {"questions": len(rows), "scores": core.scores(rows)}
                                     for d in core.DELAYS if (rows := [r for r in selected_rows if r["delay_seconds"] == d])}}
    if not args.smoke and (strata["primary_past"]["targets"] != 26 or strata["secondary_prospective"]["targets"] != 8
                           or overlap_count != 15):
        raise ValueError("fixed strata/shared15-question replay incomplete")
    result.update(scope=SCOPE, task_strata=strata, exact_overlap_query_replays=overlap_count,
                  accuracy_used_as_gate=False, candidate_data_id=DATA_ID)
    write_json(path, result)


def run(args):
    prepare(args)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TRACE_WANDB": "0",
           "TRACE_SWANLAB": "0", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    run_jobs([("gpu-check", [sys.executable, "-c", "import torch; assert torch.cuda.device_count()==4; "
                            "assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24"], list(range(4)))],
             env, args.out / "logs")
    command = [sys.executable, "-m", "trace_vl.core_vlm.longstream_delayed_candidates", "worker", "--out", str(args.out)]
    run_jobs([(f"smoke-{i}", [*command, "--shard", str(i), "--smoke"], [i]) for i in range(4)], env, args.out / "logs")
    args.smoke = True
    report(args)
    inputs = json.loads((args.out / "inputs.json").read_text())
    expected = sum(len(s["queries"]) for s in inputs["sources"] if s["source_group"] in inputs["smoke_source_groups"])
    smoke = json.loads((args.out / "smoke-ok.json").read_text())
    if smoke["questions"] != expected or smoke["source_videos"] != 2:
        raise ValueError("smoke must cover longest and distinct multi-target sources")
    run_jobs([(f"full-{i}", [*command, "--shard", str(i)], [i]) for i in range(4)], env, args.out / "logs")
    args.smoke = False
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "worker", "run", "report"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("/data/yyy/tracevl-data/streaming-delayed-candidates-20261125-001"))
    p.add_argument("--reference", type=Path, default=Path("/data/yyy/tracevl-runs/core-longstream-natural-c2f39df-001"))
    p.add_argument("--overlap", type=Path, default=Path("/data/yyy/tracevl-runs/core-longstream-delayed-92550e9-001"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    {"prepare": prepare, "worker": worker, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
