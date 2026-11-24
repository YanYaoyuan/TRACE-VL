"""Frozen six-arm one-pass diagnostic on existing natural long-prefix development media."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

from trace_vl.streaming_vision.media import VideoProbeCache
from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream
from .control import checkpoint_identity
from .data import sha256
from .experiment import run_jobs
from .prepare import digest, write_json
from .train import checked_checkpoint

DATA_ID = "071825c289d9be09a775ff0607ce18b010fe44dd1971ae892d63405e4b267851"
ARMS = longstream.BASELINES + ("prefix4",)
SCOPE = (
    "Natural full-prefix development diagnostic, all9 Prior/multiple queries>=600s inside8 "
    "already-used StreamingBench development source groups. Frozen Plain LoRA,32 history slots, "
    "128 read tokens, no question-time re-encoding or training. Visual-only, no audio/subtitles. "
    "Primary fp32 final letter head over bf16 hidden states; bf16 scores also reported. "
    "Not official60-second Main, full benchmark, unseen-test or600-second evidence-retention proof."
)


def validate_rows(rows, allowed_groups):
    seen = set()
    for r in rows:
        if r["id"] in seen or r["source_group"] not in allowed_groups:
            raise ValueError("duplicate question or source outside existing development boundary")
        seen.add(r["id"])
        if (
            not math.isfinite(r["query_time"])
            or r["query_time"] < 600
            or r.get("temporal_clue_type") != "Prior"
            or r.get("frames_required") != "multiple"
            or r.get("evidence") is not None
        ):
            raise ValueError("fixed natural long-prefix metadata selection changed")


def prefix_calls(times, query):
    return int(sum(pair[-1] <= query for pair in times))


def smoke_groups(sources):
    longest = max(sources, key=lambda s: (max(q["query_time"] for q in s["queries"]), s["source_group"]))
    multiple = [s for s in sources if len(s["queries"]) > 1]
    if not multiple:
        raise ValueError("smoke must include a source with multiple arrivals")
    multi = sorted(multiple, key=lambda s: s["source_group"])[0]
    selected = {longest["source_group"], multi["source_group"]}
    if len(selected) != 2:
        raise ValueError("smoke requires distinct longest and multiple-arrival sources")
    return sorted(selected)


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.reference / "inputs.json").read_text())
    ready = json.loads((args.data / "ready.json").read_text())
    check = dict(ready)
    check.pop("id")
    if digest(check) != ready["id"] or ready["id"] != DATA_ID or not ready["completed"]:
        raise ValueError("natural data preparation identity changed")
    if sha256(args.data / "questions.jsonl") != ready["questions_sha256"]:
        raise ValueError("official question manifest changed")
    old = Path("/nas-data/vlm/yyy/tracevl-runs/streaming-vision-a566104-001/streamingbench.jsonl")
    if sha256(old) != ready["old_development_manifest_sha256"]:
        raise ValueError("existing development boundary changed")
    allowed = {json.loads(line)["source_group"] for line in old.read_text().splitlines()}
    rows = [json.loads(line) for line in (args.data / "questions.jsonl").read_text().splitlines()]
    validate_rows(rows, allowed)
    groups = {r["source_group"] for r in rows}
    if len(rows) != 9 or len(groups) != 8 or groups != {s["source_group"] for s in ready["sources"]}:
        raise ValueError("preserve all8 fixed sources and9 questions")
    if checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]:
        raise ValueError("Plain checkpoint changed")
    if checkpoint_identity(Path(base["model"])) != base["model_identity"]:
        raise ValueError("base model changed")
    fingerprints = {
        "source_code_sha256": sha256(longstream.__file__),
        "natural_source_code_sha256": sha256(__file__),
        "natural_ready_sha256": sha256(args.data / "ready.json"),
    }
    if (args.out / "inputs.json").exists():
        prior = json.loads((args.out / "inputs.json").read_text())
        if any(prior[k] != v for k, v in fingerprints.items()):
            raise ValueError("existing run identity changed; use a new directory")
        for s in prior["sources"]:
            if sha256(s["video"]) != s["video_sha256"] or sha256(s["metadata"]) != s["metadata_sha256"]:
                raise ValueError("prepared media changed")
        return
    train = json.loads((Path(base["plain"]).parent / "run.json").read_text())
    training_hashes = set(train["videos"].values())
    cache = VideoProbeCache(args.out / "media-index", stage_media=False)
    sources = []
    for s in sorted(ready["sources"], key=lambda s: s["source_group"]):
        if (
            sha256(s["video"]) != s["sha256"]
            or s["sha256"] in training_hashes
            or not s["video"].startswith("/data/")
        ):
            raise ValueError("media changed, left /data or exact-byte training overlap")
        local, meta, checksum = cache.source(s["video"], stage=False)
        metadata = args.out / "metadata" / f"{checksum}.metadata.json"
        write_json(metadata, meta)
        queries = sorted(
            [r for r in rows if r["source_group"] == s["source_group"]],
            key=lambda r: (r["query_time"], r["id"]),
        )
        indices, times, dropped = longstream.schedule(meta, max(q["query_time"] for q in queries))
        queries = [
            {
                **q,
                "video": str(local),
                "additional_wait_seconds": 0,
                "planned_encoder_calls": prefix_calls(times, q["query_time"]),
            }
            for q in queries
        ]
        sources.append(
            {
                "source_group": s["source_group"],
                "video": str(local),
                "video_sha256": checksum,
                "metadata": str(metadata),
                "metadata_sha256": sha256(metadata),
                "queries": queries,
                "sampled_frame_indices": indices.tolist(),
                "sampled_frame_times": times.tolist(),
                "planned_stream_encoder_calls": len(times),
                "dropped_bad_pts_frames": dropped,
                "memory_seed": int(digest(["natural-longprefix-writer-v1", s["source_group"]])[:8], 16),
            }
        )
    inputs = {
        **copy.deepcopy(base),
        **fingerprints,
        "protocol": "natural-longprefix-existing-dev-v1",
        "scope": SCOPE,
        "natural_data_id": DATA_ID,
        "natural_data": str(args.data),
        "sources": sources,
        "questions": 9,
        "position_controls": True,
        "smoke_source_groups": smoke_groups(sources),
        "primary_score": "fp32_letter_head_over_bf16_hidden",
        "secondary_score": "bf16_letter_head",
        "original_reference_input_id": base["id"],
    }
    inputs.pop("id")
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)
    write_json(
        args.out / "data-audit.json",
        {
            "completed": True,
            "source_videos": 8,
            "questions": 9,
            "existing_development_sources_only": True,
            "exact_byte_training_overlap": False,
            "all_media_on_data": True,
            "query_time_seconds": [r["query_time"] for r in rows],
            "full_sequential_decode_required_in_worker": True,
            "scope": SCOPE,
        },
    )


def report(args):
    inputs = json.loads((args.out / "inputs.json").read_text())
    if inputs["natural_source_code_sha256"] != sha256(__file__):
        raise ValueError("natural protocol source changed")
    sources = [
        s for s in inputs["sources"] if not args.smoke or s["source_group"] in inputs["smoke_source_groups"]
    ]
    population = []
    precision = []
    for source in sources:
        done = json.loads((args.out / "sources" / f"{digest(source['source_group'])}.json").read_text())
        if (
            not done["completed"]
            or done["input_id"] != inputs["id"]
            or not done["stream_encoder_calls"]
            == done["sampled_steps"]
            == source["planned_stream_encoder_calls"]
        ):
            raise ValueError("source encoder count differs from prepared schedule")
        for q in source["queries"]:
            r = json.loads((args.out / "questions" / f"{digest(q['id'])}.json").read_text())
            if (
                r["id"] != q["id"]
                or r["gold"] != q["answer"]
                or r["input_id"] != inputs["id"]
                or r["stream_encoder_calls"] != q["planned_encoder_calls"]
                or r["query_time"] != q["query_time"]
                or r["evidence_gap_seconds"] is not None
                or set(r["predictions"]) != set(ARMS)
            ):
                raise ValueError("question identity/count or evidence-age inference failed")
            m = r["memory"]
            if (
                m["capacity"] != 32
                or m["fifo_steps"] > 32
                or m["reservoir_steps"] > 32
                or m["recent_steps"] > 8
                or m["prefix_control_steps"] != 4
                or m["seen"] != q["planned_encoder_calls"]
                or m["joint_experiment_unique_retained_steps"] > 76
            ):
                raise ValueError("retained-state budget failed")
            for arm, p in r["predictions"].items():
                if (
                    p["visual_tokens"] > 128
                    or p["query_time_encoder_calls"] != 0
                    or max((t for pair in p["read_times"] for t in pair), default=0) > q["query_time"]
                ):
                    raise ValueError("read budget, query replay or causality failed")
                for logits, choice in (
                    (p["option_logits"], p["choice"]),
                    (p["option_logits_fp32"], p["choice_fp32"]),
                ):
                    if (
                        len(logits) != len(q["options"])
                        or not all(math.isfinite(v) for v in logits)
                        or max(range(len(logits)), key=logits.__getitem__) != choice
                    ):
                        raise ValueError("nonfinite or mis-scored option logits")
                if p["choice"] != p["choice_fp32"]:
                    precision.append(
                        {"id": r["id"], "arm": arm, "bf16": p["choice"], "fp32": p["choice_fp32"]}
                    )
            population.append({**r, "task": q["task"]})
    results = {}
    for task in ["overall"] + sorted({r["task"] for r in population}):
        subset = population if task == "overall" else [r for r in population if r["task"] == task]
        results[task] = {
            "questions": len(subset),
            "source_videos": len({r["source_group"] for r in subset}),
            "scores": {
                arm: {
                    "correct_fp32": sum(r["predictions"][arm]["choice_fp32"] == r["gold"] for r in subset),
                    "correct_bf16": sum(r["predictions"][arm]["choice"] == r["gold"] for r in subset),
                }
                for arm in ARMS
            },
        }
    write_json(
        args.out / ("smoke-ok.json" if args.smoke else "report.json"),
        {
            "completed": True,
            "smoke": args.smoke,
            "input_id": inputs["id"],
            "scope": SCOPE,
            "questions": len(population),
            "source_videos": len(sources),
            "primary_score": inputs["primary_score"],
            "results": results,
            "precision_differences": precision,
            "architecture_accepted": False,
            "no_training": True,
        },
    )


def run(args):
    import sys

    prepare(args)
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TRACE_WANDB": "0",
        "TRACE_SWANLAB": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    run_jobs(
        [
            (
                "gpu-check",
                [
                    sys.executable,
                    "-c",
                    "import torch; assert torch.cuda.device_count()==4; "
                    "assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24",
                ],
                list(range(4)),
            )
        ],
        env,
        args.out / "logs",
    )
    command = [sys.executable, "-m", "trace_vl.core_vlm.longstream", "worker", "--out", str(args.out)]
    run_jobs(
        [(f"smoke-{i}", [*command, "--shard", str(i), "--smoke"], [i]) for i in range(4)],
        env,
        args.out / "logs",
    )
    args.smoke = True
    report(args)
    smoke = json.loads((args.out / "smoke-ok.json").read_text())
    if smoke["questions"] != 3 or smoke["source_videos"] != 2:
        raise ValueError("smoke must cover longest prefix and multiple arrivals")
    run_jobs([(f"full-{i}", [*command, "--shard", str(i)], [i]) for i in range(4)], env, args.out / "logs")
    args.smoke = False
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run", "report"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--data", type=Path, default=Path("/data/yyy/tracevl-data/streaming-longprefix-existing-dev-eec763c")
    )
    p.add_argument(
        "--reference", type=Path, default=Path("/data/yyy/tracevl-runs/core-longstream-29e5163-001")
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
