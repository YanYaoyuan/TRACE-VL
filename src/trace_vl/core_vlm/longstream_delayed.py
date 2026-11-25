"""Snapshot-anchored delayed questions over unchanged natural development videos."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream, longstream_natural
from .control import checkpoint_identity
from .data import presentation_frames, sha256
from .experiment import run_jobs
from .native import NativeVideoQALoss, prepare_native
from .prepare import digest, write_json
from .qa import VisualStep, grid_pool
from .train import checked_checkpoint

DELAYS = (0, 15, 60, 180, 300)
ARMS = ("text_only", "recent8_now", "fifo_time8", "reservoir_time8", "prefix_time8", "protected_snapshot8")
REFERENCE_ID = "7b82c34012d3e0fd8c17b115a97ce4c1abc67229a755358df88dccc60cec4412"
REFERENCE_REPORT_SHA256 = "2aa21fb3b1f6cd927fcaafd0e545234603082f7b1f32044c81f5cf2b18e493bf"
SCOPE = (
    "Controlled development diagnostic, original9 targets/8 already-used sources, unchanged natural video. "
    "Questions explicitly anchored to their original cutoff; repeated at0/15/60/180/300s if media exist. "
    "Eligibility never uses model outcomes. Target-relative causal time reads,32 historical+4 recent "
    "slots per normal bank, at most8 steps/128 tokens per read. recent8_now is a later-frame control. "
    "protected_snapshot8 has privileged target-time writes, not a deployable memory or gold oracle. "
    "Frozen Plain LoRA2500, fp32 final letter head over bf16 hidden primary, bf16 secondary. "
    "No training, audio, subtitles, pixel replay at query or LM KV memory. Not an official benchmark score."
)


def anchored_question(question, target):
    if not math.isfinite(target) or target < 0:
        raise ValueError("invalid target cutoff")
    return (
        f"The following question refers to the video up to {target:.3f} seconds from its start. "
        "Answer as of that cutoff, ignoring any later events.\n" + question
    )


def delayed_queries(originals, last_time):
    """All targets survive; only unavailable continuation excludes a delay condition."""
    if not math.isfinite(last_time):
        raise ValueError("invalid media end")
    queries = []
    for q in originals:
        target = q["query_time"]
        if target > last_time + 1e-5:
            raise ValueError("original target outside media")
        for delay in DELAYS:
            if target + delay > last_time + 1e-5:
                continue
            queries.append({
                **copy.deepcopy(q), "id": q["id"] + f"/delay{delay}", "target_id": q["id"],
                "original_question": q["question"], "question": anchored_question(q["question"], target),
                "target_time": target, "query_time": target + delay, "delay_seconds": delay,
                "additional_wait_seconds": delay, "cohort": "natural_snapshot_delayed",
                "observation": {"start": 0.0, "end": target + delay},
                "role": "controlled-existing-source-development-diagnostic",
            })
    return sorted(queries, key=lambda q: (q["query_time"], q["id"]))


def events_for(queries):
    targets = {}
    for q in queries:
        target = (q["target_time"], q["question"], q["options"], q["answer"])
        if q["target_id"] in targets and targets[q["target_id"]] != target:
            raise ValueError("target changed across delays")
        targets[q["target_id"]] = target
    captures = [(v[0], 0, k) for k, v in targets.items()]
    asks = [(q["query_time"], 1, q["id"]) for q in queries]
    return sorted(captures + asks)


class CausalEvents:
    """Events on a frame-pair end run after its encoding; earlier events run before."""

    def __init__(self, events, dispatch):
        self.events, self.dispatch, self.cursor = events, dispatch, 0

    def due(self, boundary, *, inclusive):
        while self.cursor < len(self.events):
            event = self.events[self.cursor]
            if event[0] > boundary or (not inclusive and event[0] == boundary):
                break
            self.dispatch(event)
            self.cursor += 1


def read_steps(bank, arm, target):
    if arm == "recent8_now":
        return list(bank.recent)
    history = {"fifo_time8": bank.fifo, "reservoir_time8": bank.reservoir,
               "prefix_time8": bank.prefix}.get(arm)
    if history is None:
        raise ValueError("unknown normal read policy")
    # Only the common FOUR recent slots belong to a historical policy. The
    # extra four cached by the recent8 baseline cannot augment its budget.
    available = {s.indices: s for s in (*history, *list(bank.recent)[-4:]) if s.end <= target}
    return sorted(available.values(), key=lambda s: s.end)[-8:]


def step_identity(step):
    return {"indices": list(step.indices), "times": list(step.times), "rgb_sha256": step.rgb_sha256}


def memory_audit(bank, snapshots):
    views = {"recent8_control": list(bank.recent), "fifo": list(bank.fifo),
             "reservoir": bank.reservoir, "prefix": bank.prefix}
    views.update({"snapshot:" + k: v for k, v in snapshots.items()})
    units = {s.indices: s for values in views.values() for s in values}
    bytes_per_step = (bank.recent[-1].tokens.numel() * bank.recent[-1].tokens.element_size() + 96
                      if bank.recent else 0)
    per_policy = {
        key: len({s.indices for s in (*views[key], *list(bank.recent)[-4:])})
        for key in ("fifo", "reservoir", "prefix")
    }
    protected = {s.indices for values in snapshots.values() for s in values}
    protected.update(s.indices for s in bank.recent)
    per_policy.update(recent8_now=len(bank.recent), protected_snapshot8=len(protected))
    if (len(bank.fifo) > 32 or len(bank.reservoir) > 32 or len(bank.recent) > 8
            or len(bank.prefix) > 4 or len(snapshots) > 2
            or any(len(v) > 8 for v in snapshots.values())
            or per_policy["fifo"] > 36 or per_policy["reservoir"] > 36
            or per_policy["prefix"] > 8 or per_policy["protected_snapshot8"] > 24 or len(units) > 92):
        raise ValueError("bounded storage capacity exceeded")
    return {
        "capacity": bank.capacity, "seen": bank.seen, "old_seen": bank.old_seen,
        "lineage": bank.lineage.hexdigest(), "per_policy_unique_retained_steps": per_policy,
        "per_policy_feature_and_fixed_metadata_bytes": {k: n * bytes_per_step for k, n in per_policy.items()},
        "joint_experiment_unique_steps": len(units),
        "joint_experiment_feature_and_fixed_metadata_bytes": len(units) * bytes_per_step,
        "retained": {key: [step_identity(s) for s in values] for key, values in views.items()},
        "accounting": "Tensor payload plus96-byte logical metadata per unique slot, not RSS; joint controls share payloads. Frozen PTS indices/FFmpeg buffers are input infrastructure, never readable visual memory.",
    }


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.reference / "inputs.json").read_text())
    done = json.loads((args.reference / "report.json").read_text())
    if (base["id"] != REFERENCE_ID or digest({k: v for k, v in base.items() if k != "id"}) != REFERENCE_ID
            or not done["completed"] or done["input_id"] != REFERENCE_ID or done["questions"] != 9
            or sha256(args.reference / "report.json") != REFERENCE_REPORT_SHA256):
        raise ValueError("completed original natural reference changed")
    if checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]:
        raise ValueError("frozen checkpoint changed")
    if checkpoint_identity(Path(base["model"])) != base["model_identity"]:
        raise ValueError("base model changed")
    sources = []
    for s in base["sources"]:
        if sha256(s["video"]) != s["video_sha256"] or sha256(s["metadata"]) != s["metadata_sha256"]:
            raise ValueError("natural source or full PTS metadata changed")
        metadata = json.loads(Path(s["metadata"]).read_text())
        pts, keep = presentation_frames(metadata["frames"], drop_bad_pts=True)
        last_time = float((pts - pts[0])[np.flatnonzero(keep)[-1]])
        queries = delayed_queries(s["queries"], last_time)
        indices, times, dropped = longstream.schedule(metadata, max(q["query_time"] for q in queries))
        if indices[:len(s["sampled_frame_indices"])].tolist() != s["sampled_frame_indices"]:
            raise ValueError("continuation altered original prefix sampling")
        for q in queries:
            q["planned_encoder_calls"] = longstream_natural.prefix_calls(times, q["query_time"])
            q["target_encoder_calls"] = longstream_natural.prefix_calls(times, q["target_time"])
        sources.append({**copy.deepcopy(s), "queries": queries, "last_valid_source_time": last_time,
                        "sampled_frame_indices": indices.tolist(), "sampled_frame_times": times.tolist(),
                        "planned_stream_encoder_calls": len(times), "dropped_bad_pts_frames": dropped,
                        "events": events_for(queries)})
    cohorts = {str(d): {"questions": sum(q["delay_seconds"] == d for s in sources for q in s["queries"]),
                       "sources": sum(any(q["delay_seconds"] == d for q in s["queries"]) for s in sources)}
               for d in DELAYS}
    if cohorts != {"0": {"questions": 9, "sources": 8}, "15": {"questions": 9, "sources": 8},
                   "60": {"questions": 7, "sources": 6}, "180": {"questions": 7, "sources": 6},
                   "300": {"questions": 3, "sources": 2}}:
        raise ValueError("availability-only frozen cohorts changed")
    inputs = {**copy.deepcopy(base), "protocol": "natural-snapshot-delayed-v1", "scope": SCOPE,
              "delayed_source_code_sha256": sha256(__file__), "source_code_sha256": sha256(longstream.__file__),
              "sources": sources, "questions": sum(v["questions"] for v in cohorts.values()),
              "reference": str(args.reference), "reference_input_id": REFERENCE_ID,
              "reference_report_sha256": sha256(args.reference / "report.json"), "cohorts": cohorts,
              "delays_seconds": list(DELAYS), "normal_history_slots": 32, "normal_recent_slots": 4,
              "protected_target_limit": 2, "protected_total_step_limit": 24,
              "smoke_source_groups": ["streamingbench-real:0137", "streamingbench-real:0199"]}
    inputs.pop("id")
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)
    write_json(args.out / "data-audit.json", {"completed": True, "input_id": inputs["id"],
               "source_videos": 8, "original_targets": 9, "questions": inputs["questions"],
               "cohorts": cohorts, "model_outcomes_used_for_eligibility": False,
               "original_labels_unchanged": True, "media_unchanged": True,
               "full_metadata_reused_with_verified_sha256": True, "scope": SCOPE})


@torch.no_grad()
def evaluate_source(bb, inputs, source, out, verify, extension=None):
    if sha256(source["video"]) != source["video_sha256"] or sha256(source["metadata"]) != source["metadata_sha256"]:
        raise ValueError("staged source changed")
    metadata = json.loads(Path(source["metadata"]).read_text())
    indices, times, dropped = longstream.schedule(metadata, source["queries"][-1]["query_time"])
    if indices.tolist() != source["sampled_frame_indices"] or times.tolist() != source["sampled_frame_times"]:
        raise ValueError("frozen schedule changed")
    stream = metadata["streams"][0]
    hw = frame_size(stream["height"], stream["width"], inputs["pixels"], 32)
    bank = longstream.BoundedBank(32, source["memory_seed"], keep_prefix=True)
    qa = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    queries = {q["id"]: q for q in source["queries"]}
    target_calls = {q["target_id"]: q["target_encoder_calls"] for q in source["queries"]}
    snapshots, canonical, results = {}, {}, []
    encoded = 0
    started = time.monotonic()

    def audit_memory():
        result = memory_audit(bank, snapshots)
        if extension is not None:
            result["writer_controls"] = extension.audit()
        return result

    def dispatch(event):
        when, kind, identity = event
        if kind == 0:
            if identity in snapshots or encoded != target_calls[identity]:
                raise ValueError("snapshot captured twice or at a noncausal count")
            snapshots[identity] = list(bank.recent)
            if any(s.end > when for s in snapshots[identity]):
                raise ValueError("snapshot saw target future")
            audit_memory()
            return
        q = queries[identity]
        if encoded != q["planned_encoder_calls"]:
            raise ValueError("question saw wrong prefix encode count")
        before = audit_memory()
        predictions = {}
        for arm in (*ARMS, *(extension.arms if extension is not None else ())):
            selected = (snapshots[q["target_id"]] if arm == "protected_snapshot8" else
                        [] if arm == "text_only" else read_steps(bank, arm, q["target_time"]) if arm in ARMS
                        else extension.read(arm, q["target_time"]))
            p = (longstream.answer(bb, qa, selected, q, metadata, hw, source["video_sha256"]) if selected
                 else longstream.text_answer(bb, q, source["video_sha256"]))
            if arm not in ("text_only", "recent8_now") and any(s.end > q["target_time"] for s in selected):
                raise ValueError("target-relative reader used post-target evidence")
            p["empty_visual_memory"] = not selected
            predictions[arm] = p
            key = (q["target_id"], arm)
            # Protected writes and text prompts stay identical as video advances.
            # Gate complete logits, packing/read identities and choices, not accuracy.
            if arm in ("text_only", "protected_snapshot8"):
                if key in canonical and p != canonical[key]:
                    raise ValueError("fixed snapshot/text input changed output across delays")
                canonical[key] = p
        if not results:
            replay = longstream.answer(bb, qa, snapshots[q["target_id"]], q, metadata, hw, source["video_sha256"])
            replay["empty_visual_memory"] = False
            if replay != predictions["protected_snapshot8"]:
                raise ValueError("fresh no-KV question replay changed output")
        if audit_memory() != before:
            raise ValueError("read mutated stored state")
        result = {"id": q["id"], "target_id": q["target_id"], "input_id": inputs["id"],
                  "source_group": source["source_group"], "gold": q["answer"],
                  "target_time": q["target_time"], "query_time": q["query_time"],
                  "delay_seconds": q["delay_seconds"], "target_encoder_calls": q["target_encoder_calls"],
                  "stream_encoder_calls": encoded, "memory": before, "predictions": predictions,
                  "elapsed_seconds": time.monotonic() - started}
        results.append(result)
        write_json(out / "questions" / f"{digest(q['id'])}.json", result)
        print(json.dumps({"question": q["id"], "steps": encoded, "arms": len(predictions)}), flush=True)

    events = CausalEvents(events_for(source["queries"]), dispatch)
    pixels = longstream.stream_pixels(source["video"], indices, *hw)
    try:
        for i, frames in enumerate(pixels):
            pair = tuple(float(t) for t in times[i])
            events.due(pair[-1], inclusive=False)
            if verify and i == 0:
                record = {**source["queries"][0], "answer": 0}
                audit = {"source_fps": float(Fraction(stream["avg_frame_rate"])),
                         "source_frame_count": len(metadata["frames"]), "frame_indices": indices[:2].tolist(),
                         "decoded_rgb_sha256": hashlib.sha256(frames[None].tobytes()).hexdigest()}
                layout = prepare_native(bb.processor, frames[None], np.asarray([pair]), record, audit,
                                        allow_native_timestamp_approximation=True)
            values, grid = bb.preprocess(frames)
            dense = bb.encode(values, grid)
            encoded += 1
            shape = bb.merged_grid(grid)
            if verify and i == 0:
                raw = VisualStep(dense, shape, pair[-1], pair)
                write_json(out / f"native-parity-{digest(source['source_group'])}.json",
                           longstream.verify_native_step(bb, layout, raw, record))
                del raw, layout, record
            pooled = grid_pool(dense, shape, (4, 4))[0].detach().clone()
            observation = longstream.StoredStep(pooled, shape, pair, tuple(int(x) for x in indices[2*i:2*i+2]),
                                                hashlib.sha256(frames.tobytes()).hexdigest())
            bank.append(observation)
            if extension is not None:
                extension.append(observation)
            audit_memory()
            events.due(pair[-1], inclusive=True)
            if i % 30 == 0:
                write_json(out / f"progress-{os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}.json",
                           {"source": source["source_group"], "steps": encoded, "target_steps": len(times),
                            "queries_completed_this_source": len(results), "updated_unix": time.time()})
        events.due(float("inf"), inclusive=True)
    finally:
        pixels.close()
        if extension is not None:
            extension.close()
    if len(results) != len(queries) or events.cursor != len(events.events):
        raise ValueError("source did not finish all fixed queries/events")
    write_json(out / "sources" / f"{digest(source['source_group'])}.json", {
        "completed": True, "input_id": inputs["id"], "source": source["source_group"],
        "queries": [r["id"] for r in results], "stream_encoder_calls": encoded, "sampled_steps": len(times),
        "dropped_bad_pts_frames": dropped, "verification_extra_encoder_calls": int(verify),
        "snapshot_and_text_delay_invariance_passed": True, "final_memory": audit_memory(),
        "peak_two_frame_rgb_buffer_bytes": 2 * hw[0] * hw[1] * 3,
        "shared_input_pts_index_bytes": len(metadata["frames"]) * 8,
        **({"extension_trace": extension.receipt()} if extension is not None else {})})


def worker(args, extension_factory=None):
    if not 0 <= args.shard < 4:
        raise ValueError("invalid shard")
    inputs = json.loads((args.out / "inputs.json").read_text())
    if inputs["delayed_source_code_sha256"] != sha256(__file__) or inputs["source_code_sha256"] != sha256(longstream.__file__):
        raise ValueError("worker code identity changed")
    if checked_checkpoint(Path(inputs["plain"]), inputs["checkpoint"]["run"]) != inputs["checkpoint"]:
        raise ValueError("frozen checkpoint changed")
    selected = inputs["sources"][args.shard::4]
    if args.smoke:
        selected = [s for s in selected if s["source_group"] in inputs["smoke_source_groups"]]
    if not selected:
        return
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    bb = Backbone(BackboneConfig(name=inputs["model"], dtype="bfloat16", local_files_only=True),
                  StreamConfig(pixels_per_frame=inputs["pixels"]), device="cuda:0")
    bb.add_lora(8, 16, dropout=0.0, adapter=str(Path(inputs["plain"]) / "adapter"))
    bb.model.eval().requires_grad_(False)
    for number, source in enumerate(selected):
        done = args.out / "sources" / f"{digest(source['source_group'])}.json"
        if done.exists():
            old = json.loads(done.read_text())
            if old["input_id"] != inputs["id"] or not old["completed"]:
                raise ValueError("completed source belongs to another run")
            continue
        extension = extension_factory(source, args.out) if extension_factory is not None else None
        evaluate_source(bb, inputs, source, args.out, verify=number == 0, extension=extension)


def validate_predictions(result, query, arms=ARMS):
    if set(result["predictions"]) != set(arms):
        raise ValueError("missing or extra prediction arm")
    for arm, p in result["predictions"].items():
        cutoff = query["query_time"] if arm == "recent8_now" else query["target_time"]
        if (p["query_time_encoder_calls"] != 0 or p["visual_tokens"] > 128
                or p["visual_tokens"] != 16 * len(p["read_times"])
                or max((t for pair in p["read_times"] for t in pair), default=0) > cutoff):
            raise ValueError("target/arrival causality, read budget or encoder replay failed")
        for logits, choice in ((p["option_logits"], p["choice"]), (p["option_logits_fp32"], p["choice_fp32"])):
            if (len(logits) != len(query["options"]) or not all(math.isfinite(x) for x in logits)
                    or max(range(len(logits)), key=logits.__getitem__) != choice):
                raise ValueError("candidate logit scoring failed")


def scores(rows, arms=ARMS):
    return {arm: {"correct_fp32": sum(r["predictions"][arm]["choice_fp32"] == r["gold"] for r in rows),
                  "correct_bf16": sum(r["predictions"][arm]["choice"] == r["gold"] for r in rows),
                  "empty_reads": sum(r["predictions"][arm]["empty_visual_memory"] for r in rows)}
            for arm in arms}


def report(args, arms=ARMS, output_path=None):
    inputs = json.loads((args.out / "inputs.json").read_text())
    if inputs["delayed_source_code_sha256"] != sha256(__file__):
        raise ValueError("report source changed")
    selected = [s for s in inputs["sources"] if not args.smoke or s["source_group"] in inputs["smoke_source_groups"]]
    rows, precision, canonical = [], [], {}
    for s in selected:
        done = json.loads((args.out / "sources" / f"{digest(s['source_group'])}.json").read_text())
        if (not done["completed"] or done["input_id"] != inputs["id"]
                or not done["stream_encoder_calls"] == done["sampled_steps"] == s["planned_stream_encoder_calls"]
                or not done["snapshot_and_text_delay_invariance_passed"]):
            raise ValueError("singlepass count or source completion failed")
        for q in s["queries"]:
            r = json.loads((args.out / "questions" / f"{digest(q['id'])}.json").read_text())
            if (r["id"] != q["id"] or r["input_id"] != inputs["id"] or r["gold"] != q["answer"]
                    or r["target_id"] != q["target_id"] or r["target_time"] != q["target_time"]
                    or r["query_time"] != q["query_time"] or r["delay_seconds"] != q["delay_seconds"]
                    or r["stream_encoder_calls"] != q["planned_encoder_calls"]
                    or r["target_encoder_calls"] != q["target_encoder_calls"]
                    or r["memory"]["seen"] != q["planned_encoder_calls"]):
                raise ValueError("fixed question identity or causal encode count failed")
            validate_predictions(r, q, arms=arms)
            for arm, p in r["predictions"].items():
                key = (q["target_id"], arm)
                if arm in ("text_only", "protected_snapshot8"):
                    if key in canonical and canonical[key] != p:
                        raise ValueError("protected/text delay invariance failed")
                    canonical[key] = p
                if p["choice"] != p["choice_fp32"]:
                    precision.append({"id": r["id"], "arm": arm, "bf16": p["choice"], "fp32": p["choice_fp32"]})
            rows.append(r)
    by_delay = {str(d): {"questions": len(subset), "sources": len({r["source_group"] for r in subset}),
                        "scores": scores(subset, arms=arms)}
                for d in DELAYS if (subset := [r for r in rows if r["delay_seconds"] == d])}
    matched = {}
    # Fixed common populations, rather than comparing9/9 to3/3 as if iid.
    for endpoint in (15, 180, 300):
        targets = {r["target_id"] for r in rows if r["delay_seconds"] == endpoint}
        all_matched = [r for r in rows if r["target_id"] in targets and r["delay_seconds"] <= endpoint]
        matched[str(endpoint)] = {
            "target_ids": sorted(targets), "targets": len(targets),
            "sources": len({r["source_group"] for r in all_matched}),
            "delays": {str(d): {"questions": len(subset), "scores": scores(subset, arms=arms)}
                       for d in DELAYS if d <= endpoint
                       and (subset := [r for r in all_matched if r["delay_seconds"] == d])}}
    if not args.smoke and any(by_delay[k]["questions"] != v["questions"] or by_delay[k]["sources"] != v["sources"]
                              for k, v in inputs["cohorts"].items()):
        raise ValueError("full report population differs from availability-only cohorts")
    write_json(output_path or args.out / ("smoke-ok.json" if args.smoke else "report.json"), {
        "completed": True, "smoke": args.smoke, "input_id": inputs["id"], "scope": SCOPE,
        "questions": len(rows), "source_videos": len(selected), "original_targets": len({r["target_id"] for r in rows}),
        "primary_score": inputs["primary_score"], "by_delay": by_delay, "matched_cohorts": matched,
        "precision_differences": precision, "snapshot_and_text_delay_invariance_passed": True,
        "no_training": True, "architecture_accepted": False})


def run(args):
    prepare(args)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TRACE_WANDB": "0", "TRACE_SWANLAB": "0", "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    run_jobs([("gpu-check", [sys.executable, "-c", "import torch; assert torch.cuda.device_count()==4; "
                            "assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24"], list(range(4)))],
             env, args.out / "logs")
    command = [sys.executable, "-m", "trace_vl.core_vlm.longstream_delayed", "worker", "--out", str(args.out)]
    run_jobs([(f"smoke-{i}", [*command, "--shard", str(i), "--smoke"], [i]) for i in range(4)], env, args.out / "logs")
    args.smoke = True
    report(args)
    smoke = json.loads((args.out / "smoke-ok.json").read_text())
    if smoke["questions"] != 14 or smoke["source_videos"] != 2 or smoke["original_targets"] != 3:
        raise ValueError("smoke must exercise920s target and overlapping targets, including300s retention")
    run_jobs([(f"full-{i}", [*command, "--shard", str(i)], [i]) for i in range(4)], env, args.out / "logs")
    args.smoke = False
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "worker", "run", "report"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path("/data/yyy/tracevl-runs/core-longstream-natural-c2f39df-001"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    {"prepare": prepare, "worker": worker, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
