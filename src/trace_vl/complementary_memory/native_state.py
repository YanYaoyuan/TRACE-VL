"""Test A: carried native state x historical KV on the frozen 77-source development cohort."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.core_vlm import longstream
from trace_vl.core_vlm import longstream_delayed as delayed
from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.experiment import run_jobs
from trace_vl.core_vlm.native import NativeVideoQALoss, prepare_native
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VisualStep, grid_pool
from trace_vl.core_vlm.train import checked_checkpoint
from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from . import analysis, state_stream
from .state_stream import Block, NativeStream, block_inputs, replay, score, segment_layout

REFERENCE_ID = "090db3b555dd368f8d7478a2de7195408a4aa6788e451e02636fc4d91cc59e4b"
RECENT, CAPACITY, MEMORY_SEEDS = 8, 32, (17, 29, 43)
PARITY_STEPS, PARITY_TOLERANCE = 8, .5
NATURAL = ("text_only", "recent_replay", "window_stream", "full_stream", "snapshot_replay")
ARMS = NATURAL + tuple(f"reservoir{s}:{f}" for s in MEMORY_SEEDS for f in analysis.FACTORS)
SCOPE = ("Reused development diagnostic, 77 sources/385 anchored questions; not an official or unseen score. "
         "Frozen reader, no training. Question text states its past cutoff; no arm filters reads by it except the "
         "privileged snapshot_replay. S/K fork arms are mechanism interventions, not deployable baselines. "
         "Live KV was computed with the carried state, so K0 recent entries can still contain history. "
         "full_stream is unbounded. Intervals are pointwise and exploratory.")
CODE = {"native_state": __file__, "state_stream": state_stream.__file__, "analysis": analysis.__file__,
        "longstream": longstream.__file__, "longstream_delayed": delayed.__file__}


def code_identity():
    from trace_vl.core_vlm import native, qa
    from trace_vl.v3 import backbone, positions
    files = {**CODE, "native": native.__file__, "qa": qa.__file__, "backbone": backbone.__file__,
             "positions": positions.__file__}
    return {k: sha256(v) for k, v in files.items()}


def versions():
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__}


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.reference / "inputs.json").read_text())
    if base["id"] != REFERENCE_ID or digest({k: v for k, v in base.items() if k != "id"}) != REFERENCE_ID:
        raise ValueError("frozen 77-source reference changed")
    if len(base["sources"]) != 77 or any(len(s["queries"]) != 5 for s in base["sources"]):
        raise ValueError("all 77 sources and five delays required")
    if checkpoint_identity(Path(base["model"])) != base["model_identity"]:
        raise ValueError("base model changed")
    if checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]:
        raise ValueError("frozen checkpoint changed")
    inputs = {k: copy.deepcopy(base[k]) for k in ("model", "model_identity", "plain", "checkpoint", "pixels",
                                                  "sources", "smoke_source_groups")}
    inputs.update(protocol="complementary-memory-test-a-v1", scope=SCOPE, reference=str(args.reference),
                  reference_input_id=REFERENCE_ID, weights=args.weights, recent=RECENT, capacity=CAPACITY,
                  memory_seeds=list(MEMORY_SEEDS), arms=list(ARMS), code=code_identity(), versions=versions(),
                  parity={"steps": PARITY_STEPS, "logit_tolerance": PARITY_TOLERANCE},
                  contrasts=analysis.CONTRASTS, bootstrap={"draws": analysis.DRAWS, "seed": analysis.SEED})
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)


def load_inputs(args):
    inputs = json.loads((args.out / "inputs.json").read_text())
    if inputs["code"] != code_identity() or inputs["versions"] != versions():
        raise ValueError("code or library versions differ from the prepared run")
    return inputs


def memory_seed(group, seed):
    return int(digest(["complementary-test-a-reservoir-v1", group, seed])[:8], 16)


def native_prompt(bb, query, indices, times, metadata, hw):
    """Head/tail tokens from the real processor. Zero pixels only create placeholders."""
    images = np.zeros((1, 2, *hw, 3), np.uint8)
    audit = {"source_fps": float(Fraction(metadata["streams"][0]["avg_frame_rate"])),
             "source_frame_count": len(metadata["frames"]), "frame_indices": [int(i) for i in indices[:2]],
             "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    layout = prepare_native(bb.processor, images, np.asarray(times[:1]), {**query, "answer": 0}, audit,
                            allow_native_timestamp_approximation=True)
    return segment_layout(layout.inputs["input_ids"][0].tolist(), bb.video_token, bb.vision_start, bb.vision_end,
                          stamp_ids(bb, indices[:2], audit["source_fps"]))


def stamp_ids(bb, indices, fps):
    return bb.ids(f"<{float(np.mean(np.asarray(indices) / fps)):.1f} seconds>")


@torch.no_grad()
def check_parity(bb, head, tail, blocks, raw, frames, query, indices, times, metadata):
    """Exact native packing, then cached-chunk versus uncached answer logits."""
    n = len(blocks)
    images = np.stack(frames)
    audit = {"source_fps": float(Fraction(metadata["streams"][0]["avg_frame_rate"])),
             "source_frame_count": len(metadata["frames"]), "frame_indices": [int(i) for i in indices[:2 * n]],
             "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    safe = {**query, "answer": 0}
    layout = prepare_native(bb.processor, images, np.asarray(times[:n]), safe, audit,
                            allow_native_timestamp_approximation=True)
    qa = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    embeds, positions, *_ = qa.prefix(raw, {**safe, "native_layout": layout})
    counter = state_stream.PositionCounter()
    parts = [(bb.embed_ids(head), counter.text(len(head), bb.device))]
    parts += [block_inputs(bb, b, counter) for b in blocks]
    parts.append((bb.embed_ids(tail), counter.text(len(tail), bb.device)))
    mine, coords = torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
    if not torch.equal(mine, embeds) or not torch.equal(coords, positions):
        raise RuntimeError("stream packing differs from native packing")
    letters = [bb.ids(chr(65 + i))[0] for i in range(len(query["options"]))]
    hidden = bb.lm(embeds, positions, cache=None, use_cache=False)[0, -1]
    direct = bb.lm_head.weight[letters].float() @ hidden.float()
    stream = replay(bb, head, blocks)
    cached = torch.tensor(score(bb, stream.fork(), stream.counter.pos, tail, len(letters))["option_logits_fp32"],
                          device=direct.device)
    gap = float((direct - cached).abs().max())
    top2 = direct.topk(2).values
    same = int(direct.argmax()) == int(cached.argmax())
    if not gap <= PARITY_TOLERANCE or (not same and float(top2[0] - top2[1]) > 2 * gap):
        raise RuntimeError(f"cached/uncached answer mismatch: gap={gap}, same_choice={same}")
    return {"steps": n, "exact_packing": True, "max_letter_logit_gap": gap, "same_choice": same,
            "tolerance": PARITY_TOLERANCE}


@torch.no_grad()
def evaluate_source(bb, inputs, source, out):
    if sha256(source["video"]) != source["video_sha256"] or sha256(source["metadata"]) != source["metadata_sha256"]:
        raise ValueError("source media changed")
    metadata = json.loads(Path(source["metadata"]).read_text())
    queries = {q["id"]: q for q in source["queries"]}
    indices, times, _ = longstream.schedule(metadata, max(q["query_time"] for q in queries.values()))
    if indices.tolist() != source["sampled_frame_indices"] or times.tolist() != source["sampled_frame_times"]:
        raise ValueError("frozen frame schedule changed")
    stream_meta = metadata["streams"][0]
    fps = float(Fraction(stream_meta["avg_frame_rate"]))
    hw = frame_size(stream_meta["height"], stream_meta["width"], inputs["pixels"], 32)
    prompts = {k: native_prompt(bb, q, indices, times, metadata, hw) for k, q in queries.items()}
    head = next(iter(prompts.values()))[0]
    if any(h != head for h, _ in prompts.values()):
        raise ValueError("question changed the question-blind prompt head")
    live = {"window_stream": NativeStream(bb, head, "window", recent=RECENT),
            "full_stream": NativeStream(bb, head, "full", recent=RECENT)}
    for s in MEMORY_SEEDS:
        live[f"reservoir{s}"] = NativeStream(bb, head, "reservoir", recent=RECENT, capacity=CAPACITY,
                                             seed=memory_seed(source["source_group"], s))
    recent = deque(maxlen=RECENT)  # (block, position at block start)
    snapshots, rows = {}, []
    parity_parts = ([], [], [])
    parity, encoded = None, 0
    seconds = {"vision": 0., "lm_step_all_streams": 0., "questions": 0.}
    torch.cuda.reset_peak_memory_stats()

    def ask(q):
        if encoded != q["planned_encoder_calls"]:
            raise ValueError("question saw the wrong causal prefix")
        started = time.monotonic()
        tail, options = prompts[q["id"]][1], len(q["options"])
        before = {name: s.snapshot() for name, s in live.items()}
        blocks = [b for b, _ in recent]
        donor = replay(bb, head, blocks, positions_start=recent[0][1])
        fresh = replay(bb, head, blocks)
        privileged = replay(bb, head, snapshots[q["target_id"]])

        def run(stream, position=None, **kw):
            return score(bb, stream.fork(**kw), stream.counter.pos if position is None else position, tail, options)

        text = longstream.text_answer(bb, q, source["video_sha256"])
        predictions = {"text_only": {"choice": text["choice_fp32"], "option_logits_fp32": text["option_logits_fp32"]},
                       "recent_replay": run(fresh), "window_stream": run(live["window_stream"]),
                       "full_stream": run(live["full_stream"]), "snapshot_replay": run(privileged)}
        for s in MEMORY_SEEDS:
            stream = live[f"reservoir{s}"]
            if stream.counter.pos != donor.counter.pos:
                raise RuntimeError("donor and live coordinates differ")
            for factor in analysis.FACTORS:
                predictions[f"reservoir{s}:{factor}"] = run(
                    stream, history=factor.endswith("K1"), state_from=None if factor.startswith("S1") else donor)
        if run(live["window_stream"]) != predictions["window_stream"]:
            raise RuntimeError("question replay is not deterministic")
        for name, s in live.items():
            s.verify(before[name])
        torch.cuda.synchronize()
        seconds["questions"] += time.monotonic() - started
        row = {"completed": True, "input_id": inputs["id"], "id": q["id"], "source_group": source["source_group"],
               "target_id": q["target_id"], "task": q.get("task"), "gold": q["answer"],
               "delay_seconds": q["delay_seconds"], "target_time": q["target_time"], "query_time": q["query_time"],
               "steps_seen": encoded, "recent_steps": [b.index for b in blocks],
               "snapshot_steps": [b.index for b in snapshots[q["target_id"]]],
               "streams": {name: s.audit() for name, s in live.items()}, "predictions": predictions}
        write_json(out / "questions" / f"{digest(q['id'])}.json", row)
        rows.append(q["id"])

    def dispatch(event):
        when, kind, identity = event
        if kind == 1:
            return ask(queries[identity])
        blocks = [b for b, _ in recent]
        if not blocks or any(b.end > when for b in blocks):
            raise ValueError("noncausal or empty privileged snapshot")
        snapshots[identity] = blocks

    events = delayed.CausalEvents(delayed.events_for(source["queries"]), dispatch)
    pixels = longstream.stream_pixels(source["video"], indices, *hw)
    try:
        for i, frames in enumerate(pixels):
            end = float(times[i][-1])
            events.due(end, inclusive=False)
            started = time.monotonic()
            values, grid = bb.preprocess(frames)
            dense = bb.encode(values, grid)
            shape = bb.merged_grid(grid)
            tokens, rr, cc = grid_pool(dense, shape, (4, 4))
            torch.cuda.synchronize()
            seconds["vision"] += time.monotonic() - started
            encoded += 1
            block = Block(i, end, tuple(stamp_ids(bb, indices[2 * i:2 * i + 2], fps)), tokens.detach().clone(),
                          rr, cc, shape)
            started = time.monotonic()
            recent.append((block, live["window_stream"].counter.pos))
            for s in live.values():
                s.append(block)
            torch.cuda.synchronize()
            seconds["lm_step_all_streams"] += time.monotonic() - started
            if parity is None:
                pair = tuple(float(t) for t in times[i])
                for store, item in zip(parity_parts, (block, VisualStep(dense, shape, end, pair), frames), strict=True):
                    store.append(item)
                if len(parity_parts[0]) == PARITY_STEPS:
                    q = min(queries.values(), key=lambda x: x["query_time"])
                    parity = check_parity(bb, head, prompts[q["id"]][1], *parity_parts, q, indices, times, metadata)
                    parity_parts = ([], [], [])
            events.due(end, inclusive=True)
            if i % 30 == 0:
                print(json.dumps({"source": source["source_group"], "step": encoded, "of": len(times)}), flush=True)
        events.due(float("inf"), inclusive=True)
    finally:
        pixels.close()
    if len(rows) != len(queries) or events.cursor != len(events.events) or parity is None:
        raise ValueError("source did not complete every question and the parity check")
    write_json(out / "sources" / f"{digest(source['source_group'])}.json", {
        "completed": True, "input_id": inputs["id"], "source_group": source["source_group"], "steps": encoded,
        "parity": parity, "live_streams": len(live), "seconds": seconds,
        "lm_seconds_per_stream_step": seconds["lm_step_all_streams"] / (encoded * len(live)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "timing_scope": "includes synchronization; five live diagnostic streams share one vision encode"})


def selected_sources(inputs, smoke):
    return [s for s in inputs["sources"] if not smoke or s["source_group"] in inputs["smoke_source_groups"]]


def worker(args):
    inputs = load_inputs(args)
    if not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    # Shard after filtering so the smoke sources occupy different GPUs.
    sources = selected_sources(inputs, args.smoke)[args.shard::args.shards]
    if not sources:
        return
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    bb = Backbone(BackboneConfig(name=inputs["model"], dtype="bfloat16", local_files_only=True),
                  StreamConfig(pixels_per_frame=inputs["pixels"]), device="cuda:0")
    if inputs["weights"] == "plain":
        bb.add_lora(8, 16, dropout=0., adapter=str(Path(inputs["plain"]) / "adapter"))
    bb.model.eval().requires_grad_(False)
    for source in sources:
        done = args.out / "sources" / f"{digest(source['source_group'])}.json"
        if done.exists():
            previous = json.loads(done.read_text())
            if previous["input_id"] != inputs["id"] or not previous["completed"]:
                raise ValueError("completed source belongs to another run")
            continue
        evaluate_source(bb, inputs, source, args.out)


def report(args):
    inputs = load_inputs(args)
    rows, sources = [], []
    for s in selected_sources(inputs, args.smoke):
        sources.append(json.loads((args.out / "sources" / f"{digest(s['source_group'])}.json").read_text()))
        for q in s["queries"]:
            r = json.loads((args.out / "questions" / f"{digest(q['id'])}.json").read_text())
            if (not r["completed"] or r["input_id"] != inputs["id"] or r["id"] != q["id"] or r["gold"] != q["answer"]
                    or r["steps_seen"] != q["planned_encoder_calls"] or set(r["predictions"]) != set(ARMS)):
                raise ValueError("question identity, prefix or arms changed")
            if any(len(p["option_logits_fp32"]) != len(q["options"]) for p in r["predictions"].values()):
                raise ValueError("answer option count changed")
            for name, a in r["streams"].items():
                limit = {"window": RECENT, "reservoir": RECENT + CAPACITY}.get(a["policy"])
                if limit is not None and a["kv_steps"] > limit:
                    raise ValueError(f"{name} exceeded its KV budget")
            rows.append(r)
    if any(not s["completed"] or s["input_id"] != inputs["id"] for s in sources):
        raise ValueError("incomplete source")
    result = {"completed": True, "smoke": args.smoke, "input_id": inputs["id"], "weights": inputs["weights"],
              "scope": SCOPE, "sources": len(sources), "questions": len(rows),
              "predictions": len(rows) * len(ARMS),
              "max_parity_gap": max(s["parity"]["max_letter_logit_gap"] for s in sources),
              "max_peak_allocated_gib": max(s["peak_allocated_gib"] for s in sources),
              "mean_lm_seconds_per_stream_step": float(np.mean([s["lm_seconds_per_stream_step"] for s in sources])),
              "total_lm_hours": sum(s["seconds"]["lm_step_all_streams"] for s in sources) / 3600}
    if not args.smoke:  # smoke output contains no accuracy
        if len(sources) != 77 or len(rows) != 385:
            raise ValueError("all 77 sources and 385 questions required")
        result["analysis"] = analysis.analyze(rows, MEMORY_SEEDS)
        result["outcome"] = analysis.outcome(result["analysis"])
    write_json(args.out / ("smoke-ok.json" if args.smoke else "report.json"), result)


def run(args):
    gpus = [int(x) for x in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus):
        raise ValueError("distinct GPUs required")
    prepare(args)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TRACE_WANDB": "0",
           "TRACE_SWANLAB": "0", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    check = (f"import torch; assert torch.cuda.device_count()=={len(gpus)}; "
             f"assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range({len(gpus)}))>24")
    run_jobs([("gpu-check", [sys.executable, "-c", check], gpus)], env, args.out / "logs")
    command = [sys.executable, "-m", "trace_vl.complementary_memory.native_state", "worker", "--out", str(args.out),
               "--shards", str(len(gpus))]
    for phase in (("smoke",) if args.smoke_only else ("smoke", "full")):
        extra = ["--smoke"] if phase == "smoke" else []
        run_jobs([(f"{phase}-{i}", [*command, "--shard", str(i), *extra], [gpu]) for i, gpu in enumerate(gpus)],
                 env, args.out / "logs")
        args.smoke = phase == "smoke"
        report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "worker", "run", "report"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path("/data/yyy/tracevl-runs/core-longstream-states-3dc8c83-001"))
    p.add_argument("--weights", choices=("plain", "base"), default="plain")
    p.add_argument("--gpus", default="2,3")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    {"prepare": prepare, "worker": worker, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
