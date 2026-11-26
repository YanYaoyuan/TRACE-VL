"""Bounded real-processor mechanics audit; never a question-quality evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.complementary_memory import native_state as test_a
from trace_vl.complementary_memory.state_stream import block_inputs, segment_layout
from trace_vl.core_vlm import longstream
from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.native import prepare_native
from trace_vl.core_vlm.prepare import digest
from trace_vl.core_vlm.qa import VisualStep, question_and_answer
from trace_vl.core_vlm.train import checked_checkpoint
from trace_vl.reservoir_capacity.run import REFERENCE, REFERENCE_ID, code_identity
from trace_vl.reservoir_capacity.supervisor import atomic_json, budget, run_jobs
from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.positions import PositionCounter

from .mechanics import DelayedReader, DenseArrival, feature_block
from .reference import mixed_native_prefix

STEPS, CEILING_SECONDS, LOGIT_TOLERANCE = 16, 900, .5
PROTOCOL = Path(__file__).parents[3] / "docs/complementary_memory/real_processor_mechanics.md"
SYNTHETIC = {"question": "Mechanics check: select one placeholder.",
             "options": ["Placeholder one", "Placeholder two", "Placeholder three"], "answer": 0}


def read(path):
    return json.loads(Path(path).read_text())


def prepare(out):
    base = read(REFERENCE / "inputs.json")
    if (base["id"] != REFERENCE_ID or digest({k: v for k, v in base.items() if k != "id"}) != REFERENCE_ID
            or base["weights"] != "plain" or base["code"] != test_a.code_identity()
            or base["versions"] != test_a.versions()):
        raise ValueError("unchanged Test A inputs/helpers/runtime required")
    if (checkpoint_identity(Path(base["model"])) != base["model_identity"]
            or checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]):
        raise ValueError("frozen base/Plain LoRA changed")
    sources = []
    for source in test_a.selected_sources(base, True):
        # Do not copy real question/options/gold into this mechanics manifest.
        sources.append({k: source[k] for k in ("source_group", "video", "video_sha256", "metadata",
                                              "metadata_sha256", "sampled_frame_indices", "sampled_frame_times")})
        sources[-1]["schedule_end"] = max(q["query_time"] for q in source["queries"])
    if len(sources) != 2 or any(len(s["sampled_frame_times"]) < STEPS for s in sources):
        raise ValueError("exactly the original two smoke sources with >=16 steps required")
    value = {"protocol": "delayed-history-real-mechanics-v1", "reference_input": REFERENCE_ID,
             "sources": sources, "model": base["model"], "plain": base["plain"], "pixels": base["pixels"],
             "steps": STEPS, "synthetic_prompt": SYNTHETIC, "ceiling_seconds": CEILING_SECONDS,
             "gates": {"positions_exact": True, "embeddings_bit_exact": True,
                       "max_option_logit_gap": LOGIT_TOLERANCE, "deterministic_reanswer": True},
             "protocol_sha256": sha256(PROTOCOL), "code": code_identity(), "versions": test_a.versions()}
    value["id"] = digest(value)
    atomic_json(out / "inputs.json", value)
    return value


@torch.no_grad()
def audit_source(bb, source, pixels_per_frame):
    started = time.monotonic()
    if sha256(source["video"]) != source["video_sha256"] or sha256(source["metadata"]) != source["metadata_sha256"]:
        raise ValueError("old smoke source media changed")
    metadata = read(source["metadata"])
    indices, times, _ = longstream.schedule(metadata, source["schedule_end"])
    if indices.tolist() != source["sampled_frame_indices"] or times.tolist() != source["sampled_frame_times"]:
        raise ValueError("old causal schedule changed")
    indices, times = indices[:STEPS * 2], times[:STEPS]
    meta = metadata["streams"][0]
    fps = float(Fraction(meta["avg_frame_rate"]))
    hw = frame_size(meta["height"], meta["width"], pixels_per_frame, 32)
    frames, raw, processor_values, processor_grids = [], [], [], []
    pixels = longstream.stream_pixels(source["video"], indices, *hw)
    try:
        for i, image in enumerate(pixels):
            values, grid = bb.preprocess(image)
            processor_values.append(values.cpu())
            processor_grids.append(grid.cpu())
            raw.append(VisualStep(bb.encode(values, grid), bb.merged_grid(grid),
                                  float(times[i][-1]), tuple(map(float, times[i]))))
            frames.append(image)
    finally:
        pixels.close()
    if len(raw) != STEPS:
        raise ValueError("incomplete mechanics prefix")
    images = np.stack(frames)
    query = {**SYNTHETIC, "query_time": float(times[-1][-1])}
    observation = {"source_fps": fps, "source_frame_count": len(metadata["frames"]),
                   "frame_indices": indices.tolist(),
                   "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    layout = prepare_native(bb.processor, images, times, query, observation,
                            allow_native_timestamp_approximation=True)
    pixels_equal = torch.equal(torch.cat(processor_values), layout.inputs["pixel_values_videos"].cpu())
    grid_equal = all(torch.equal(g[0, 1:], layout.inputs["video_grid_thw"][0, 1:])
                     and int(g[0, 0]) == 1 for g in processor_grids)
    if not pixels_equal or not grid_equal:
        return {"completed": True, "passed": False, "source_group": source["source_group"],
                "steps": STEPS, "processor_pixels_bit_exact": pixels_equal, "processor_grids_exact": grid_equal,
                "observation": layout.audit, "error": "per-step/full-processor pixels or grids differ"}
    first_stamp = test_a.stamp_ids(bb, indices[:2], fps)
    head, tail = segment_layout(layout.inputs["input_ids"][0].tolist(), bb.video_token,
                                bb.vision_start, bb.vision_end, first_stamp)
    arrivals = [DenseArrival(i, s.end, tuple(test_a.stamp_ids(bb, indices[2*i:2*i+2], fps)), s.tokens, s.grid)
                for i, s in enumerate(raw)]
    record = {**query, "native_layout": layout}
    variants = {}
    for name, history_grid, capacity in (("fine32", (4, 4), 32), ("coarse128", (2, 2), 128)):
        grids = [history_grid] * 8 + [(4, 4)] * 8
        expected, native_pos, _ = mixed_native_prefix(bb, raw, record, grids)
        counter = PositionCounter()
        pieces = [(bb.embed_ids(head), counter.text(len(head), bb.device))]
        pieces.extend(block_inputs(bb, feature_block(s, g), counter) for s, g in zip(arrivals, grids, strict=True))
        pieces.append((bb.embed_ids(tail), counter.text(len(tail), bb.device)))
        actual = torch.cat([x for x, _ in pieces], 1)
        positions = torch.cat([p for _, p in pieces], 1)
        shape_ok = actual.shape == expected.shape and positions.shape == native_pos.shape
        same_embeds = shape_ok and torch.equal(actual, expected)
        same_positions = shape_ok and torch.equal(positions, native_pos)
        result = {"history_grid": list(history_grid), "prefix_tokens_including_tail": expected.shape[1],
                  "shape_equal": shape_ok, "embeddings_bit_exact": same_embeds, "positions_exact": same_positions,
                  "max_embedding_abs_gap": float((actual.float() - expected.float()).abs().max()) if shape_ok else None}
        variants[name] = result
        if not same_embeds or not same_positions:
            # Save measurements, but never silently weaken this first-attempt gate.
            continue
        reader = DelayedReader(bb, head, capacity=capacity, history_grid=history_grid, recent=8, seed=17)
        for item in arrivals:
            reader.append(item)
        pred, audit = reader.answer(tail, 3, query_time=query["query_time"])
        again, _ = reader.answer(tail, 3, query_time=query["query_time"])
        hidden = bb.lm(expected, native_pos, cache=None, use_cache=False)[0, -1]
        letters = [bb.ids(c) for c in "ABC"]
        if any(len(ids) != 1 for ids in letters):
            raise ValueError("single-token synthetic options required")
        direct = bb.lm_head.weight[[ids[0] for ids in letters]].float() @ hidden.float()
        cached = torch.tensor(pred["option_logits_fp32"], device=direct.device)
        gap = float((direct - cached).abs().max())
        result.update(max_option_logit_gap=gap, deterministic_reanswer=pred == again,
                      readout=audit, passed=bool(torch.isfinite(direct).all()) and gap <= LOGIT_TOLERANCE
                      and pred == again and audit["committed_steps"] == 8
                      and audit["readout_recent_indices"] == list(range(8, 16))
                      and audit["readout_kv_tokens"] == expected.shape[1] - len(tail))
        del reader, expected, actual, pieces
    torch.cuda.synchronize()
    return {"completed": True, "passed": all(v.get("passed", False) for v in variants.values()),
            "source_group": source["source_group"], "steps": STEPS, "variants": variants,
            "processor_pixels_bit_exact": pixels_equal, "processor_grids_exact": grid_equal,
            "observation": layout.audit, "question_sha256": digest(question_and_answer(query)[0]),
            "seconds": time.monotonic() - started,
            "scope": "Synthetic prompt; no real-question logits, choices, accuracy or NLL are saved."}


def worker(out):
    inputs = read(out / "inputs.json")
    if (inputs["id"] != digest({k: v for k, v in inputs.items() if k != "id"})
            or inputs["code"] != code_identity() or inputs["versions"] != test_a.versions()
            or inputs["protocol_sha256"] != sha256(PROTOCOL)):
        raise ValueError("mechanics inputs/code/runtime/protocol changed")
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    bb = Backbone(BackboneConfig(name=inputs["model"], dtype="bfloat16", local_files_only=True),
                  StreamConfig(pixels_per_frame=inputs["pixels"]), device="cuda:0")
    bb.add_lora(8, 16, dropout=0., adapter=str(Path(inputs["plain"]) / "adapter"))
    bb.model.eval().requires_grad_(False)
    rows = []
    for source in inputs["sources"]:
        row = audit_source(bb, source, inputs["pixels"])
        atomic_json(out / "sources" / f"{digest(source['source_group'])}.json", row)
        rows.append(row)
        print(json.dumps({"source_group": source["source_group"], "mechanics_passed": row["passed"]}), flush=True)
    report = {"completed": True, "passed": all(r["passed"] for r in rows), "input_id": inputs["id"],
              "sources": len(rows), "steps_per_source": STEPS, "quality_evaluation": False,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    atomic_json(out / "mechanics-report.json", report)
    if not report["passed"]:
        raise RuntimeError("mechanics gate failed; inspect saved source measurements, do not relax thresholds")


def run(out, gpu):
    import fcntl

    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (out / ".attempt").open("x") as attempt:
        attempt.write("One bounded attempt; failed runs cannot reset their deadline.\n")
    status = {"stage": "running", "started_unix": time.time(), "ceiling_seconds": CEILING_SECONDS}
    atomic_json(out / "run-status.json", status)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TRACE_WANDB": "0",
           "TRACE_SWANLAB": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    try:
        with budget(CEILING_SECONDS) as deadline:
            inputs = prepare(out)
            check = "import torch; assert torch.cuda.device_count()==1; assert torch.cuda.mem_get_info()[0]>24*2**30"
            run_jobs([("gpu-check", [sys.executable, "-c", check], [gpu])], env, out / "logs", deadline,
                     out / "gpu-check.json")
            command = [sys.executable, "-m", "trace_vl.delayed_history.real_audit", "worker", "--out", str(out)]
            run_jobs([("mechanics", command, [gpu])], env, out / "logs", deadline, out / "worker.json")
            report = read(out / "mechanics-report.json")
            if not report["completed"] or not report["passed"] or report["input_id"] != inputs["id"]:
                raise RuntimeError("incomplete/mismatched mechanics report")
            status.update(stage="completed", input_id=inputs["id"], report_sha256=sha256(out / "mechanics-report.json"))
    except BaseException as exc:
        status.update(stage="held_or_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["finished_unix"] = time.time()
        atomic_json(out / "run-status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "worker"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--execute-mechanics-only", action="store_true")
    args = parser.parse_args()
    if args.stage == "run":
        if not args.execute_mechanics_only or args.gpu < 0:
            parser.error("explicit --execute-mechanics-only and nonnegative GPU required")
        run(args.out.resolve(), args.gpu)
    else:
        worker(args.out.resolve())


if __name__ == "__main__":
    main()
