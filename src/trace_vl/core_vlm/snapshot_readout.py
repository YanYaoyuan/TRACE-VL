"""Offline spatial/readout capacity control; this is not a streaming-memory result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream
from .control import checkpoint_identity
from .data import decode_frame_indices, sha256
from .experiment import run_jobs
from .native import NativeVideoQALoss, prepare_native
from .prepare import digest, write_json
from .qa import VisualStep, grid_pool
from .train import checked_checkpoint

NATURAL_ID = "7b82c34012d3e0fd8c17b115a97ce4c1abc67229a755358df88dccc60cec4412"
DELAYED_ID = "2b4ca77baee10a9d84f293f9bb0d6502e9577da00c573053afb1c31239c594c5"
SMOKE_ID = "Real-Time Visual Understanding_sample_199_2"
ARMS = tuple(f"{mode}_{prompt}" for prompt in ("original", "anchored") for mode in ("text", "grid", "dense"))
REPLAY_KEYS = ("choice", "choice_fp32", "option_logits", "option_logits_fp32", "visual_tokens",
               "read_frame_indices", "read_times", "real_feature_rgb_hashes", "source_video_sha256")
SCOPE = (
    "Offline readout capacity diagnostic, all9 original targets/8 existing development sources. "
    "Exact logged16 frames per target, original versus anchored prompt, text/grid4x4/full native spatial. "
    "Same frozen Plain LoRA2500 and bf16 backbone; fp32 final letter head primary, bf16 secondary. "
    "Full spatial consumes more than128 visual tokens; disclose actual counts. Pixel retrieval and "
    "re-encoding explicitly allowed in this reference. Not a streaming, matched-budget, memory-method "
    "or official benchmark result. No outcome filtering, training, audio or subtitles."
)


def verify_replay(actual, expected):
    if any(actual[k] != expected[k] for k in REPLAY_KEYS):
        raise ValueError("original control did not exactly replay; do not interpret dense results")


def fast_frame(path, timestamp, hw):
    h, w = hw
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1", "-noautorotate",
         "-ss", str(max(0.0, timestamp - 1e-5)), "-i", str(path), "-map", "0:v:0", "-an", "-frames:v", "1",
         "-vf", f"scale={w}:{h}", "-filter_threads", "1", "-threads", "1", "-pix_fmt", "rgb24",
         "-vsync", "0", "-f", "rawvideo", "pipe:1"], timeout=60, stderr=subprocess.PIPE)
    if len(raw) != h * w * 3:
        raise ValueError("seek did not return one complete frame")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()


def verified_pixels(video, prediction, hw):
    indices = prediction["read_frame_indices"]
    times = [t for pair in prediction["read_times"] for t in pair]

    def matches(images):
        return [hashlib.sha256(pair.tobytes()).hexdigest() for pair in images] == prediction["real_feature_rgb_hashes"]

    fast = True
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pixels = np.stack(list(pool.map(lambda t: fast_frame(video, t, hw), times))).reshape(-1, 2, *hw, 3)
        if not matches(pixels):
            raise ValueError("seek RGB differs from original input")
    except (ValueError, subprocess.SubprocessError):
        fast = False
        pixels = decode_frame_indices(video, indices, *hw, 900, stop_after_selection=True).reshape(-1, 2, *hw, 3)
    if not matches(pixels):
        raise ValueError("reference RGB differs from actual original model input")
    return pixels, fast


@torch.no_grad()
def dense_answer(bb, steps, selected, query, metadata, hw, video_hash):
    if len(steps) != len(selected) or not 1 <= len(steps) <= 8:
        raise ValueError("invalid snapshot size")
    images = np.zeros((len(steps), 2, *hw, 3), dtype=np.uint8)
    times = np.asarray([s.times for s in selected])
    indices = [i for s in selected for i in s.indices]
    audit = {"source_fps": float(Fraction(metadata["streams"][0]["avg_frame_rate"])),
             "source_frame_count": len(metadata["frames"]), "frame_indices": indices,
             "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    safe = {**query, "answer": 0}
    safe["native_layout"] = prepare_native(bb.processor, images, times, safe, audit,
                                            allow_native_timestamp_approximation=True)
    qa = NativeVideoQALoss(bb, mode="full")
    embeds, positions, _, _, info = qa.prefix(steps, safe)
    hidden = bb.lm(embeds, positions, cache=None, use_cache=False)[:, -1:]
    candidates = [bb.ids(chr(65 + i)) for i in range(len(query["options"]))]
    if any(len(ids) != 1 for ids in candidates):
        raise ValueError("single-token answer letters required")
    ids = [x[0] for x in candidates]
    logits = bb.logits(hidden)[0, 0, ids].float()
    fp32 = bb.lm_head.weight[ids].float() @ hidden[0, 0].float()
    count = sum(len(s.tokens) for s in steps)
    if info["visual_tokens_total"] != count:
        raise ValueError("full spatial packing lost native tokens")
    return {"choice": int(logits.argmax()), "choice_fp32": int(fp32.argmax()),
            "option_logits": logits.tolist(), "option_logits_fp32": fp32.tolist(), "visual_tokens": count,
            "read_frame_indices": indices, "read_times": times.tolist(),
            "real_feature_rgb_hashes": [s.rgb_sha256 for s in selected], "source_video_sha256": video_hash,
            "layout_uses_zero_pixels_only_for_packing": True, "query_time_encoder_calls": 0,
            "scope": "Full spatial offline capacity reference, larger read budget; features re-encoded before answers, not a streaming result."}


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    natural = json.loads((args.natural / "inputs.json").read_text())
    delayed = json.loads((args.delayed / "inputs.json").read_text())
    for root, inputs, identity, report_sha in (
        (args.natural, natural, NATURAL_ID, "2aa21fb3b1f6cd927fcaafd0e545234603082f7b1f32044c81f5cf2b18e493bf"),
        (args.delayed, delayed, DELAYED_ID, "469ae5c82e0a86bb13077589028b41d2a9f09468aa95d3b0cede36c3864c8643"),
    ):
        if (inputs["id"] != identity or digest({k: v for k, v in inputs.items() if k != "id"}) != identity
                or sha256(root / "report.json") != report_sha or not json.loads((root / "report.json").read_text())["completed"]):
            raise ValueError("completed reference identity changed")
    if natural["checkpoint"] != delayed["checkpoint"] or natural["model_identity"] != delayed["model_identity"]:
        raise ValueError("reference weights differ")
    anchored = {q["target_id"]: q for s in delayed["sources"] for q in s["queries"] if q["delay_seconds"] == 0}
    records = []
    with tarfile.open(args.natural / "reports.tar.gz") as na, tarfile.open(args.delayed / "reports.tar.gz") as da:
        for source in natural["sources"]:
            for q in source["queries"]:
                a = anchored[q["id"]]
                original_path = f"questions/{digest(q['id'])}.json"
                anchor_path = f"questions/{digest(a['id'])}.json"
                original_raw = (args.natural / original_path).read_bytes()
                anchor_raw = (args.delayed / anchor_path).read_bytes()
                if na.extractfile(original_path).read() != original_raw or da.extractfile(anchor_path).read() != anchor_raw:
                    raise ValueError("reference predictions differ from immutable archive")
                original = json.loads(original_raw)["predictions"]
                anchor = json.loads(anchor_raw)["predictions"]
                if any(original["recent8"][k] != anchor["protected_snapshot8"][k] for k in
                       ("read_frame_indices", "read_times", "real_feature_rgb_hashes")):
                    raise ValueError("reference pixels do not match")
                records.append({"id": q["id"], "source_group": source["source_group"],
                                "video": source["video"], "video_sha256": source["video_sha256"],
                                "metadata": source["metadata"], "metadata_sha256": source["metadata_sha256"],
                                "queries": {"original": q, "anchored": a},
                                "expected": {"text_original": original["text_only"], "grid_original": original["recent8"],
                                             "text_anchored": anchor["text_only"], "grid_anchored": anchor["protected_snapshot8"]},
                                "reference_question_sha256": {"natural": hashlib.sha256(original_raw).hexdigest(),
                                                              "delayed": hashlib.sha256(anchor_raw).hexdigest()}})
    if len(records) != 9 or len({r["source_group"] for r in records}) != 8:
        raise ValueError("all9 targets/8 sources must survive")
    inputs = {"protocol": "offline-snapshot-readout-v1", "scope": SCOPE, "questions": records,
              "model": natural["model"], "model_identity": natural["model_identity"], "plain": natural["plain"],
              "checkpoint": natural["checkpoint"], "pixels": natural["pixels"],
              "primary_score": natural["primary_score"], "secondary_score": natural["secondary_score"],
              "natural": str(args.natural), "natural_input_id": NATURAL_ID,
              "delayed": str(args.delayed), "delayed_input_id": DELAYED_ID,
              "source_code_sha256": sha256(__file__), "longstream_code_sha256": sha256(longstream.__file__)}
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)


@torch.no_grad()
def evaluate_question(bb, inputs, record, out, verify):
    if sha256(record["video"]) != record["video_sha256"] or sha256(record["metadata"]) != record["metadata_sha256"]:
        raise ValueError("source identity changed")
    meta = json.loads(Path(record["metadata"]).read_text())
    stream = meta["streams"][0]
    hw = frame_size(stream["height"], stream["width"], inputs["pixels"], 32)
    expected = record["expected"]["grid_original"]
    pixels, fast = verified_pixels(record["video"], expected, hw)
    steps, selected = [], []
    for i, frames in enumerate(pixels):
        values, grid = bb.preprocess(frames)
        dense = bb.encode(values, grid).detach()
        shape = bb.merged_grid(grid)
        pair = tuple(expected["read_times"][i])
        raw = VisualStep(dense, shape, pair[-1], pair)
        steps.append(raw)
        selected.append(longstream.StoredStep(grid_pool(dense, shape, (4, 4))[0].detach().clone(), shape, pair,
                                             tuple(expected["read_frame_indices"][2*i:2*i+2]), expected["real_feature_rgb_hashes"][i]))
        if verify and i == 0:
            query = {**record["queries"]["original"], "answer": 0}
            audit = {"source_fps": float(Fraction(stream["avg_frame_rate"])), "source_frame_count": len(meta["frames"]),
                     "frame_indices": expected["read_frame_indices"][:2],
                     "decoded_rgb_sha256": hashlib.sha256(frames[None].tobytes()).hexdigest()}
            layout = prepare_native(bb.processor, frames[None], np.asarray([pair]), query, audit,
                                    allow_native_timestamp_approximation=True)
            write_json(out / f"native-parity-{digest(record['id'])}.json", longstream.verify_native_step(bb, layout, raw, query))
    if len(steps) != 8:
        raise ValueError("reference must contain8 two-frame steps")
    predictions = {}
    qa = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    for prompt, query in record["queries"].items():
        predictions["text_" + prompt] = longstream.text_answer(bb, query, record["video_sha256"])
        predictions["grid_" + prompt] = longstream.answer(bb, qa, selected, query, meta, hw, record["video_sha256"])
        for mode in ("text", "grid"):
            arm = mode + "_" + prompt
            verify_replay(predictions[arm], record["expected"][arm])
        predictions["dense_" + prompt] = dense_answer(bb, steps, selected, query, meta, hw, record["video_sha256"])
    write_json(out / "questions" / f"{digest(record['id'])}.json", {
        "completed": True, "id": record["id"], "source_group": record["source_group"], "input_id": inputs["id"],
        "gold": record["queries"]["original"]["answer"], "predictions": predictions, "exact_control_replay": True,
        "offline_vision_encoder_calls": 8, "verification_extra_encoder_calls": int(verify),
        "logged_rgb_pairs_verified": 8, "fast_seek_verified": fast,
        "dense_feature_payload_bytes": sum(s.tokens.numel() * s.tokens.element_size() for s in steps),
        "compressed_feature_payload_bytes": sum(s.tokens.numel() * s.tokens.element_size() for s in selected),
        "peak16_frame_rgb_bytes": int(pixels.nbytes), "scope": SCOPE})
    print(json.dumps({"completed_question": record["id"], "control_replay": True}), flush=True)


def worker(args):
    if not 0 <= args.shard < 4:
        raise ValueError("invalid shard")
    inputs = json.loads((args.out / "inputs.json").read_text())
    if inputs["source_code_sha256"] != sha256(__file__) or inputs["longstream_code_sha256"] != sha256(longstream.__file__):
        raise ValueError("worker source changed")
    if checked_checkpoint(Path(inputs["plain"]), inputs["checkpoint"]["run"]) != inputs["checkpoint"]:
        raise ValueError("frozen checkpoint changed")
    if checkpoint_identity(Path(inputs["model"])) != inputs["model_identity"]:
        raise ValueError("base weights changed")
    selected = inputs["questions"][args.shard::4]
    if args.smoke:
        selected = [r for r in selected if r["id"] == SMOKE_ID]
    if not selected:
        return
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    bb = Backbone(BackboneConfig(name=inputs["model"], dtype="bfloat16", local_files_only=True),
                  StreamConfig(pixels_per_frame=inputs["pixels"]), device="cuda:0")
    bb.add_lora(8, 16, dropout=0.0, adapter=str(Path(inputs["plain"]) / "adapter"))
    bb.model.eval().requires_grad_(False)
    native_done = any((args.out / f"native-parity-{digest(r['id'])}.json").exists() for r in selected)
    for record in selected:
        path = args.out / "questions" / f"{digest(record['id'])}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["input_id"] != inputs["id"] or not old["completed"]:
                raise ValueError("saved question belongs to another run")
            continue
        evaluate_question(bb, inputs, record, args.out, verify=not native_done)
        native_done = True


def report(args):
    inputs = json.loads((args.out / "inputs.json").read_text())
    selected = [r for r in inputs["questions"] if not args.smoke or r["id"] == SMOKE_ID]
    rows, precision = [], []
    for record in selected:
        row = json.loads((args.out / "questions" / f"{digest(record['id'])}.json").read_text())
        if (not row["completed"] or not row["exact_control_replay"] or row["input_id"] != inputs["id"]
                or row["id"] != record["id"] or row["gold"] != record["queries"]["original"]["answer"]
                or row["offline_vision_encoder_calls"] != 8 or row["logged_rgb_pairs_verified"] != 8
                or set(row["predictions"]) != set(ARMS)):
            raise ValueError("reference identity, decode or count check failed")
        for arm, p in row["predictions"].items():
            if arm in record["expected"]:
                verify_replay(p, record["expected"][arm])
            original = record["expected"]["grid_original"]
            if not arm.startswith("text_"):
                if any(p[k] != original[k] for k in ("read_frame_indices", "read_times", "real_feature_rgb_hashes")):
                    raise ValueError("spatial conditions changed actual pixels")
                if arm.startswith("grid_") and p["visual_tokens"] != 128:
                    raise ValueError("compressed control budget changed")
                if arm.startswith("dense_") and p["visual_tokens"] <= 128:
                    raise ValueError("native spatial capacity reference did not increase read budget")
            for key, choice in (("option_logits", "choice"), ("option_logits_fp32", "choice_fp32")):
                logits = p[key]
                if (len(logits) != len(record["queries"]["original"]["options"])
                        or not all(math.isfinite(v) for v in logits)
                        or max(range(len(logits)), key=logits.__getitem__) != p[choice]):
                    raise ValueError("candidate scoring failed")
            if p["choice"] != p["choice_fp32"]:
                precision.append({"id": row["id"], "arm": arm, "bf16": p["choice"], "fp32": p["choice_fp32"]})
        rows.append(row)
    write_json(args.out / ("smoke-ok.json" if args.smoke else "report.json"), {
        "completed": True, "smoke": args.smoke, "input_id": inputs["id"], "scope": SCOPE,
        "questions": len(rows), "source_videos": len({r["source_group"] for r in rows}),
        "predictions": len(rows) * len(ARMS), "primary_score": inputs["primary_score"],
        "scores": {a: {"correct_fp32": sum(r["predictions"][a]["choice_fp32"] == r["gold"] for r in rows),
                       "correct_bf16": sum(r["predictions"][a]["choice"] == r["gold"] for r in rows),
                       "visual_tokens": sorted({r["predictions"][a]["visual_tokens"] for r in rows})} for a in ARMS},
        "offline_vision_encoder_calls": sum(r["offline_vision_encoder_calls"] for r in rows),
        "verification_extra_encoder_calls": sum(r["verification_extra_encoder_calls"] for r in rows),
        "precision_differences": precision, "all_controls_exactly_replayed": True,
        "architecture_accepted": False, "no_training": True})


def run(args):
    prepare(args)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TRACE_WANDB": "0",
           "TRACE_SWANLAB": "0", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    run_jobs([("gpu-check", [sys.executable, "-c", "import torch; assert torch.cuda.device_count()==4; "
                            "assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24"], list(range(4)))],
             env, args.out / "logs")
    command = [sys.executable, "-m", "trace_vl.core_vlm.snapshot_readout", "worker", "--out", str(args.out)]
    run_jobs([(f"smoke-{i}", [*command, "--shard", str(i), "--smoke"], [i]) for i in range(4)], env, args.out / "logs")
    args.smoke = True
    report(args)
    smoke = json.loads((args.out / "smoke-ok.json").read_text())
    if smoke["questions"] != 1 or smoke["predictions"] != 6:
        raise ValueError("smoke must cover both prompts and all three representations")
    run_jobs([(f"full-{i}", [*command, "--shard", str(i)], [i]) for i in range(4)], env, args.out / "logs")
    args.smoke = False
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "worker", "run", "report"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--natural", type=Path, default=Path("/data/yyy/tracevl-runs/core-longstream-natural-c2f39df-001"))
    p.add_argument("--delayed", type=Path, default=Path("/data/yyy/tracevl-runs/core-longstream-delayed-92550e9-001"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    {"prepare": prepare, "worker": worker, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
