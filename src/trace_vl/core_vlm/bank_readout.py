"""Offline reconstruction/readout diagnostic on actual stored reservoir banks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tarfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream
from . import snapshot_readout as reconstruction
from .control import checkpoint_identity
from .data import sha256
from .experiment import run_jobs
from .native import NativeVideoQALoss, prepare_native
from .prepare import digest, write_json
from .qa import VisualStep, grid_pool
from .train import checked_checkpoint

REFERENCE_ID = "fffbd282b72fb0dff10e221ba94dbeb93f4a25c7f21b91f94d4cb6f87d5d594c"
REPORT_SHA256 = "4b10112e4b187c6563b449de0bf2fd58a3ad26e04a8f120addc719f458499350"
ARCHIVE_SHA256 = "a3db3987a515f444d60aca4b060c9ea8a829cfc796dfdd9084f3cfd289d05847"
SEEDS = ("original", "29", "43")
POLICIES = ("latest8", "uniform8", "all_bank")
ARMS = ("text_only", "protected_snapshot8", *(f"{s}_{p}" for s in SEEDS for p in POLICIES))
SCOPE = (
    "Offline bank reconstruction/readout diagnostic, all77 frozen targets at300s and all3reservoir memory seeds. "
    "Exact original prompts/options/gold and actual stored bank membership; only pre-target frames. "
    "Reconstruct original pooled values by verified original RGB pairs; re-encoding explicitly allowed OFFLINE. "
    "Original latest8/128token outputs, text and privileged snapshot must exactly replay. Uniform8 same read "
    "budget, all_bank up to32steps/512tokens is a separately labeled larger-read-budget diagnostic. "
    "Not a streaming-memory/latency, equal-budget fullbank method, official benchmark or training result. "
    "One frozen PlainLoRA2500; fp32 letter head over bf16 hidden primary, bf16 secondary. "
    "No new reserved sources, correctness filtering, audio/subtitles, or post-target frames. "
    "Shared offline reconstruction cache is diagnostic infrastructure, not a deployable memory bank."
)


def select_slots(slots, target, policy):
    if policy not in POLICIES or not math.isfinite(target) or target < 0:
        raise ValueError("invalid read policy/cutoff")
    by_id = {tuple(s["indices"]): s for s in slots}
    if len(by_id) != len(slots) or len(slots) > 36:
        raise ValueError("invalid stored bank population")
    available = sorted((s for s in slots if s["times"][-1] <= target), key=lambda s: s["times"][-1])
    if len(available) > 32:
        raise ValueError("300second delayed historical read cannot exceed32 slots")
    if policy == "latest8":
        return available[-8:]
    if policy == "all_bank":
        return available
    return (
        [available[i] for i in np.linspace(0, len(available) - 1, min(8, len(available))).round().astype(int)]
        if available
        else []
    )


def union_slots(record):
    # The cache may also hold protected controls. Selectors NEVER read this union.
    slots = {}
    for item in [
        *record["snapshot"],
        *(
            s
            for bank in record["banks"].values()
            for s in select_slots(bank, record["query"]["target_time"], "all_bank")
        ),
    ]:
        key = tuple(item["indices"])
        if key in slots and slots[key] != item:
            raise ValueError("conflicting stored RGB/time identity")
        slots[key] = item
    return sorted(slots.values(), key=lambda s: s["times"][-1])


def reconstruction_view(slots):
    return {
        "read_frame_indices": [i for s in slots for i in s["indices"]],
        "read_times": [s["times"] for s in slots],
        "real_feature_rgb_hashes": [s["rgb_sha256"] for s in slots],
    }


def verify_prediction(actual, expected):
    if actual != expected:
        raise ValueError("original complete candidate prediction did not exactly replay")


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.reference / "inputs.json").read_text())
    if (
        base["id"] != REFERENCE_ID
        or digest({k: v for k, v in base.items() if k != "id"}) != REFERENCE_ID
        or sha256(args.reference / "report.json") != REPORT_SHA256
        or sha256(args.reference / "reports.tar.gz") != ARCHIVE_SHA256
    ):
        raise ValueError("complete writer reference changed")
    prior = json.loads((args.reference / "report.json").read_text())
    if not prior["completed"] or prior["questions"] != 385 or prior["source_videos"] != 77:
        raise ValueError("all77-source385query reference required")
    if checked_checkpoint(Path(base["plain"]), base["checkpoint"]["run"]) != base["checkpoint"]:
        raise ValueError("frozen checkpoint changed")
    if checkpoint_identity(Path(base["model"])) != base["model_identity"]:
        raise ValueError("base model changed")
    records = []
    archive_expected = {}
    for source in base["sources"]:
        queries = [q for q in source["queries"] if q["delay_seconds"] == 300]
        if len(queries) != 1:
            raise ValueError("one unchanged300s query per source required")
        q = queries[0]
        name = "questions/" + digest(q["id"]) + ".json"
        path = args.reference / name
        raw = path.read_bytes()
        r = json.loads(raw)
        archive_expected[name] = hashlib.sha256(raw).hexdigest()
        memory = r["memory"]
        new = memory["writer_controls"]["banks"]
        banks = {
            "original": memory["retained"]["reservoir"] + memory["retained"]["recent8_control"][-4:],
            "29": new["reservoir_seed29_time8"]["retained"]["history"]
            + new["reservoir_seed29_time8"]["retained"]["recent"],
            "43": new["reservoir_seed43_time8"]["retained"]["history"]
            + new["reservoir_seed43_time8"]["retained"]["recent"],
        }
        expected = {
            "text_only": r["predictions"]["text_only"],
            "protected_snapshot8": r["predictions"]["protected_snapshot8"],
            "original_latest8": r["predictions"]["reservoir_time8"],
            "29_latest8": r["predictions"]["reservoir_seed29_time8"],
            "43_latest8": r["predictions"]["reservoir_seed43_time8"],
        }
        record = {
            k: copy.deepcopy(source[k])
            for k in ("source_group", "video", "video_sha256", "metadata", "metadata_sha256")
        }
        record.update(
            id=q["id"],
            query=copy.deepcopy(q),
            banks=banks,
            snapshot=memory["retained"]["snapshot:" + q["target_id"]],
            expected=expected,
            reference_record_sha256=archive_expected[name],
        )
        for seed in SEEDS:
            view = reconstruction_view(select_slots(banks[seed], q["target_time"], "latest8"))
            if any(view[k] != expected[seed + "_latest8"][k] for k in view):
                raise ValueError("stored bank does not reproduce original read")
        selected = union_slots(record)
        if any(s["times"][-1] > q["target_time"] for s in selected):
            raise ValueError("offline reconstruction saw target future")
        record["planned_offline_encoder_calls"] = len(selected)
        records.append(record)
    found = set()
    with tarfile.open(args.reference / "reports.tar.gz", "r|gz") as archive:
        for member in archive:
            if member.name in archive_expected:
                if (
                    hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                    != archive_expected[member.name]
                ):
                    raise ValueError("reference record differs from complete archive")
                found.add(member.name)
    if (
        found != set(archive_expected)
        or len(records) != 77
        or len({r["source_group"] for r in records}) != 77
    ):
        raise ValueError("complete77-source population/archive required")
    shortest = min(records, key=lambda r: (r["planned_offline_encoder_calls"], r["id"]))
    widest = max(
        records,
        key=lambda r: (
            max(len(select_slots(v, r["query"]["target_time"], "all_bank")) for v in r["banks"].values()),
            r["id"],
        ),
    )
    if shortest["id"] == widest["id"]:
        raise ValueError("distinct sparse/wide smoke targets required")
    inputs = {
        k: copy.deepcopy(base[k])
        for k in (
            "model",
            "model_identity",
            "plain",
            "checkpoint",
            "pixels",
            "primary_score",
            "secondary_score",
        )
    }
    inputs.update(
        protocol="offline-reservoir-bank-readout-v1",
        scope=SCOPE,
        records=records,
        targets=77,
        sources=77,
        predictions=847,
        arms=list(ARMS),
        reference=str(args.reference),
        reference_input_id=REFERENCE_ID,
        reference_report_sha256=REPORT_SHA256,
        reference_archive_sha256=ARCHIVE_SHA256,
        smoke_ids=sorted([shortest["id"], widest["id"]]),
        source_code_sha256=sha256(__file__),
        longstream_code_sha256=sha256(longstream.__file__),
        reconstruction_code_sha256=sha256(reconstruction.__file__),
        analysis_plan={
            "same_budget_primary": "mean3seed uniform8 minus mean3seed original latest8 within each source",
            "larger_budget_diagnostic": "mean3seed all pretarget bank minus latest8; actual16tokens/step, not equal-budgetmethod",
            "bootstrap_unit": "77sources with seeds averaged within source",
            "draws": 4000,
            "seed": 7,
            "accuracy_smoke_gate": False,
            "streaming_claim": False,
        },
    )
    inputs["id"] = digest(inputs)
    ensure_run_manifest(args.out / "inputs.json", inputs)
    write_json(
        args.out / "data-audit.json",
        {
            "completed": True,
            "input_id": inputs["id"],
            "targets": 77,
            "sources": 77,
            "planned_offline_encoder_calls": sum(r["planned_offline_encoder_calls"] for r in records),
            "all_recorded_stores_and_queries_frozen": True,
            "original_complete_prediction_replays_required": 385,
            "new_reserved_sources": 0,
            "scope": SCOPE,
        },
    )


@torch.no_grad()
def evaluate_record(bb, inputs, record, out, verify):
    if (
        sha256(record["video"]) != record["video_sha256"]
        or sha256(record["metadata"]) != record["metadata_sha256"]
    ):
        raise ValueError("source bytes/metadata changed")
    metadata = json.loads(Path(record["metadata"]).read_text())
    stream = metadata["streams"][0]
    hw = frame_size(stream["height"], stream["width"], inputs["pixels"], 32)
    slots = union_slots(record)
    pixels, fast = reconstruction.verified_pixels(record["video"], reconstruction_view(slots), hw)
    cache = {}
    for i, (item, frames) in enumerate(zip(slots, pixels, strict=True)):
        values, grid = bb.preprocess(frames)
        dense = bb.encode(values, grid)
        shape = bb.merged_grid(grid)
        pair = tuple(item["times"])
        cache[tuple(item["indices"])] = longstream.StoredStep(
            grid_pool(dense, shape, (4, 4))[0].detach().clone(),
            shape,
            pair,
            tuple(item["indices"]),
            item["rgb_sha256"],
        )
        if verify and i == 0:
            safe = {**record["query"], "answer": 0}
            raw = VisualStep(dense, shape, pair[-1], pair)
            audit = {
                "source_fps": float(Fraction(stream["avg_frame_rate"])),
                "source_frame_count": len(metadata["frames"]),
                "frame_indices": item["indices"],
                "decoded_rgb_sha256": hashlib.sha256(frames[None].tobytes()).hexdigest(),
            }
            layout = prepare_native(
                bb.processor,
                frames[None],
                np.asarray([pair]),
                safe,
                audit,
                allow_native_timestamp_approximation=True,
            )
            write_json(
                out / f"native-parity-{digest(record['id'])}.json",
                longstream.verify_native_step(bb, layout, raw, safe),
            )
    del pixels
    if len(cache) != record["planned_offline_encoder_calls"]:
        raise ValueError("offline reconstruction count changed")
    qa = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    q = record["query"]
    predictions = {}
    predictions["text_only"] = {
        **longstream.text_answer(bb, q, record["video_sha256"]),
        "empty_visual_memory": True,
    }

    def answer(items, limit=8):
        selected = [cache[tuple(s["indices"])] for s in items]
        if any(s.end > q["target_time"] for s in selected):
            raise ValueError("normal read used post-target frame")
        p = (
            longstream.answer(bb, qa, selected, q, metadata, hw, record["video_sha256"], max_read_steps=limit)
            if selected
            else longstream.text_answer(bb, q, record["video_sha256"])
        )
        return {**p, "empty_visual_memory": not selected}

    predictions["protected_snapshot8"] = answer(record["snapshot"])
    verify_prediction(predictions["text_only"], record["expected"]["text_only"])
    verify_prediction(predictions["protected_snapshot8"], record["expected"]["protected_snapshot8"])
    selections = {}
    for seed in SEEDS:
        for policy in POLICIES:
            arm = seed + "_" + policy
            items = select_slots(record["banks"][seed], q["target_time"], policy)
            selections[arm] = reconstruction_view(items)
            predictions[arm] = answer(items, 32 if policy == "all_bank" else 8)
            if policy == "latest8":
                verify_prediction(predictions[arm], record["expected"][arm])
    write_json(
        out / "questions" / f"{digest(record['id'])}.json",
        {
            "completed": True,
            "input_id": inputs["id"],
            "id": record["id"],
            "target_id": q["target_id"],
            "source_group": record["source_group"],
            "gold": q["answer"],
            "target_time": q["target_time"],
            "arrival_time": q["query_time"],
            "predictions": predictions,
            "selections": selections,
            "exact_control_replays": 5,
            "offline_vision_encoder_calls": len(cache),
            "verification_extra_encoder_calls": int(verify),
            "actual_logged_rgb_pairs_verified": len(cache),
            "fast_seek_verified": fast,
            "scope": SCOPE,
            "reconstruction_feature_payload_bytes": sum(
                s.tokens.numel() * s.tokens.element_size() for s in cache.values()
            ),
        },
    )
    print(
        json.dumps(
            {
                "target": record["id"],
                "offline_steps": len(cache),
                "exact_control_replays": 5,
                "predictions": 11,
            }
        ),
        flush=True,
    )


def check_code(inputs):
    if (
        inputs["source_code_sha256"] != sha256(__file__)
        or inputs["longstream_code_sha256"] != sha256(longstream.__file__)
        or inputs["reconstruction_code_sha256"] != sha256(reconstruction.__file__)
    ):
        raise ValueError("readout diagnostic code changed")


def worker(args):
    if not 0 <= args.shard < 4:
        raise ValueError("invalid shard")
    inputs = json.loads((args.out / "inputs.json").read_text())
    check_code(inputs)
    if checked_checkpoint(Path(inputs["plain"]), inputs["checkpoint"]["run"]) != inputs["checkpoint"]:
        raise ValueError("checkpoint changed")
    if checkpoint_identity(Path(inputs["model"])) != inputs["model_identity"]:
        raise ValueError("model changed")
    records = inputs["records"][args.shard :: 4]
    if args.smoke:
        records = [r for r in records if r["id"] in inputs["smoke_ids"]]
    if not records:
        return
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    bb = Backbone(
        BackboneConfig(name=inputs["model"], dtype="bfloat16", local_files_only=True),
        StreamConfig(pixels_per_frame=inputs["pixels"]),
        device="cuda:0",
    )
    bb.add_lora(8, 16, dropout=0.0, adapter=str(Path(inputs["plain"]) / "adapter"))
    bb.model.eval().requires_grad_(False)
    for index, record in enumerate(records):
        path = args.out / "questions" / f"{digest(record['id'])}.json"
        if path.exists():
            prior = json.loads(path.read_text())
            if prior["input_id"] != inputs["id"] or not prior["completed"]:
                raise ValueError("existing target identity mismatch")
            continue
        evaluate_record(bb, inputs, record, args.out, verify=index == 0)


def report(args):
    inputs = json.loads((args.out / "inputs.json").read_text())
    check_code(inputs)
    records = [r for r in inputs["records"] if not args.smoke or r["id"] in inputs["smoke_ids"]]
    rows = []
    precision = []
    for record in records:
        row = json.loads((args.out / "questions" / f"{digest(record['id'])}.json").read_text())
        if (
            not row["completed"]
            or row["input_id"] != inputs["id"]
            or row["id"] != record["id"]
            or row["source_group"] != record["source_group"]
            or row["target_time"] != record["query"]["target_time"]
            or row["arrival_time"] != record["query"]["query_time"]
            or row["gold"] != record["query"]["answer"]
            or row["exact_control_replays"] != 5
            or row["offline_vision_encoder_calls"] != record["planned_offline_encoder_calls"]
            or set(row["predictions"]) != set(ARMS)
        ):
            raise ValueError("fixed target or reconstruction completion changed")
        for arm, expected in record["expected"].items():
            verify_prediction(row["predictions"][arm], expected)
        for seed in SEEDS:
            for policy in POLICIES:
                arm = seed + "_" + policy
                view = reconstruction_view(
                    select_slots(record["banks"][seed], record["query"]["target_time"], policy)
                )
                p = row["predictions"][arm]
                limit = 512 if policy == "all_bank" else 128
                if (
                    any(p[k] != v for k, v in view.items())
                    or p["visual_tokens"] != 16 * len(view["read_times"])
                    or p["visual_tokens"] > limit
                ):
                    raise ValueError("stored bank selection/read token budget changed")
        for arm, p in row["predictions"].items():
            if (
                p["query_time_encoder_calls"] != 0
                or max((t for pair in p["read_times"] for t in pair), default=0) > row["target_time"]
            ):
                raise ValueError("post-reconstruction answer encoded pixels or saw target future")
            for values, choice in (("option_logits", "choice"), ("option_logits_fp32", "choice_fp32")):
                if (
                    len(p[values]) != len(record["query"]["options"])
                    or not all(math.isfinite(v) for v in p[values])
                    or p[choice] != max(range(len(p[values])), key=p[values].__getitem__)
                ):
                    raise ValueError("candidate scores changed")
            if p["choice"] != p["choice_fp32"]:
                precision.append({"id": row["id"], "arm": arm})
        rows.append(row)
    if not args.smoke and len(rows) != 77:
        raise ValueError("all77targets required")
    paired = {}
    for policy in ("uniform8", "all_bank"):
        differences = np.asarray(
            [
                np.mean(
                    [
                        int(r["predictions"][seed + "_" + policy]["choice_fp32"] == r["gold"])
                        - int(r["predictions"][seed + "_latest8"]["choice_fp32"] == r["gold"])
                        for seed in SEEDS
                    ]
                )
                for r in rows
            ]
        )
        draws = np.random.default_rng(7).choice(differences, (4000, len(rows)), replace=True).mean(axis=1)
        paired[policy] = {
            "delta_pp": float(100 * differences.mean()),
            "ci95_pp": (100 * np.quantile(draws, [0.025, 0.975])).tolist(),
            "sources": len(rows),
            "draws": 4000,
            "seed": 7,
            "seeds_averaged_within_source": True,
            "same_read_budget": policy == "uniform8",
        }
    write_json(
        args.out / ("smoke-ok.json" if args.smoke else "report.json"),
        {
            "completed": True,
            "smoke": args.smoke,
            "input_id": inputs["id"],
            "scope": SCOPE,
            "targets": len(rows),
            "sources": len({r["source_group"] for r in rows}),
            "predictions": 11 * len(rows),
            "exact_control_replays": 5 * len(rows),
            "scores": {
                a: {
                    "correct_fp32": sum(r["predictions"][a]["choice_fp32"] == r["gold"] for r in rows),
                    "correct_bf16": sum(r["predictions"][a]["choice"] == r["gold"] for r in rows),
                    "visual_tokens": [r["predictions"][a]["visual_tokens"] for r in rows],
                }
                for a in ARMS
            },
            "offline_vision_encoder_calls": sum(r["offline_vision_encoder_calls"] for r in rows),
            "verification_extra_encoder_calls": sum(r["verification_extra_encoder_calls"] for r in rows),
            "precision_changes": precision,
            "paired_source_analysis": paired,
            "accuracy_used_as_gate": False,
            "streaming_method_claim": False,
        },
    )


def run(args):
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
    command = [sys.executable, "-m", "trace_vl.core_vlm.bank_readout", "worker", "--out", str(args.out)]
    run_jobs(
        [(f"smoke-{i}", [*command, "--shard", str(i), "--smoke"], [i]) for i in range(4)],
        env,
        args.out / "logs",
    )
    args.smoke = True
    report(args)
    smoke = json.loads((args.out / "smoke-ok.json").read_text())
    if smoke["targets"] != 2 or smoke["exact_control_replays"] != 10:
        raise ValueError("two-source exact replay smoke incomplete")
    if (
        max(
            v
            for arm in smoke["scores"]
            if arm.endswith("_all_bank")
            for v in smoke["scores"][arm]["visual_tokens"]
        )
        <= 128
    ):
        raise ValueError("smoke did not exercise wider-read diagnostic")
    run_jobs([(f"full-{i}", [*command, "--shard", str(i)], [i]) for i in range(4)], env, args.out / "logs")
    args.smoke = False
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "worker", "run", "report"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, default=Path("/data/yyy/tracevl-runs/core-writer-controls-60bd0df-001")
    )
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    {"prepare": prepare, "worker": worker, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
