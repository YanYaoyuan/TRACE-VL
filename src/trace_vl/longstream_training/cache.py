"""Encode each real causal prefix once; retain only a bounded Reservoir bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.core_vlm.control import checkpoint_identity
from trace_vl.core_vlm.data import presentation_frames, probe_video, sha256
from trace_vl.core_vlm.longstream import schedule, stream_pixels
from trace_vl.core_vlm.native import check_native_forward, prepare_native
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.core_vlm.qa import VisualStep, grid_pool
from trace_vl.core_vlm.train import sample_indices
from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from .prepare import rows
from .selection import retained_indices


def build(args):
    audit = json.loads((args.data / "data-audit.json").read_text())
    assert audit["completed"]
    for split, h in audit["manifest_sha256"].items():
        assert sha256(args.data / f"{split}.jsonl") == h
    population = rows(args.data / "train.jsonl") + rows(args.data / "dev.jsonl")
    identity = {
        "data": audit["input_id"],
        "manifests": audit["manifest_sha256"],
        "model": checkpoint_identity(args.model),
        "pixels": 65536,
        "sampling_hz": 2,
        "history": 32,
        "recent": 4,
        "memory_seed": 17,
        "code_sha256": sha256(__file__),
    }
    identity["id"] = digest(identity)
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_run_manifest(args.out / "inputs.json", identity)
    torch.cuda.set_per_process_memory_fraction(0.4)
    if torch.cuda.mem_get_info()[0] < 18 * 2**30:
        raise RuntimeError("co-running pilot needs18GiB free before loading its model")
    bb = Backbone(
        BackboneConfig(name=str(args.model), dtype="bfloat16", local_files_only=True),
        StreamConfig(pixels_per_frame=65536),
        device="cuda:0",
    )
    bb.model.eval().requires_grad_(False)
    queue = population[args.shard :: 2]
    if args.smoke:
        training = rows(args.data / "train.jsonl")
        needed = {training[i]["id"] for u in range(2) for i in sample_indices(len(training), u, 1, 4, 0, 17)}
        needed.add(rows(args.data / "dev.jsonl")[0]["id"])
        queue = [r for r in queue if r["id"] in needed]
    for row in queue:
        path = args.out / f"{digest(row['id'])}.pt"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            old = json.loads(receipt.read_text())
            assert old["input_id"] == identity["id"] and sha256(path) == old["sha256"]
            continue
        if sha256(row["video"]) != row["media_sha256"]:
            raise ValueError("staged media changed")
        meta = probe_video(row["video"], 300)
        pts, good = presentation_frames(meta["frames"], drop_bad_pts=True)
        last = float((pts - pts[0])[good][-1])
        indices, times, dropped = schedule(meta, min(row["query_time"], last))
        if any(pair[-1] > row["query_time"] for pair in times):
            raise ValueError("future observation")
        seed = int(digest(["matched-longstream-reader-v1", row["source_group"], 17])[:8], 16)
        keep = retained_indices(times[:, -1].tolist(), seed=seed)
        kept, rgb, calls = {}, {}, 0
        stream = meta["streams"][0]
        hw = frame_size(stream["height"], stream["width"], 65536, 32)
        pixels = stream_pixels(row["video"], indices, *hw)
        try:
            batch = []
            for index, frames in enumerate(pixels):
                batch.append((index, frames))
                if len(batch) < 8 and index + 1 < len(times):
                    continue
                with torch.no_grad():
                    images = np.stack([f for _, f in batch])
                    prepared = bb.processor.video_processor(
                        videos=[images.reshape(-1, *images.shape[2:])],
                        do_sample_frames=False,
                        do_resize=False,
                        return_tensors="pt",
                    )
                    grid = prepared["video_grid_thw"]
                    if grid.shape != (1, 3) or int(grid[0, 0]) != len(batch):
                        raise ValueError("processor changed temporal batching")
                    dense = bb.encode(prepared["pixel_values_videos"], grid)
                    shape = bb.merged_grid(grid)
                    chunks = dense.split(shape[0] * shape[1])
                    for (i, frames), feature in zip(batch, chunks, strict=True):
                        calls += 1
                        if i in keep:
                            kept[i] = grid_pool(feature, shape, (4, 4))[0].cpu().clone()
                            rgb[i] = hashlib.sha256(frames.tobytes()).hexdigest()
                    if batch[0][0] == 0:
                        # BF16 GEMMs can round differently for one step versus
                        # eight. Compare the bridge with native forward on the
                        # SAME complete temporal batch; never relax tolerance.
                        images = np.stack([f for _, f in batch])
                        bt = times[: len(batch)]
                        native_audit = {
                            "source_fps": float(Fraction(stream["avg_frame_rate"])),
                            "source_frame_count": len(meta["frames"]),
                            "frame_indices": indices[: 2 * len(batch)].tolist(),
                            "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
                        }
                        safe = {**row, "answer": 0}
                        layout = prepare_native(
                            bb.processor,
                            images,
                            bt,
                            safe,
                            native_audit,
                            allow_native_timestamp_approximation=True,
                        )
                        if not torch.equal(
                            layout.inputs["pixel_values_videos"], prepared["pixel_values_videos"]
                        ):
                            raise ValueError("native and cached batch preprocessing differ")
                        raw = [
                            VisualStep(f, shape, float(t[-1]), tuple(t.tolist()))
                            for f, t in zip(chunks, bt, strict=True)
                        ]
                        parity = check_native_forward(bb, layout, raw, {**safe, "native_layout": layout})
                        parity["matched_temporal_batch_steps"] = len(batch)
                batch = []
        finally:
            pixels.close()
        if calls != len(times) or sorted(kept) != keep:
            raise ValueError("stream or bounded bank incomplete")
        entry = {
            "input_id": identity["id"],
            "id": row["id"],
            "features": torch.stack([kept[i] for i in keep]),
            "times": times[keep].tolist(),
            "indices": [indices[2 * i : 2 * i + 2].tolist() for i in keep],
            "rgb_sha256": [rgb[i] for i in keep],
            "grid": shape,
            "hw": hw,
            "fps": float(Fraction(stream["avg_frame_rate"])),
            "frame_count": len(meta["frames"]),
            "stream_steps": len(times),
            "dropped_bad_pts": dropped,
            "writer_seed": seed,
            "stream_encoder_calls": calls,
            "native_parity": parity,
        }
        temp = path.with_suffix(".partial")
        torch.save(entry, temp)
        temp.replace(path)
        write_json(
            receipt,
            {
                "input_id": identity["id"],
                "id": row["id"],
                "sha256": sha256(path),
                "steps": len(times),
                "retained": len(keep),
                "native_parity": parity,
            },
        )
        print(
            json.dumps({"cached": row["id"], "stream_steps": len(times), "retained": len(keep)}), flush=True
        )
    write_json(
        args.out / f"{'smoke-' if args.smoke else ''}shard-{args.shard}.json",
        {"completed": True, "input_id": identity["id"]},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "out", "model"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--shard", type=int, choices=(0, 1), required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
