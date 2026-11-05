"""Deterministic synthetic moving-object clips. These are software fixtures, not a paper benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import validate_record, write_jsonl


def make_synthetic(output, train=12, val=4, test=4, seed=7, video=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for split, count in (("train", train), ("val", val), ("test", test)):
        for i in range(count):
            positive = i % 2 == 0
            frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
            base = rng.integers(35, 80, size=3, dtype=np.uint8)
            frames[:] = base
            for t in range(8):
                frames[t, 45:, :] = base + 25
                # Person-like gray bar moves away. Orange object persists only for positives.
                x = min(54, 8 + t * 6)
                frames[t, 18:45, x : x + 5] = (145, 160, 175)
                if positive and t >= 2:
                    frames[t, 40:49, 29:39] = (230, 140, 35)
                elif not positive:
                    frames[t, 32:41, min(x + 6, 54) : min(x + 13, 63)] = (230, 140, 35)
            times = np.arange(8, dtype=np.float64) / 2
            stem = f"{split}_{i:04d}"
            np.savez_compressed(output / f"{stem}.npz", frames=frames, timestamps=times)
            if video:
                import cv2

                writer = cv2.VideoWriter(
                    str(output / f"{stem}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 2, (64, 64)
                )
                if not writer.isOpened():
                    raise RuntimeError("OpenCV MJPG writer unavailable")
                for frame in frames:
                    writer.write(frame[:, :, ::-1])
                writer.release()
            event = {
                "id": f"{stem}:object_left",
                "category": "object_left",
                "start": 1.0,
                "end": 3.5,
                "known_at": 1.5,
                "deadline": 2.5,
            }
            row = {
                "schema_version": 1,
                "id": stem,
                "split": split,
                "source": "synthetic_fixture",
                "source_group": f"synthetic-camera-{split}-{i}",
                "video": f"{stem}.npz",
                "protocol": "streaming",
                "observation": {"start": 0.0, "end": 3.5},
                "monitor_category": "object_left",
                "rule": "Alert when an orange object is left behind after the person moves away.",
                "question": "Was an orange object left behind? Answer yes or no.",
                "answer": "yes" if positive else "no",
                "answer_type": "text",
                "events": [event] if positive else [],
                "category": "object_presence",
                "provenance": "Procedural smoke-test fixture; not real surveillance or research evidence.",
            }
            rows.append(validate_record(row))
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, rows)
    for split in ("train", "val", "test"):
        write_jsonl(output / (split + ".jsonl"), [r for r in rows if r["split"] == split])
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "synthetic_fixture",
                "seed": seed,
                "counts": {"train": train, "val": val, "test": test},
                "suitable_for": "software validation only",
                "generator": "trace_vl.data.synthetic",
            },
            indent=2,
        )
        + "\n"
    )
    return manifest
