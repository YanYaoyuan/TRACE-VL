"""Independent mixed-grid reference selected from HF's full native layout.

Used with synthetic layouts for CPU mechanics tests; not real-processor parity.
"""
from __future__ import annotations

import torch

from trace_vl.core_vlm.native import NativeVideoQALoss


@torch.no_grad()
def mixed_native_prefix(bb, steps, record, grids):
    full, native_positions, counter, _, _ = NativeVideoQALoss(bb, mode="full").prefix(steps, record)
    ids = record["native_layout"].inputs["input_ids"][0]
    indices = (ids == bb.video_token).nonzero().flatten()
    chunks = indices.split([s.grid[0] * s.grid[1] for s in steps])
    if len(grids) != len(steps):
        raise ValueError("one declared grid per step required")
    parts, positions, cursor = [], [], 0
    for step, locations, target in zip(steps, chunks, grids, strict=True):
        h, w = step.grid
        rows, cols = target
        if not 1 <= rows <= h or not 1 <= cols <= w:
            raise ValueError("reference grid must fit the dense native span")
        lo, hi = int(locations[0]), int(locations[-1]) + 1
        if hi - lo != len(locations):
            raise ValueError("dense native visual span is not contiguous")
        parts.append(full[:, cursor:lo])
        positions.append(native_positions[:, cursor:lo])
        for row in range(rows):
            a, b = row * h // rows, (row + 1) * h // rows
            for col in range(cols):
                c, d = col * w // cols, (col + 1) * w // cols
                members = [int(locations[r * w + k]) for r in range(a, b) for k in range(c, d)]
                # Reduce canonical dense embeddings; do not call the writer's
                # pooling or PositionCounter/block_inputs implementation.
                parts.append(full[:, members].mean(1, keepdim=True))
                centre = int(locations[((a + b - 1) // 2) * w + (c + d - 1) // 2])
                positions.append(native_positions[:, centre:centre + 1])
        cursor = hi
    parts.append(full[:, cursor:])
    positions.append(native_positions[:, cursor:])
    return torch.cat(parts, 1), torch.cat(positions, 1), counter
