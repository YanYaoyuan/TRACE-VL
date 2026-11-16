"""Evaluation-only interventions; fixed timestamp slots, no gold-dependent choices."""
from __future__ import annotations

import hashlib
from dataclasses import replace


def block_order(count, video_sha256, seed=17):
    if count < 2:
        raise ValueError('block shuffle requires at least two temporal blocks')
    order = sorted(range(count), key=lambda i: hashlib.sha256(
        f'core-block-shuffle-v1:{seed}:{video_sha256}:{i}'.encode()).digest())
    if order == list(range(count)):
        order = order[1:] + order[:1]
    return order


def shuffle_blocks(steps, video_sha256, seed=17):
    """Permute encoded two-frame content; preserve spatial grids and time slots.

    This is NOT raw-frame shuffling: encoder time information may remain inside
    each block. All source blocks were observed before the question. Timestamp
    slots deliberately remain chronological, creating a diagnostic mismatch.
    """
    order = block_order(len(steps), video_sha256, seed)
    if len({s.grid for s in steps}) != 1:
        raise ValueError('shuffle requires identical spatial grids')
    return [replace(slot, tokens=steps[j].tokens) for slot, j in zip(steps, order)], order


def disable_temporal(net):
    if not hasattr(net, 'temporal'):
        raise ValueError('temporal-off requires a trained temporal module')
    del net.temporal
