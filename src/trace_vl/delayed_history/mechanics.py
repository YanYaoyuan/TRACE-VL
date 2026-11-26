"""Causal delayed history: commit departed features; replay recent only on forks.

This changes historical contextualization relative to the original carried R32.
It is a mechanics prototype, not a validated replacement or trained method.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass

import torch

from trace_vl.complementary_memory.state_stream import Block, NativeStream, block_inputs, score
from trace_vl.core_vlm.qa import grid_pool


@dataclass(frozen=True)
class DenseArrival:
    index: int
    end: float
    stamp: tuple[int, ...]
    tokens: torch.Tensor
    grid: tuple[int, int]


def feature_block(step, pool):
    tokens, rows, cols = grid_pool(step.tokens, step.grid, pool)
    return Block(step.index, step.end, tuple(step.stamp), tokens.detach().clone(),
                 rows.detach().clone(), cols.detach().clone(), step.grid)


class HistoryOnly(NativeStream):
    """The unchanged Reservoir append law with no recent owners at this layer.

    NativeStream's constructor requires recent>=1. Init its head normally, then
    use recent=0: every committed input immediately enters historical Reservoir.
    The wrapper below, not this cache, owns the most recent fine features.
    """

    def __init__(self, bb, head, capacity, seed):
        super().__init__(bb, head, "reservoir", recent=1, capacity=capacity, seed=seed)
        self.recent_steps = 0

    def append(self, block):
        if block.index != self.seen:
            raise ValueError("history commits must be consecutive causal indices")
        super().append(block)
        audit = self.audit()
        if audit["recent"] or audit["kv_steps"] != min(self.seen, self.capacity):
            raise RuntimeError("history-only ownership budget diverged")


class DelayedReader:
    """Question-blind lagged prefix plus a bounded recent feature queue.

    Both fine/coarse arms must use this timing. A question consumes recent fine
    exactly once on a fork whose carried state ends before those steps.
    """

    def __init__(self, bb, head, *, capacity, history_grid=(2, 2), recent=8, seed=0):
        if type(recent) is not int or recent < 1 or history_grid not in ((2, 2), (4, 4)):
            raise ValueError("positive recent budget and declared history grid required")
        self.bb = bb
        self.history = HistoryOnly(bb, head, capacity, seed)
        self.history_grid, self.recent_steps = history_grid, recent
        self.recent = deque()
        self.seen, self.last_end = 0, -math.inf

    @torch.no_grad()
    def append(self, step):
        h, w = step.grid
        if (type(step.index) is not int or step.index != self.seen or not math.isfinite(step.end)
                or step.end < 0 or step.end <= self.last_end or h < 4 or w < 4
                or step.tokens.shape != (h * w, self.bb.hidden)
                or not torch.isfinite(step.tokens).all()
                or any(type(t) is not int or t < 0 for t in step.stamp)):
            raise ValueError("finite chronological consecutive dense arrivals on a >=4x4 grid required")
        fine, historical = feature_block(step, (4, 4)), feature_block(step, self.history_grid)
        # The caller cannot mutate an already observed feature queue via its dense tensor.
        self.recent.append((fine, historical))
        self.seen += 1
        self.last_end = step.end
        if len(self.recent) > self.recent_steps:
            _, departed = self.recent.popleft()
            self.history.append(departed)
        self.audit()

    def audit(self):
        recent = [fine.index for fine, _ in self.recent]
        past = self.history.audit()
        committed = max(0, self.seen - self.recent_steps)
        if (past["seen"] != committed or recent != list(range(committed, self.seen))
                or set(recent) & set(past["history"]) or len(self.recent) > self.recent_steps):
            raise RuntimeError("lagged prefix and recent feature ownership diverged")
        feature_bytes = sum(t.numel() * t.element_size() for pair in self.recent
                            for block in pair for t in (block.tokens, block.rows, block.cols))
        return {"seen": self.seen, "committed_steps": committed, "recent_indices": recent,
                "history_indices": past["history"], "history_kv_tokens": past["kv_tokens"],
                "history_kv_bytes": past["kv_bytes"], "history_state_bytes": past["state_bytes"],
                "recent_feature_bytes": feature_bytes,
                "readout_visual_tokens": len(past["history"]) * math.prod(self.history_grid) + len(recent) * 16,
                "scope": "Visual count only; history metadata and transient fork/replay memory are additional."}

    @torch.no_grad()
    def answer(self, tail, options, *, query_time):
        if not self.recent or not math.isfinite(query_time) or self.last_end > query_time:
            raise ValueError("a question needs a nonempty causal observed prefix")
        before = self.history.snapshot()
        logical = (self.audit(), self.history.rng.getstate())
        queue_before = [(b.index, b.tokens.clone(), b.rows.clone(), b.cols.clone())
                        for pair in self.recent for b in pair]
        cache = self.history.fork()
        counter = copy.copy(self.history.counter)
        for fine, _ in self.recent:
            embeds, positions = block_inputs(self.bb, fine, counter)
            self.bb.lm(embeds, positions, cache)
        readout = {**self.audit(), "next_position": counter.pos,
                   "readout_kv_tokens": self.bb.kv_length(cache), "readout_kv_bytes": self.bb.kv_bytes(cache),
                   "readout_state_bytes": self.bb.state_bytes(cache),
                   "readout_recent_indices": [fine.index for fine, _ in self.recent]}
        prediction = score(self.bb, cache, counter.pos, tail, options)
        self.history.verify(before)
        if (self.audit(), self.history.rng.getstate()) != logical:
            raise RuntimeError("question modified live lagged ownership or RNG")
        after = [b for pair in self.recent for b in pair]
        if any(i != b.index or not all(torch.equal(x, y) for x, y in zip(tensors,
               (b.tokens, b.rows, b.cols), strict=True))
               for (i, *tensors), b in zip(queue_before, after, strict=True)):
            raise RuntimeError("question modified the recent feature queue")
        return prediction, readout
