"""Bounded visual memory and a trainable interval-routed Qwen residual."""
from __future__ import annotations

from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from trace_vl.core_vlm.qa import VisualStep, grid_pool


class VisualHistory:
    """Incremental FIFO storage. Each old block retains a small spatial grid per step."""

    def __init__(self, recent, block, capacity):
        self.recent = deque()
        self._pending = []
        self.slots = deque(maxlen=capacity)
        self.recent_limit = recent
        self.block_size = block

    def append(self, step: VisualStep):
        if self.recent and step.end <= self.recent[-1].end:
            raise ValueError("history requires strictly increasing observations")
        self.recent.append(step)
        if len(self.recent) <= self.recent_limit:
            return
        old = self.recent.popleft()
        h, w = old.grid
        pooled, _, _ = grid_pool(old.tokens, old.grid, (min(2, h), min(2, w)))
        pooled = pooled.reshape(min(2, h), min(2, w), -1)
        if h == 1:
            pooled = pooled.expand(2, -1, -1)
        if w == 1:
            pooled = pooled.expand(-1, 2, -1)
        pooled = pooled.reshape(4, -1)
        self._pending.append(VisualStep(pooled, (2, 2), old.end, old.frame_times))
        if len(self._pending) == self.block_size:
            self.slots.append(tuple(self._pending))
            self._pending.clear()

    def tensors(self):
        tokens = [torch.cat([s.tokens for s in steps]) for steps in self.slots]
        times = [[steps[0].frame_times[0], steps[-1].end] for steps in self.slots]
        if not tokens:
            d = self.recent[-1].tokens.shape[-1]
            return self.recent[-1].tokens.new_empty((0, self.block_size * 4, d)), torch.empty((0, 2))
        return torch.stack(tokens), torch.tensor(times, dtype=torch.float32)


def evidence_indices(slots, span, query_id):
    """Return the most-covered labelled block and a same-video disjoint control."""
    matches, controls = [], []
    for index, slot in enumerate(slots):
        frames = [t for step in slot for t in step.frame_times]
        count = sum(span[0] <= t <= span[1] for t in frames)
        if count:
            matches.append((count, -index))
        elif not (frames[0] < span[1] and frames[-1] > span[0]):
            controls.append(index)
    if not matches or not controls:
        return None
    target = -max(matches)[1]
    # The control uses the nearest fully disjoint block. This keeps the delay
    # comparable while guaranteeing zero overlap in the actual sampled frames.
    control = min(controls, key=lambda k: (abs(k - target), (k + int(query_id[-1], 16)) % 2))
    return target, control


class IntervalReader(nn.Module):
    """Same visual read budget for cosine and trained retrieval arms.

    Queries come from the frozen decoder's recent-question hidden state. A
    low-rank gated cross-attention residual goes directly to the native Qwen
    vocabulary projection. Option labels never enter interval selection.
    """

    def __init__(self, hidden, width, mode):
        super().__init__()
        if mode not in {"cosine", "learned"}:
            raise ValueError("unknown retrieval mode")
        self.mode = mode
        self.read_query = nn.Linear(hidden, width, bias=False)
        self.read_key = nn.Linear(hidden, width, bias=False)
        self.read_value = nn.Linear(hidden, width, bias=False)
        self.output = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        if mode == "learned":
            self.query = nn.Linear(hidden, width, bias=False)
            self.key = nn.Linear(hidden, width, bias=False)
            self.time = nn.Linear(2, width, bias=False)

    def forward(self, query, memory, relative_times, *, heads):
        if memory.shape[0] == 0:
            return heads @ query, query.new_empty(0), None
        keys = memory.mean(1)
        if self.mode == "learned":
            learned_keys = self.key(keys) + self.time(relative_times)
            scores = F.normalize(learned_keys, dim=-1) @ F.normalize(self.query(query), dim=-1)
        else:
            scores = F.normalize(keys, dim=-1) @ F.normalize(query, dim=-1)
        scores = scores * 12
        index = scores.detach().argmax()
        selected = memory[index]
        if self.mode == "learned" and self.training:
            soft = torch.einsum("s,std->td", scores.softmax(0), memory)
            selected = selected + (soft - soft.detach())
        k = self.read_key(selected)
        attn = (k @ self.read_query(query) / k.shape[-1]**.5).softmax(0)
        residual = self.output(attn @ self.read_value(selected))
        return heads @ (query + self.gate.tanh() * residual), scores, index
