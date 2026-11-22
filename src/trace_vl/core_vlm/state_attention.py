"""Causal storage and decoder-side attention over two observations of one track."""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

READERS = ("cross", "difference", "antisymmetric")


def region_feature(tokens, grid, box):
    """Pool token centres in a normalized xyxy region on this observation's grid."""
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("expected a nonempty normalized xyxy box")
    h, w = grid
    if tokens.ndim != 2 or len(tokens) != h * w or min(h, w) <= 0:
        raise ValueError("tokens do not match the spatial grid")
    x = (torch.arange(w, device=tokens.device) + .5) / w
    y = (torch.arange(h, device=tokens.device) + .5) / h
    inside = (x >= x1) & (x <= x2)
    mask = ((y >= y1) & (y <= y2))[:, None] & inside[None, :]
    if not mask.any():
        col = min(w - 1, int((x1 + x2) * w / 2))
        row = min(h - 1, int((y1 + y2) * h / 2))
        mask[row, col] = True
    return tokens[mask.flatten()].float().mean(0).to(tokens.dtype)


@dataclass(frozen=True)
class StatePair:
    track: str
    t_before: float
    t_after: float
    before: torch.Tensor
    after: torch.Tensor


class StatePairBuffer:
    """One last observation per track, plus a bounded FIFO of completed pairs."""

    def __init__(self, capacity, *, max_tracks=32):
        if min(capacity, max_tracks) < 1:
            raise ValueError("positive pair and track limits required")
        self.pairs = deque(maxlen=capacity)
        self.tracks = OrderedDict()
        self.max_tracks = max_tracks
        self.t = -math.inf

    def observe(self, time, tracks):
        if not math.isfinite(time) or time < 0 or time <= self.t:
            raise ValueError("observations must have increasing nonnegative times")
        self.t = time
        for track, value in sorted(tracks.items()):
            value = value.detach().clone()
            old = self.tracks.pop(track, None)
            if old is not None:
                t_before, before = old
                self.pairs.append(StatePair(track, t_before, time, before, value))
            self.tracks[track] = (time, value)
            if len(self.tracks) > self.max_tracks:
                self.tracks.popitem(last=False)

    def read(self, query_time):
        if not math.isfinite(query_time) or query_time < self.t:
            raise ValueError("read from the causal prefix at the query time")
        return list(self.pairs)


class OnlineBoxMatcher:
    """Greedy one-to-one appearance matching from causal region proposals."""

    def __init__(self, *, minimum_cosine=.65, max_gap=2, max_tracks=32):
        self.minimum_cosine = minimum_cosine
        self.max_gap, self.max_tracks = max_gap, max_tracks
        self.active = OrderedDict()
        self.next_track = 0

    def update(self, step, features):
        stale = [key for key, (_, last) in self.active.items() if step - last > self.max_gap]
        for key in stale:
            del self.active[key]
        candidates = []
        if self.active and features:
            keys = list(self.active)
            previous = torch.stack([self.active[key][0] for key in keys])
            similarity = F.normalize(previous.float()) @ F.normalize(torch.stack(features).float()).T
            for i, scores in enumerate(similarity.tolist()):
                candidates.extend((score, keys[i], j) for j, score in enumerate(scores)
                                  if score >= self.minimum_cosine)
        assignment, used = {}, set()
        for _, key, index in sorted(candidates, reverse=True):
            if key not in used and index not in assignment:
                assignment[index] = key
                used.add(key)
        tracks = {}
        for i, feature in enumerate(features):
            key = assignment.get(i)
            if key is None:
                key = f"track-{self.next_track:08d}"
                self.next_track += 1
            self.active.pop(key, None)
            self.active[key] = (feature.detach(), step)
            tracks[key] = feature
        while len(self.active) > self.max_tracks:
            self.active.popitem(last=False)
        return tracks


class StateAttention(nn.Module):
    """Matched-size readers sharing a gated residual and one slot per pair."""

    def __init__(self, hidden, width, reader):
        super().__init__()
        if reader not in READERS or width < 2:
            raise ValueError("invalid reader or projection width")
        self.reader = reader
        self.width = width
        self.state = nn.Linear(hidden, width, bias=False)
        self.first = nn.Linear(width, width, bias=False)
        self.second = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(hidden, width, bias=False)
        self.output = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def key_value(self, pairs):
        state = self.state(F.layer_norm(pairs.float(), (pairs.shape[-1],)))
        a, b = state.unbind(-2)
        ua, ub = self.first(a), self.first(b)
        wa, wb = self.second(a), self.second(b)
        if self.reader == "cross":
            key = ua + wb
            value = key
        else:
            left, right = ua + wa, ub + wb
            key = (left + right) * .5 + (right - left).abs()
            value = right - left if self.reader == "difference" else ua * wb - ub * wa
        return self.key(key), F.layer_norm(value, (self.width,))

    def read(self, query, pairs, valid):
        keys, values = self.key_value(pairs)
        q = self.query(F.layer_norm(query.float(), (query.shape[-1],)))
        scores = torch.einsum("bd,bkd->bk", q, keys) / math.sqrt(self.width)
        scores = scores.masked_fill(~valid, -torch.inf)
        # The fixed zero-value slot allows the reader to abstain from all pairs.
        weights = F.pad(scores, (0, 1)).softmax(-1)[:, :-1]
        values = values.masked_fill(~valid[..., None], 0)
        return self.output(torch.einsum("bk,bkd->bd", weights, values)), weights

    def forward(self, query, pairs, valid):
        value, _ = self.read(query, pairs, valid)
        return (value * self.gate.tanh()).to(query.dtype)


def answer_logits(backbone, embeds, positions, valid_tokens, final_positions, reader=None,
                  pairs=None, pair_mask=None, *, layer=None):
    """Inject only at the last prompt token, and scope the hook to this forward."""
    batch = torch.arange(embeds.shape[0], device=embeds.device)
    handle = None
    if reader is not None:
        if layer is None:
            raise ValueError("decoder layer required for state attention")

        def inject(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            residual = reader(hidden[batch, final_positions], pairs, pair_mask)
            updated = hidden.clone()
            updated[batch, final_positions] += residual
            return (updated, *output[1:]) if isinstance(output, tuple) else updated

        handle = backbone.language_model.layers[layer].register_forward_hook(inject)
    try:
        hidden = backbone.lm(embeds, positions, cache=None, use_cache=False,
                             attention_mask=valid_tokens)
        return backbone.logits(hidden[batch, final_positions][:, None])[:, 0].float()
    finally:
        if handle is not None:
            handle.remove()
