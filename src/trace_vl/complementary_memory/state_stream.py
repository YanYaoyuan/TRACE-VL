"""Continuous native Qwen3.5 streams with bounded KV and isolated question forks (Test A)."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch

from trace_vl.v3.backbone import branch_cache
from trace_vl.v3.positions import PositionCounter

PROMPT = -1  # owner of protected prompt KV entries


@dataclass(frozen=True)
class Block:
    """One stored step: <stamp><vision_start> pooled tokens <vision_end>."""

    index: int
    end: float
    stamp: tuple[int, ...]
    tokens: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    grid: tuple[int, int]


def segment_layout(ids, video_token, vision_start, vision_end, first_stamp):
    """Split one native single-video prompt into (head, tail) around its step blocks."""
    ids = list(ids)
    video = [i for i, x in enumerate(ids) if x == video_token]
    if not video:
        raise ValueError("native prompt has no video tokens")
    first, last = video[0], video[-1]
    if first < 1 or ids[first - 1] != vision_start or last + 1 >= len(ids) or ids[last + 1] != vision_end:
        raise ValueError("unexpected native vision delimiters")
    head, stamp = ids[:first - 1], list(first_stamp)
    if stamp:
        if head[-len(stamp):] != stamp:
            raise ValueError("native timestamp tokens do not match the stream protocol")
        head = head[:-len(stamp)]
    tail = ids[last + 2:]
    if not tail:
        raise ValueError("native prompt has no question tail")
    return head, tail


def block_inputs(bb, block, counter):
    lead = [*block.stamp, bb.vision_start]
    embeds = torch.cat([bb.embed_ids(lead), block.tokens[None].to(bb.device, bb.dtype),
                        bb.embed_ids([bb.vision_end])], 1)
    positions = torch.cat([counter.text(len(lead), bb.device),
                           counter.points(block.rows, block.cols, *block.grid),
                           counter.text(1, bb.device)], 1)
    return embeds, positions


class NativeStream:
    """One question-blind live cache. Recurrent/conv state is never evicted; KV follows `policy`.

    window: prompt + `recent` steps. reservoir: plus at most `capacity` uniformly sampled older steps.
    full: unbounded KV reference.
    """

    def __init__(self, bb, head, policy, *, recent=8, capacity=32, seed=0):
        if policy not in ("window", "reservoir", "full") or recent < 1 or capacity < 1 or not head:
            raise ValueError("invalid stream policy, budget or prompt")
        self.bb, self.policy, self.recent_steps, self.capacity = bb, policy, recent, capacity
        self.rng = random.Random(seed)
        self.cache, self.counter = bb.new_cache(), PositionCounter()
        self.recent, self.history = deque(), []
        self.old_seen = self.seen = 0
        self.owners = [PROMPT] * len(head)
        bb.lm(bb.embed_ids(head), self.counter.text(len(head), bb.device), self.cache)

    def append(self, block):
        embeds, positions = block_inputs(self.bb, block, self.counter)
        self.bb.lm(embeds, positions, self.cache)
        self.owners += [block.index] * embeds.shape[1]
        self.recent.append(block.index)
        self.seen += 1
        if len(self.recent) > self.recent_steps:
            old = self.recent.popleft()
            self.old_seen += 1
            dropped = None
            if self.policy == "window":
                dropped = old
            elif self.policy == "full" or len(self.history) < self.capacity:
                self.history.append(old)
            else:
                slot = self.rng.randrange(self.old_seen)
                dropped = old
                if slot < self.capacity:
                    dropped, self.history[slot] = self.history[slot], old
            if dropped is not None:
                keep = torch.tensor([o != dropped for o in self.owners])
                self.bb.evict(self.cache, keep)
                self.owners = [o for o in self.owners if o != dropped]
        if self.bb.kv_length(self.cache) != len(self.owners):
            raise RuntimeError("KV ownership diverged from the native cache")
        if self.policy != "full" and len(set(self.owners) - {PROMPT}) > self.recent_steps + (
                self.capacity if self.policy == "reservoir" else 0):
            raise RuntimeError("KV step budget exceeded")

    def fork(self, *, history=True, state_from=None):
        """Question fork. Optionally drop historical KV and/or install another stream's recurrent state."""
        cache = branch_cache(self.cache)
        if not history:
            keep = {PROMPT, *self.recent}
            self.bb.evict(cache, torch.tensor([o in keep for o in self.owners]))
        if state_from is not None:
            swap_state(self.bb, cache, state_from.cache)
        return cache

    def snapshot(self):
        """References/clones sufficient to prove that a question did not change the live stream."""
        kv = [(self.cache.layers[i].keys, self.cache.layers[i].values) for i in self.bb.full_layers]
        return kv, [t.clone() for t in state_tensors(self.bb, self.cache)], self.counter.pos, list(self.owners)

    def verify(self, snapshot):
        kv, state, position, owners = snapshot
        now = [(self.cache.layers[i].keys, self.cache.layers[i].values) for i in self.bb.full_layers]
        if (any(a is not c or b is not d for (a, b), (c, d) in zip(kv, now, strict=True))
                or any(not torch.equal(a, b) for a, b in
                       zip(state, state_tensors(self.bb, self.cache), strict=True))
                or position != self.counter.pos or owners != self.owners):
            raise RuntimeError("a question changed the live stream")

    def audit(self):
        return {"policy": self.policy, "seen": self.seen, "kv_tokens": len(self.owners),
                "kv_steps": len(set(self.owners) - {PROMPT}), "history": sorted(self.history),
                "recent": list(self.recent), "kv_bytes": self.bb.kv_bytes(self.cache),
                "state_bytes": self.bb.state_bytes(self.cache), "next_position": self.counter.pos,
                "state_dtypes": sorted({str(t.dtype) for t in state_tensors(self.bb, self.cache)})}


def state_tensors(bb, cache):
    return [value for i in bb.linear_layers for name in ("recurrent_states", "conv_states")
            for _, value in sorted(getattr(cache.layers[i], name).items()) if value is not None]


def swap_state(bb, target, donor):
    """Replace recurrent+conv state of a fork; full-attention KV is untouched."""
    for i in bb.linear_layers:
        a, b = target.layers[i], donor.layers[i]
        for name in ("recurrent_states", "conv_states"):
            old, new = getattr(a, name), getattr(b, name)
            if old.keys() != new.keys() or any(old[k].shape != new[k].shape for k in old):
                raise ValueError("incompatible recurrent state donor")
            setattr(a, name, {k: v.clone() for k, v in new.items()})


def replay(bb, head, blocks, *, positions_start=None):
    """Fresh cache over `head` and selected blocks.

    With positions_start the blocks keep the live stream's coordinates, for use as a state donor.
    Otherwise coordinates are contiguous, as in an ordinary recent-window deployment.
    """
    stream = NativeStream(bb, head, "full", recent=max(1, len(blocks)))
    if positions_start is not None:
        if positions_start < stream.counter.pos:
            raise ValueError("replay positions overlap the prompt")
        stream.counter.pos = positions_start
    for block in blocks:
        stream.append(block)
    return stream


@torch.no_grad()
def score(bb, cache, position, tail, options):
    """Answer on a disposable fork; gold is not an argument."""
    letters = [bb.ids(chr(65 + i)) for i in range(options)]
    if any(len(x) != 1 for x in letters):
        raise ValueError("single-token answer options required")
    hidden = bb.lm(bb.embed_ids(tail), PositionCounter(position).text(len(tail), bb.device), cache)[0, -1]
    values = bb.lm_head.weight[[x[0] for x in letters]].float() @ hidden.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite answer logits")
    return {"choice": int(values.argmax()), "option_logits_fp32": values.tolist()}
