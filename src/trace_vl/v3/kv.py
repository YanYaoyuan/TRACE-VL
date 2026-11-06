"""Bookkeeping and eviction for the full-attention KV cache (the only part of the hybrid cache that grows).

Every token the LM consumed has metadata (kind, time, group, protected). Policies:
- full:   keep everything (short videos, upper bound)
- window: sink + dialogue + last `window_seconds` (StreamingVLM-style)
- fifo:   window + most recent older groups up to `budget_tokens`
- drift:  window + older groups with the largest Gated DeltaNet state drift (DeltaS-style)
- tkm:    window + transition groups by retention score (v3 transition KV memory)
Linear-attention (GDN) layers are never touched: they integrate every token into a fixed-size state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import MemoryConfig

KINDS = {"prompt": 0, "text": 1, "frame": 2, "gist": 3, "before": 4, "after": 5, "delta": 6}


@dataclass
class KVMemory:
    cfg: MemoryConfig
    kind: list[int] = field(default_factory=list)
    time: list[float] = field(default_factory=list)
    group: list[tuple] = field(default_factory=list)
    protected: list[bool] = field(default_factory=list)
    scores: dict = field(default_factory=dict)  # group -> retention score
    evicted_tokens: int = 0

    def append(self, n: int, kind: str | list[str], time: float, group: tuple | list[tuple], protected=False):
        kinds = kind if isinstance(kind, list) else [kind] * n
        groups = group if isinstance(group, list) else [group] * n
        if len(kinds) != n or len(groups) != n:
            raise ValueError("metadata length mismatch")
        self.kind += [KINDS[k] for k in kinds]
        self.time += [time] * n
        self.group += groups
        self.protected += [protected] * n

    def __len__(self):
        return len(self.kind)

    def keep_mask(self, now: float) -> torch.Tensor | None:
        cfg = self.cfg
        if cfg.policy == "full":
            return None
        recent_from = now - cfg.window_seconds
        keep = [p or t >= recent_from for p, t in zip(self.protected, self.time)]
        sizes: dict[tuple, int] = {}
        newest: dict[tuple, float] = {}
        for i, g in enumerate(self.group):
            if keep[i]:
                continue
            sizes[g] = sizes.get(g, 0) + 1
            newest[g] = max(newest.get(g, float("-inf")), self.time[i])
        if cfg.policy == "window" or not sizes:
            chosen = set()
        else:
            if cfg.policy == "fifo":
                ranked = sorted(sizes, key=lambda g: newest[g], reverse=True)
            else:  # drift / tkm: by score, ties broken by recency
                ranked = sorted(sizes, key=lambda g: (self.scores.get(g, float("-inf")), newest[g]), reverse=True)
            chosen, used = set(), 0
            for g in ranked:
                if used + sizes[g] > cfg.budget_tokens:
                    continue
                chosen.add(g)
                used += sizes[g]
        mask = torch.tensor([k or g in chosen for k, g in zip(keep, self.group)], dtype=torch.bool)
        return None if bool(mask.all()) else mask

    def apply(self, mask: torch.Tensor):
        idx = mask.nonzero().flatten().tolist()
        self.evicted_tokens += len(self.kind) - len(idx)
        self.kind = [self.kind[i] for i in idx]
        self.time = [self.time[i] for i in idx]
        self.group = [self.group[i] for i in idx]
        self.protected = [self.protected[i] for i in idx]
        alive = set(self.group)
        self.scores = {g: s for g, s in self.scores.items() if g in alive}

    def counts(self) -> dict:
        names = {v: k for k, v in KINDS.items()}
        out: dict[str, int] = {}
        for k in self.kind:
            out[names[k]] = out.get(names[k], 0) + 1
        return out
