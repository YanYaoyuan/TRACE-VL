"""Bookkeeping and eviction for the full-attention KV cache (the only part of the hybrid cache that grows).

Every token the LM consumed has metadata (kind, time, group, protected). Policies:
- full:   keep everything (short videos, upper bound)
- window: sink + dialogue + last `window_seconds` (StreamingVLM-style)
- fifo:   window + most recent older groups up to `budget_tokens`
- drift:  window + older groups with the largest Gated DeltaNet state drift (DeltaS-style)
- tkm:    window + groups by the learned retention score (v3 transition KV memory)
- mass:   window + groups by surprise mass (non-learned control for tkm at the same budget)
Older groups (transition slots and whole steps alike, so the pooled B4 arm is scored too) are tracked by
RetentionBook and rescored with their current age before every eviction.
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

    def keep_mask(self, now: float, policy: str | None = None) -> torch.Tensor | None:
        """Tokens to keep. `policy` overrides cfg.policy (counterfactual audits on the same memory state).

        fill "groups": admit whole older groups in rank order while they fit (a group that does not fit is
        skipped), so different rankings can leave different unused capacity.
        fill "exact" (pilot budget repair): admit whole groups while they fit, then the newest tokens of the next
        ranked group up to the budget, so every policy keeps exactly min(budget, older tokens) older tokens."""
        cfg = self.cfg
        policy = policy or cfg.policy
        if policy == "full":
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
        chosen, partial = set(), {}
        if policy != "window" and sizes:
            if policy == "fifo":
                ranked = sorted(sizes, key=lambda g: newest[g], reverse=True)
            else:  # drift / tkm / mass: by score, ties broken by recency
                ranked = sorted(sizes, key=lambda g: (self.scores.get(g, float("-inf")), newest[g]), reverse=True)
            used = 0
            for g in ranked:
                if used + sizes[g] <= cfg.budget_tokens:
                    chosen.add(g)
                    used += sizes[g]
                elif cfg.fill == "exact":
                    if cfg.budget_tokens - used > 0:
                        partial[g] = cfg.budget_tokens - used
                    break
        keep_index = {i for i, (k, g) in enumerate(zip(keep, self.group)) if k or g in chosen}
        for g, take in partial.items():
            members = sorted((i for i, gg in enumerate(self.group) if gg == g and not keep[i]),
                             key=lambda i: (self.time[i], i), reverse=True)  # fmt: skip
            keep_index.update(members[:take])
        mask = torch.tensor([i in keep_index for i in range(len(self.group))], dtype=torch.bool)
        return None if bool(mask.all()) else mask

    def accounting(self, now: float) -> dict:
        """Current KV split: protected (prompt/sink), recent window, older (budgeted), and unused older capacity."""
        recent_from = now - self.cfg.window_seconds
        protected = sum(self.protected)
        window = sum(1 for p, t in zip(self.protected, self.time) if not p and t >= recent_from)
        older = len(self.kind) - protected - window
        groups = {g for p, t, g in zip(self.protected, self.time, self.group) if not p and t < recent_from}
        return {"protected": protected, "window": window, "older": older, "older_groups": len(groups),
                "older_unused": max(0, self.cfg.budget_tokens - older)}

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


@dataclass
class GroupInfo:
    first: float
    last: float
    mass: float
    is_slot: bool
    hits: int = 1


@dataclass
class RetentionBook:
    """Per-group features for retention scoring: KV groups of transition slots ("slot", id) and of whole steps
    ("step", k: gist / pooled / I-step tokens). Features are recomputed at scoring time, so age is current."""

    groups: dict = field(default_factory=dict)

    def observe(self, group: tuple, t: float, mass: float, is_slot: bool):
        info = self.groups.get(group)
        if info is None:
            self.groups[group] = GroupInfo(t, t, float(mass), is_slot)
        else:
            info.last, info.mass, info.hits = t, max(info.mass, float(mass)), info.hits + 1

    def prune(self, alive: set):
        self.groups = {g: i for g, i in self.groups.items() if g in alive}

    def keys(self) -> list[tuple]:
        return list(self.groups)

    def features(self, now: float, keys: list[tuple]) -> torch.Tensor:
        """[len(keys), 6] in RetentionScorer.FEATURES order (age, mass, drift, hits, is_slot, is_step)."""
        import math

        rows = []
        for g in keys:
            i = self.groups[g]
            rows.append([math.log1p(max(now - i.last, 0.0)), i.mass, 0.0, math.log1p(i.hits), float(i.is_slot),
                         float(not i.is_slot)])  # fmt: skip
        return torch.tensor(rows, dtype=torch.float32).reshape(len(keys), 6)

    def mass_scores(self, keys: list[tuple]) -> dict:
        return {g: self.groups[g].mass for g in keys}

    def span(self, group: tuple) -> tuple[float, float]:
        i = self.groups[group]
        return i.first, i.last


def score_retention(book: RetentionBook, memory: KVMemory, scorer, now: float):
    """Refresh memory.scores for every tracked group before eviction. Returns (keys, logits) for tkm (logits keep
    their graph when gradients are enabled; Stage 2 trains on them), else (keys, None)."""
    keys = book.keys()
    if not keys:
        return keys, None
    policy = memory.cfg.policy
    if policy == "tkm" and scorer is not None:
        param = next(scorer.parameters())
        logits = scorer(book.features(now, keys).to(param.device, param.dtype))
        memory.scores.update(zip(keys, logits.detach().float().tolist()))
        return keys, logits
    if policy in {"tkm", "mass"}:  # tkm without a scorer (native mode) degrades to the mass control
        memory.scores.update(book.mass_scores(keys))
    return keys, None

