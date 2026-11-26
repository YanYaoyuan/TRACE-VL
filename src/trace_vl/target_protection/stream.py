"""Original Reservoir until admission; privileged protected ownership afterwards."""

from __future__ import annotations

import torch

from trace_vl.complementary_memory.state_stream import PROMPT, NativeStream, block_inputs


class TargetProtectedStream(NativeStream):
    """Disjoint newest steps, protected departed steps and a shrinking uniform Reservoir.

    Protection is announced causally from the current snapshot, without gold/question text.
    It is privileged timing, not a deployable question-blind writer.
    """

    def __init__(self, bb, head, *, recent=8, capacity=32, target_size=8, seed=0):
        if not 1 <= target_size <= min(recent, capacity):
            raise ValueError("target size must fit recent and historical budgets")
        self.target_size = target_size
        self.target_steps = None
        self.protected = []
        super().__init__(bb, head, "reservoir", recent=recent, capacity=capacity, seed=seed)

    def protect(self, indices):
        indices = tuple(indices)
        if (self.target_steps is not None or len(indices) != self.target_size
                or len(set(indices)) != len(indices) or list(indices) != sorted(indices)
                or any(type(i) is not int for i in indices)
                or not set(indices).issubset(self.recent)):
            raise ValueError("one unique causal target snapshot in recent is required")
        self.target_steps = frozenset(indices)

    def _departed(self, old):
        """Return the evicted step, or None. Protected steps never count as R candidates."""
        if old in self.target_steps:
            self.protected.append(old)
            limit = self.capacity - len(self.protected)
            if len(self.history) > limit:
                return self.history.pop(self.rng.randrange(len(self.history)))
            return None
        self.old_seen += 1
        limit = self.capacity - len(self.protected)
        if len(self.history) < limit:
            self.history.append(old)
            return None
        slot = self.rng.randrange(self.old_seen)
        if slot < limit:
            dropped, self.history[slot] = self.history[slot], old
            return dropped
        return old

    def append(self, block):
        if block.index != self.seen:
            raise ValueError("stream steps must be consecutive unique indices")
        departing = self.recent[0] if len(self.recent) == self.recent_steps else None
        if not self.protected and (self.target_steps is None or departing not in self.target_steps):
            # This exact unchanged implementation is essential for the 0 s reproduction gate.
            return super().append(block)
        embeds, positions = block_inputs(self.bb, block, self.counter)
        self.bb.lm(embeds, positions, self.cache)
        self.owners += [block.index] * embeds.shape[1]
        self.recent.append(block.index)
        self.seen += 1
        dropped = self._departed(self.recent.popleft())
        if dropped is not None:
            self.bb.evict(self.cache, torch.tensor([o != dropped for o in self.owners]))
            self.owners = [o for o in self.owners if o != dropped]
        audit = self.audit()
        if (self.bb.kv_length(self.cache) != len(self.owners)
                or audit["kv_steps"] != min(self.seen, self.recent_steps + self.capacity)
                or len(self.history) > audit["reservoir_capacity"]):
            raise RuntimeError("protected ownership or KV budget diverged")

    def audit(self):
        value = super().audit()
        recent, protected, history = set(self.recent), set(self.protected), set(self.history)
        if (recent & protected or recent & history or protected & history
                or len(protected) != len(self.protected) or len(history) != len(self.history)
                or recent | protected | history != set(self.owners) - {PROMPT}):
            raise RuntimeError("recent/protected/Reservoir ownership is not disjoint and complete")
        value.update(policy="target_protected32", protected=sorted(protected),
                     target_steps=None if self.target_steps is None else sorted(self.target_steps),
                     reservoir_capacity=self.capacity - len(protected), reservoir_seen=self.old_seen)
        return value
