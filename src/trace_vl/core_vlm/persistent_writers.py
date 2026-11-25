"""Query-blind bounded history writers for future diagnostic controls, not a promoted method."""
from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import deque
from dataclasses import replace

import torch

from .longstream import StoredStep


class PersistentWriter:
    """Own original observations, four recent steps and at most capacity old steps.

    time_coverage removes the densest interior temporal sample. spatial_change
    first removes the least necessary interior point of the normalized spatial
    feature path, with the same time criterion breaking ties. Both preserve the
    first and latest historical observations. Neither averages/re-encodes values.
    All descriptors are transient and recomputed from retained values; no growing
    archive, question, answer, target timestamp or hidden descriptor cache exists.
    """

    def __init__(self, policy: str, capacity: int = 32):
        if policy not in ('time_coverage', 'spatial_change'):
            raise ValueError('unknown persistent writer policy')
        if type(capacity) is not int or capacity < 4:
            raise ValueError('at least four historical slots required')
        self.policy = policy
        self.capacity = capacity
        self.history: list[StoredStep] = []
        self.recent: deque[StoredStep] = deque(maxlen=4)
        self.seen = 0
        self.old_seen = 0
        self.lineage = hashlib.sha256()

    def _validate(self, step):
        if (step.tokens.ndim != 2 or step.tokens.shape[0] != 16 or step.tokens.shape[1] < 1
                or not step.tokens.is_floating_point() or not bool(torch.isfinite(step.tokens).all())
                or len(step.times) != 2 or not all(math.isfinite(t) for t in step.times)
                or not 0 <= step.times[0] < step.times[1]
                or len(step.indices) != 2 or any(type(i) is not int for i in step.indices)
                or not 0 <= step.indices[0] < step.indices[1]
                or len(step.grid) != 2 or any(type(n) is not int or n < 1 for n in step.grid)
                or re.fullmatch('[0-9a-f]{64}', step.rgb_sha256) is None):
            raise ValueError('invalid stored visual observation')
        if self.recent:
            last = self.recent[-1]
            if step.times[0] <= last.end or step.indices[0] <= last.indices[-1]:
                raise ValueError('strictly increasing nonoverlapping source observations required')
            if (step.tokens.shape != last.tokens.shape or step.tokens.dtype != last.tokens.dtype
                    or step.tokens.device != last.tokens.device):
                raise ValueError('feature shape, dtype and device must remain fixed')

    def _admit(self, step):
        candidates = sorted([*self.history, step], key=lambda s: s.end)
        if len(candidates) <= self.capacity:
            return candidates
        spans = [candidates[i+1].end-candidates[i-1].end for i in range(1, len(candidates)-1)]
        if self.policy == 'time_coverage':
            costs = [0.] * len(spans)
        else:
            # Fixed-size CPU FP32 workspace, not persisted/readable memory. Keep
            # every spatial position: mean pooling would hide swapped objects.
            values = torch.stack([s.tokens.detach().to(device='cpu', dtype=torch.float32) for s in candidates])
            values = torch.nn.functional.normalize(values, dim=-1).flatten(1)
            adjacent = torch.linalg.vector_norm(values[1:]-values[:-1], dim=1)
            shortcut = torch.linalg.vector_norm(values[2:]-values[:-2], dim=1)
            excess = (adjacent[:-1]+adjacent[1:]-shortcut).clamp_min(0)
            costs = [round(float(v), 6) for v in excess]
        # Oldest interior timestamp is the final deterministic tie breaker.
        remove = min(range(1, len(candidates)-1), key=lambda i: (costs[i-1], spans[i-1], candidates[i].end))
        return [s for i, s in enumerate(candidates) if i != remove]

    @torch.no_grad()
    def append(self, step: StoredStep):
        self._validate(step)
        proposed = self._admit(self.recent[0]) if len(self.recent) == 4 else self.history
        # Own the value: a caller reusing its current-frame buffer cannot erase
        # the historical observation after append returns.
        owned = replace(step, tokens=step.tokens.detach().clone())
        if len(self.recent) == 4:
            self.old_seen += 1
        self.history = proposed
        self.recent.append(owned)
        self.seen += 1
        self.lineage.update(bytes.fromhex(step.rgb_sha256))
        self.lineage.update(struct.pack('<qq', *step.indices))
        if len(self.history) > self.capacity or len(self.recent) > 4:
            raise RuntimeError('persistent storage budget exceeded')

    def read(self, target: float, count: int = 8) -> list[StoredStep]:
        if not math.isfinite(target) or target < 0 or type(count) is not int or not 1 <= count <= 8:
            raise ValueError('finite causal target and at most eight read steps required')
        # Same reader as delayed FIFO/reservoir: no query-conditioned routing.
        available = [s for s in (*self.history, *self.recent) if s.end <= target]
        return sorted(available, key=lambda s: s.end)[-count:]

    def audit(self):
        slots = [*self.history, *self.recent]
        if len({s.indices for s in slots}) != len(slots):
            raise RuntimeError('history and recent observations must be disjoint')
        payload = sum(s.tokens.numel()*s.tokens.element_size() for s in slots)
        return {'policy': self.policy, 'seen': self.seen, 'old_seen': self.old_seen,
                'capacity': self.capacity, 'history_steps': len(self.history), 'recent_steps': len(self.recent),
                'lineage': self.lineage.hexdigest(), 'feature_payload_bytes': payload,
                'feature_and_fixed_slot_metadata_bytes': payload+96*len(slots),
                'persistent_auxiliary_feature_bytes': 0,
                'retained': {key: [{'indices': list(s.indices), 'times': list(s.times), 'rgb_sha256': s.rgb_sha256}
                                   for s in values] for key, values in (('history', self.history), ('recent', self.recent))},
                'accounting': 'Actual tensor payload plus96 logical bytes/slot, not RSS. Global counters/hash and Python overhead separate. Incoming current value and capacity-bounded FP32 descriptor workspace are transient; no growing archive or descriptor cache.'}
