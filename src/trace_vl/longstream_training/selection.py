"""Question-blind reservoir writing, fixed-budget question-conditioned reading."""

from __future__ import annotations

import math
import random

import torch
from torch import nn
from torch.nn import functional as F


def retained_indices(times, *, seed, history=32, recent=4):
    if not times or any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("finite nonempty stream times required")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("strictly causal stream order required")
    if history != 32 or recent != 4:
        raise ValueError("initial matched protocol fixes32historical and4recent slots")
    rng = random.Random(seed)
    bank, tail, seen = [], [], 0
    for i in range(len(times)):
        tail.append(i)
        if len(tail) <= recent:
            continue
        old = tail.pop(0)
        seen += 1
        if len(bank) < history:
            bank.append(old)
        else:
            j = rng.randrange(seen)
            if j < history:
                bank[j] = old
    return sorted(bank + tail)


class QuestionSelector(nn.Module):
    """Hard top-k inference; straight-through soft evidence gradients during training.

    Only question embeddings, stored features and relative ages enter selection.
    Evidence intervals, answers and hidden target cutoffs are not accepted.
    The gradient surrogate is explicit; it is not claimed to be unbiased.
    """

    def __init__(self, hidden, width=128):
        super().__init__()
        self.query = nn.Linear(hidden, width, bias=False)
        self.key = nn.Linear(hidden, width, bias=False)
        self.age = nn.Linear(1, width, bias=False)

    def forward(self, features, times, query_embedding, query_time, *, count=8):
        if features.ndim != 3 or features.shape[1] != 16 or len(features) != len(times):
            raise ValueError("stored steps must contain16spatial tokens")
        if not len(features) or count != 8 or query_embedding.shape != features.shape[-1:]:
            raise ValueError("invalid fixed-budget selector inputs")
        if any(t > query_time or t < 0 for t in times):
            raise ValueError("selector saw a future frame")
        age = features.new_tensor([math.log1p(query_time - t) for t in times])[:, None]
        keys = self.key(features.float().mean(1)) + self.age(age.float())
        query = self.query(query_embedding.float())
        scores = 12 * (F.normalize(keys, dim=-1) @ F.normalize(query, dim=-1))
        k = min(count, len(features))
        indices = scores.detach().topk(k).indices.sort().values
        hard = features[indices]
        if self.training:
            # Preserve the exact hard forward values; expose candidate values to
            # answer-loss gradients through a shared soft evidence surrogate.
            soft = torch.einsum("s,std->td", scores.softmax(0), features.float())
            hard = hard + (soft - soft.detach()).to(hard.dtype)[None]
        return hard, indices, scores
