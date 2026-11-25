"""Acceptance probe for any future selector: does the question determine what is retrieved?

Relevant entries occupy random bank positions, so age carries no information.
Passing is necessary, not evidence of video-QA benefit. Nonzero gradients are
deliberately not part of this probe.
"""

from __future__ import annotations

import torch


def probe(select, *, hidden=64, bank=36, count=8, concepts=12, trials=64, seed=0):
    """select(features, times, query, query_time) -> bank indices.

    Every bank holds each concept in exactly three random slots. Each bank is
    queried twice with different concepts; the age vector is identical.
    """
    if bank != 3 * concepts or concepts > hidden:
        raise ValueError("bank must hold each concept three times")
    generator = torch.Generator().manual_seed(seed)
    codes = torch.linalg.qr(torch.randn(hidden, hidden, generator=generator))[0][:concepts]
    recall, changed = [], []
    for _ in range(trials):
        labels = torch.arange(concepts).repeat(3)[torch.randperm(bank, generator=generator)]
        noise = torch.randn(bank, 16, hidden, generator=generator) * 0.05
        features = codes[labels][:, None] + noise
        times = [float(2 * i) for i in range(bank)]
        a, b = torch.randperm(concepts, generator=generator)[:2].tolist()
        picked = []
        for concept in (a, b):
            indices = [int(i) for i in select(features, times, codes[concept], 2.0 * bank)]
            if len(indices) != count or len(set(indices)) != count:
                raise ValueError("selector violated its read budget")
            relevant = set((labels == concept).nonzero().flatten().tolist())
            recall.append(len(relevant & set(indices)) / len(relevant))
            picked.append(set(indices))
        changed.append(picked[0] != picked[1])
    return {
        "relevant_recall": sum(recall) / len(recall),
        "question_change_rate": sum(changed) / len(changed),
        "chance_recall": count / bank,
    }


def accepted(result):
    return result["relevant_recall"] >= 0.9 and result["question_change_rate"] >= 0.9
