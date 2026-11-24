"""T0: an ordinary causal temporal adapter, not the proposed novel T1.

Each spatial pool attends along time independently. The residual enters the LM
visual embeddings; the QA loss trains this module and LM LoRA jointly.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class CausalTemporalAdapter(nn.Module):
    def __init__(self, hidden: int, width: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, width)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0.,
                                       batch_first=True, norm_first=True)
            for _ in range(2)])
        self.up = nn.Linear(width, hidden, bias=False)
        # Exact appearance-preserving initialization. The output projection
        # receives gradients immediately; inner layers receive them afterwards.
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens, times):
        # [time, spatial bin, hidden] -> independent spatial-bin sequences.
        x = tokens.float()
        z = self.down(self.norm(x)).transpose(0, 1)
        t = torch.as_tensor(times, device=z.device, dtype=z.dtype)
        if t.ndim != 1 or len(t) != len(tokens) or not torch.isfinite(t).all():
            raise ValueError("finite timestamps matching temporal tokens required")
        if len(t) > 1 and not torch.all(t[1:] > t[:-1]):
            raise ValueError("temporal timestamps must increase")
        frequency = torch.exp(torch.arange(0, z.shape[-1], 2, device=z.device)
                              * (-math.log(10000.) / z.shape[-1]))
        angle = (t - t[0])[:, None] * frequency
        encoding = torch.stack((angle.sin(), angle.cos()), -1).flatten(1)
        z = z + encoding[None]
        mask = torch.ones(len(t), len(t), device=z.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            z = layer(z, src_mask=mask)
        return (x + self.up(z.transpose(0, 1))).to(tokens.dtype)


class CausalPatchPool(nn.Module):
    """Compress regions before the LM, using the preceding region to score current patches."""

    dense_input = True

    def __init__(self, hidden: int, width: int = 96):
        super().__init__()
        # Evaluation intervention; it is not a parameter or a training recipe.
        self.history_enabled = True
        self.norm = nn.LayerNorm(hidden)
        self.appearance = nn.Linear(hidden, width, bias=False)
        self.predict = nn.Linear(width, width, bias=False)
        self.score = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, tokens, grid, target, baseline):
        height, width = grid
        rows, cols = target
        blocks = [tokens[:, y * height // rows:(y + 1) * height // rows,
                         x * width // cols:(x + 1) * width // cols].flatten(1, 2)
                  for y in range(rows) for x in range(cols)]
        lengths = [block.shape[1] for block in blocks]
        widest = max(lengths)
        patches = torch.stack([
            torch.nn.functional.pad(block, (0, 0, 0, widest - length))
            for block, length in zip(blocks, lengths, strict=True)
        ], dim=1).float()
        valid = torch.arange(widest, device=tokens.device) < torch.tensor(lengths, device=tokens.device)[:, None]
        appearance = self.appearance(self.norm(patches))
        means = (appearance * valid[None, :, :, None]).sum(2) / valid.sum(-1)[:, None]
        previous = torch.cat((torch.zeros_like(means[:1]), means[:-1]), dim=0)
        if not self.history_enabled:
            previous = torch.zeros_like(previous)
        residual = appearance - self.predict(previous)[:, :, None, :]
        logits = self.score(residual.tanh()).squeeze(-1)
        weights = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        uniform = torch.zeros_like(logits).masked_fill(~valid, -torch.inf).softmax(-1)
        correction = ((weights - uniform)[..., None] * patches).sum(2)
        return (baseline.float() + correction).to(baseline.dtype)


class StaticPatchPool(CausalPatchPool):
    """Train the same region scorer with its previous-block condition always zero.

    Keep constructor RNG consumption and parameter layout identical to the causal
    arm. The retained predictor receives zero gradient and is audited unchanged;
    its presence does not imply equal effective capacity or deployment cost.
    """

    def __init__(self, hidden: int, width: int = 96):
        super().__init__(hidden, width)
        self.history_enabled = False


def attach_temporal(net, kind):
    if kind == "none":
        return
    variants = {"causal": CausalTemporalAdapter, "causal-pool": CausalPatchPool,
                "static-pool": StaticPatchPool}
    if kind not in variants or net.mode != "grid":
        raise ValueError("temporal modules require grid compression and a declared architecture")
    net.temporal = variants[kind](net.bb.hidden).to(net.bb.device)


def load_temporal(net, checkpoint):
    """Restore architecture weights from the integrity-checked training state."""
    state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
    saved = {k.removeprefix("temporal."): v for k, v in state["parameters"].items()
             if k.startswith("temporal.")}
    if not saved:
        raise ValueError("temporal checkpoint has no temporal parameters")
    net.temporal.load_state_dict(saved, strict=True)
