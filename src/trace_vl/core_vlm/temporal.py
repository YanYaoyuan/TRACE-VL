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


def attach_temporal(net, kind):
    if kind == "none":
        return
    if kind != "causal" or net.mode != "grid":
        raise ValueError("causal temporal adapter requires grid compression")
    net.temporal = CausalTemporalAdapter(net.bb.hidden).to(net.bb.device)


def load_temporal(net, checkpoint):
    """Restore architecture weights from the integrity-checked training state."""
    state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
    saved = {k.removeprefix("temporal."): v for k, v in state["parameters"].items()
             if k.startswith("temporal.")}
    if not saved:
        raise ValueError("temporal checkpoint has no temporal parameters")
    net.temporal.load_state_dict(saved, strict=True)
