"""Five causal heads with identical visual/query features and supervision.

Event state is a candidate mechanism, not an established contribution. Its state
can be overwritten by a learned gate; no event-clear labels are invented.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def position(n, d, device):
    t = torch.arange(n, device=device).float()[:, None]
    f = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000) / d))
    p = torch.zeros(n, d, device=device)
    p[:, 0::2], p[:, 1::2] = (t * f).sin(), (t * f).cos()
    return p


class ResponseHead(nn.Module):
    def __init__(self, width, cfg, arm):
        super().__init__()
        self.arm, self.cfg = arm, cfg
        d, heads, layers = cfg['dim'], cfg['heads'], cfg['layers']
        if arm == 'clock':
            self.clock = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, 1))
            return
        self.visual = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d))
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d))
        self.spatial = nn.MultiheadAttention(d, heads, dropout=0, batch_first=True)
        self.elapsed = nn.Linear(3, d)
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 1)
        if arm == 'frame':
            self.body = nn.Sequential(*[nn.Sequential(nn.Linear(d, d), nn.GELU()) for _ in range(layers)])
        elif arm == 'gru':
            self.body = nn.GRU(d, d, layers, batch_first=True)
        elif arm == 'temporal':
            # FFN=6d matches the two-attention event-state block's parameter count.
            self.body = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d, heads, dim_feedforward=6*d, dropout=0, batch_first=True, norm_first=True), layers,
                enable_nested_tensor=False)
        elif arm == 'event_state':
            self.initial = nn.Parameter(torch.randn(cfg['state_tokens'], d) * .02)
            self.body = nn.TransformerDecoder(nn.TransformerDecoderLayer(
                d, heads, dim_feedforward=4*d, dropout=0, batch_first=True, norm_first=True), layers)
            self.replace = nn.Linear(2*d, d)
        else:
            raise ValueError(arm)

    def forward(self, visual, query, elapsed):
        # B,T,K,C and B,C. Each example starts when its condition is registered.
        b, t, k, _ = visual.shape
        if self.arm == 'clock':
            clocks = torch.stack([elapsed / 120, torch.log1p(elapsed) / 5, torch.ones_like(elapsed)], -1)
            return self.clock(clocks).squeeze(-1)
        q = self.query(query)
        v = self.visual(visual).reshape(b*t, k, -1)
        qq = q[:, None].expand(-1, t, -1).reshape(b*t, 1, -1)
        x = self.spatial(qq, v, v, need_weights=False)[0].reshape(b, t, -1) + q[:, None]
        clocks = torch.stack([elapsed / 120, torch.log1p(elapsed) / 5, torch.ones_like(elapsed)], -1)
        x = x + self.elapsed(clocks)
        if self.arm == 'frame':
            y = self.body(x)
        elif self.arm == 'gru':
            y, _ = self.body(x)
        elif self.arm == 'temporal':
            idx = torch.arange(t, device=x.device)
            mask = (idx[None, :] > idx[:, None]) | (idx[None, :] < idx[:, None]-self.cfg['window']+1)
            y = self.body(x + position(t, x.shape[-1], x.device), mask=mask)
        else:
            state = self.initial[None].expand(b, -1, -1) + q[:, None]
            history, outputs = [], []
            for i in range(t):
                history.append(x[:, i:i+1])
                history = history[-self.cfg['window']:]
                mem = torch.cat(history, 1)
                mem = mem + position(mem.shape[1], mem.shape[-1], x.device)
                candidate = self.body(state, mem)
                gate = self.replace(torch.cat([state, x[:, i:i+1].expand_as(state)], -1)).sigmoid()
                state = (1-gate)*state + gate*candidate
                outputs.append(state.mean(1))
            y = torch.stack(outputs, 1)
        return self.output(self.norm(y)).squeeze(-1)
