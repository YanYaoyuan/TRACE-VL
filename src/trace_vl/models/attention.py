"""Incremental causal attention. Caches contain this layer's projected K/V only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CacheEntry:
    key: Tensor
    value: Tensor
    positions: Tensor
    time: Tensor
    valid: Tensor

    def detach(self):
        return CacheEntry(*(getattr(self, f).detach() for f in self.__dataclass_fields__))


class CausalSTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ratio: int,
        cache_frames: int,
        cache_seconds: float,
        dropout: float = 0.0,
        state_kv_norm: bool = False,
    ):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.cache_frames, self.cache_seconds = cache_frames, cache_seconds
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(dim)
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))
        self.position_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, heads))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, dim))
        self.dropout = dropout
        self.state_kv_norm = state_kv_norm

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def step(
        self,
        x: Tensor,
        positions: Tensor,
        time: Tensor,
        valid: Tensor,
        previous_state: Tensor,
        cache: tuple[CacheEntry, ...],
    ):
        normalized = self.spatial_norm(x)
        x = x + self.spatial(normalized, normalized, normalized, need_weights=False)[0]
        normalized = self.temporal_norm(x)
        current = CacheEntry(
            self._heads(self.k(normalized)), self._heads(self.v(normalized)), positions, time, valid
        )
        entries = (*cache, current)
        keys = torch.cat([e.key for e in entries], dim=2)
        values = torch.cat([e.value for e in entries], dim=2)
        pos = torch.cat([e.positions for e in entries], dim=1)
        times = torch.cat([e.time[:, None].expand(-1, e.positions.shape[1]) for e in entries], dim=1)
        valids = torch.cat([e.valid[:, None].expand(-1, e.positions.shape[1]) for e in entries], dim=1)
        delta = time[:, None] - times
        allowed = valids & (delta >= -1e-6) & (delta <= self.cache_seconds)
        xy = positions[:, :, None, :] - pos[:, None, :, :]
        dt = torch.log1p(delta.clamp_min(0))[:, None, :].expand(-1, x.shape[1], -1)
        features = torch.cat([xy, dt[..., None]], dim=-1)
        bias = self.position_bias(features.to(normalized.dtype)).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~allowed[:, None, None, :], -torch.inf)
        # State keys are always available, including for padded frames. This prevents all-masked rows.
        # Optionally reuse the temporal LayerNorm so state drift cannot dominate every layer's attention.
        state = self.temporal_norm(previous_state) if self.state_kv_norm else previous_state
        sk, sv = self._heads(self.k(state)), self._heads(self.v(state))
        keys, values = torch.cat([keys, sk], 2), torch.cat([values, sv], 2)
        state_bias = torch.zeros(*bias.shape[:-1], sk.shape[2], device=x.device, dtype=bias.dtype)
        bias = torch.cat([bias, state_bias], dim=-1)
        attended = F.scaled_dot_product_attention(
            self._heads(self.q(normalized)),
            keys,
            values,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0,
        )
        attended = attended.transpose(1, 2).reshape_as(x)
        x = x + self.out(attended)
        x = x + self.ffn(self.ffn_norm(x))
        return x, entries[-self.cache_frames :]
