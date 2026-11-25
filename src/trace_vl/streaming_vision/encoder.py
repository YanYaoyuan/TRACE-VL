"""Selective upper-ViT refresh and learned correction of cached merged features."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb_vision,
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


class ContextCorrection(nn.Module):
    """Location-specific low-rank correction relative to each tile's refresh."""

    def __init__(self, shallow_dim: int, output_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.context = nn.Linear(shallow_dim, rank, bias=False)
        self.location = nn.Linear(shallow_dim, rank)
        self.output = nn.Linear(rank, output_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, scene, refresh_scenes, refresh_tiles):
        context_delta = self.context(scene - refresh_scenes)
        coefficients = 1 + self.location(refresh_tiles).tanh()
        return self.output(context_delta * coefficients)


@dataclass
class Refresh:
    features: torch.Tensor
    refreshed: torch.Tensor
    reason: str


class SpatialRefresh:
    """Per-camera state with fixed capacity. Upper layers write active patches only."""

    def __init__(self, visual, shallow_layers: int, fraction: float, full_every: int, cut_score: float):
        if not 0 < shallow_layers < len(visual.blocks):
            raise ValueError("shallow layers must leave at least one sparse upper layer")
        if not 0 < fraction <= 1 or full_every < 1 or not math.isfinite(cut_score) or cut_score <= 0:
            raise ValueError("invalid visual refresh schedule")
        self.visual = visual
        self.shallow_layers = shallow_layers
        self.fraction = fraction
        self.full_every = full_every
        self.cut_score = cut_score
        self.unit = visual.spatial_merge_unit
        self.head_dim = visual.blocks[shallow_layers].attn.head_dim
        self.geometry = None
        self.step = 0

    def _set_geometry(self, height, width, device):
        v = self.visual
        if height < 1 or width < 1 or height % v.spatial_merge_size or width % v.spatial_merge_size:
            raise ValueError("patch grid must consist of complete Qwen spatial merge tiles")
        self.geometry = (height, width)
        self.grid = torch.tensor([[1, height, width]], device=device)
        indices, weights = get_vision_interpolation_indices_and_weights(
            self.grid, num_grid_per_side=v.num_grid_per_side, mode=v.interpolation_mode,
            align_corners=v.interpolation_align_corners, spatial_merge_size=v.spatial_merge_size, kwargs={},
        )
        self.pos = (v.pos_embed(indices) * weights[:, :, None]).sum(1).to(v.pos_embed.weight.dtype)
        position_ids = get_vision_position_ids(self.grid, v.spatial_merge_size, kwargs={})
        self.rotary = v.rotary_pos_emb(self.pos, position_ids)
        self.cu_seqlens, self.max_seqlen = get_vision_attention_seqlens(self.grid, v.config, kwargs={})
        self.all_tiles = torch.arange(height * width // self.unit, device=device)
        self.offsets = torch.arange(self.unit, device=device)
        self.step = 0
        self.kv = []

    def _native_block(self, block, hidden):
        return block(hidden, cu_seqlens=self.cu_seqlens, max_seqlen=self.max_seqlen,
                     position_embeddings=self.rotary)

    def _dense_refresh(self, hidden):
        # Reuse the native upper blocks. Read their QKV outputs during the same
        # forwards so a full refresh does not repeat the projections.
        self.kv.clear()

        def capture(module, inputs, qkv):
            heads = qkv.shape[-1] // (3 * self.head_dim)
            q, k, v = qkv.reshape(len(qkv), 3, heads, self.head_dim).unbind(dim=1)
            _, k = apply_rotary_pos_emb_vision(q, k, *self.rotary)
            self.kv.append((k, v.contiguous()))

        for block in self.visual.blocks[self.shallow_layers:]:
            hook = block.attn.qkv.register_forward_hook(capture)
            try:
                hidden = self._native_block(block, hidden)
            finally:
                hook.remove()
        return self.visual.merger(hidden)

    def _sparse_refresh(self, hidden, active):
        rotary = tuple(x.index_select(0, active) for x in self.rotary)
        upper = self.visual.blocks[self.shallow_layers:]
        for block, (keys, values) in zip(upper, self.kv, strict=True):
            attn = block.attn
            q, k, v = attn.qkv(block.norm1(hidden)).reshape(
                len(active), 3, attn.num_heads, attn.head_dim,
            ).unbind(dim=1)
            q, k = apply_rotary_pos_emb_vision(q, k, *rotary)
            keys.index_copy_(0, active, k)
            values.index_copy_(0, active, v)
            q = q.transpose(0, 1)[None]
            k = keys.transpose(0, 1)[None]
            v = values.transpose(0, 1)[None]
            mixed = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling)
            hidden = hidden + attn.proj(mixed[0].transpose(0, 1).reshape_as(hidden))
            hidden = hidden + block.mlp(block.norm2(hidden))
        return self.visual.merger(hidden)

    @torch.no_grad()
    def update(self, pixels, height: int, width: int) -> Refresh:
        if len(pixels) != height * width:
            raise ValueError("update requires exactly one Qwen temporal patch")
        if self.geometry != (height, width):
            self._set_geometry(height, width, pixels.device)
        hidden = self.visual.patch_embed(pixels) + self.pos
        for block in self.visual.blocks[:self.shallow_layers]:
            hidden = self._native_block(block, hidden)

        tiles = hidden.reshape(len(self.all_tiles), self.unit, -1).float()
        pooled = tiles.mean(1)
        scene = pooled.mean(0)
        unit_tiles = F.normalize(tiles, dim=-1)
        n_refresh = math.ceil(len(pooled) * self.fraction)
        scene_cut = False
        if self.step:
            # Compare adjacent steps for scene cuts, and compare to the last
            # refresh of each tile to catch small cumulative changes.
            distance = 1 - F.cosine_similarity(scene, self.scene, dim=0)
            scene_cut = bool(distance > self.cut_score)
        if self.step == 0 or scene_cut or self.step % self.full_every == 0 or n_refresh == len(pooled):
            reason = ("initial" if self.step == 0 else "scene-cut" if scene_cut else
                      "full-budget" if n_refresh == len(pooled) else "periodic")
            active = self.all_tiles
            self.features = self._dense_refresh(hidden)
            self.tile_reference = unit_tiles.clone()
            self.refresh_tiles = pooled.clone()
            self.refresh_scenes = scene.expand_as(pooled).clone()
        else:
            reason = "selective"
            scores = (unit_tiles - self.tile_reference).square().sum(-1).mean(-1)
            active = scores.topk(n_refresh).indices.sort().values
            patches = (active[:, None] * self.unit + self.offsets).flatten()
            fresh = self._sparse_refresh(hidden.index_select(0, patches), patches)
            self.features.index_copy_(0, active, fresh)
            self.tile_reference.index_copy_(0, active, unit_tiles.index_select(0, active))
            self.refresh_tiles.index_copy_(0, active, pooled.index_select(0, active))
            self.refresh_scenes.index_copy_(0, active, scene.expand(len(active), -1))
        self.scene = scene
        self.step += 1
        refreshed = torch.zeros(len(pooled), dtype=torch.bool, device=hidden.device)
        refreshed[active] = True
        # Returned features may enter the decoder's past observation window.
        # Copy out before a later step overwrites this camera's cache.
        return Refresh(self.features.clone(), refreshed, reason)

    def correction_inputs(self, stale):
        return self.scene, self.refresh_scenes[stale], self.refresh_tiles[stale]

    @torch.no_grad()
    def corrected(self, result: Refresh, corrector: ContextCorrection):
        stale = ~result.refreshed
        # A full refresh is exactly the dense native path.
        if result.reason == "selective":
            delta = corrector(*self.correction_inputs(stale))
            result.features[stale] += delta.to(result.features.dtype)
        return result.features
