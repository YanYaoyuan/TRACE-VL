"""TRACE-VL: observation is query-independent; readout is query-conditioned."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import ModelConfig, factor_grid
from .attention import CacheEntry, CausalSTBlock
from .backbones import HFLanguage, HFVision, TinyLanguage, TinyVision
from .memory import EvidenceReader, MemoryState, TransitionMemory


@dataclass
class StreamState:
    memory: MemoryState
    caches: tuple[tuple[CacheEntry, ...], ...]
    last_visual: Tensor
    previous_frame: Tensor | None = None

    def detach(self):
        return StreamState(
            self.memory.detach(),
            tuple(tuple(e.detach() for e in c) for c in self.caches),
            self.last_visual.detach(),
            None if self.previous_frame is None else self.previous_frame.detach(),
        )


class TraceVL(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.backend == "hf":
            self.vision = HFVision(cfg.vision_model, cfg.local_files_only)
            self.language = HFLanguage(
                cfg.language_model, cfg.local_files_only, cfg.text_limit, cfg.lora_rank, cfg.prompt_format
            )
            if (
                self.language.dim != cfg.language_dim
                or self.vision.model.config.hidden_size != cfg.vision_dim
            ):
                raise ValueError("backbone dimensions do not match the configuration")
        else:
            self.vision, self.language = (
                TinyVision(cfg.vision_dim),
                TinyLanguage(cfg.language_dim, cfg.text_limit, cfg.prompt_format),
            )
        if cfg.freeze_vision:
            self.vision.requires_grad_(False)
        if cfg.freeze_language:
            # PEFT already freezes base weights and exposes only explicitly inserted adapters.
            if not cfg.lora_rank:
                self.language.requires_grad_(False)
        self.rgb_bridge = nn.Linear(cfg.vision_dim, cfg.dim)
        self.xy_embedding = nn.Sequential(nn.Linear(2, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))
        self.motion = nn.Sequential(
            nn.Conv2d(3, cfg.dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.dim // 2, cfg.dim, 3, padding=1),
        )
        self.token_types = nn.Parameter(torch.randn(2, cfg.dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CausalSTBlock(
                    cfg.dim,
                    cfg.heads,
                    cfg.ffn_ratio,
                    cfg.cache_frames,
                    cfg.cache_seconds,
                    cfg.dropout,
                    cfg.state_kv_norm,
                )
                for _ in range(cfg.st_layers)
            ]
        )
        self.memory = TransitionMemory(cfg)
        self.reader = EvidenceReader(cfg)
        self.projector = nn.Sequential(
            nn.Linear(cfg.dim, cfg.language_dim), nn.GELU(), nn.Linear(cfg.language_dim, cfg.language_dim)
        )
        self.alert_head = nn.Sequential(nn.Linear(3 * cfg.dim + 2, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))
        if cfg.query_projection == "separate":
            # Decoupled from the visual->language projector; starts equal to the tied map.
            self.query_projection = nn.Linear(cfg.language_dim, cfg.dim)
            self.reset_query_projection()

    @torch.no_grad()
    def reset_query_projection(self):
        self.query_projection.weight.copy_(self.projector[0].weight.t())
        self.query_projection.bias.zero_()

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_vision:
            self.vision.eval()
        if self.cfg.freeze_language and not self.cfg.lora_rank:
            self.language.eval()
        return self

    def initial_state(self, batch, device=None):
        parameter = self.rgb_bridge.weight
        device = device or parameter.device
        memory = self.memory.initial(batch, device, parameter.dtype)
        visual = torch.zeros(
            batch,
            self.cfg.spatial_tokens + self.cfg.motion_tokens,
            self.cfg.dim,
            device=device,
            dtype=parameter.dtype,
        )
        return StreamState(memory, tuple(() for _ in self.blocks), visual)

    def encode_text(self, texts):
        encoded = self.language.encode_text(texts, self.cfg.query_encoding)
        if self.cfg.query_projection == "separate":
            return self.query_projection(encoded.to(self.query_projection.weight.dtype))
        weight = self.projector[0].weight
        # Tie the query projection to the transpose of the visual->language projection.
        return F.linear(encoded.to(weight.dtype), weight.t())

    def _visual_tokens(self, images, previous):
        features, (height, width) = self.vision(images)
        n = features.shape[1]
        if n < self.cfg.spatial_tokens:
            raise ValueError(f"only {n} patches, requested {self.cfg.spatial_tokens}; increase image_size")
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=images.device),
            torch.linspace(0, 1, width, device=images.device),
            indexing="ij",
        )
        grid = torch.stack([xx.flatten(), yy.flatten()], -1)
        if self.cfg.token_selection == "pool":
            selected, positions = self._pooled_tokens(features, grid, height, width)
        else:
            selected, positions = self._salient_tokens(features, grid)
        selected = self.rgb_bridge(selected.to(self.rgb_bridge.weight.dtype))
        selected = selected + self.xy_embedding(positions.to(selected.dtype)) + self.token_types[0]
        if not self.cfg.motion_tokens:
            return selected, positions
        motion, motion_pos = self._motion_tokens(images, previous)
        return torch.cat([selected, motion], dim=1), torch.cat([positions, motion_pos], dim=1)

    def _salient_tokens(self, features, grid):
        batch, n, _ = features.shape
        coverage = max(1, self.cfg.spatial_tokens // 3)
        fixed = torch.linspace(0, n - 1, coverage, device=features.device).round().long().unique()
        score = (features.float() - features.float().mean(1, keepdim=True)).square().mean(-1)
        score[:, fixed] = -torch.inf
        extra = score.topk(self.cfg.spatial_tokens - fixed.numel(), dim=1).indices
        indices = torch.cat([fixed[None].expand(batch, -1), extra], dim=1)
        selected = features.gather(1, indices[..., None].expand(-1, -1, features.shape[-1]))
        return selected, grid[indices]

    @staticmethod
    def _pool_grid(values, grid, height, width, rows, cols):
        """Average [B, H*W, C] values and their xy coordinates into a fixed rows x cols layout."""
        batch, _, channels = values.shape
        pooled = (
            F.adaptive_avg_pool2d(
                values.transpose(1, 2).reshape(batch, channels, height, width), (rows, cols)
            )
            .flatten(2)
            .transpose(1, 2)
        )
        xy = F.adaptive_avg_pool2d(grid.t().reshape(1, 2, height, width), (rows, cols))
        return pooled, xy.flatten(2).transpose(1, 2).expand(batch, -1, -1)

    def _pooled_tokens(self, features, grid, height, width):
        # Stable, uniform coverage: the same token indexes the same region in every frame.
        rows, cols = factor_grid(self.cfg.spatial_tokens)
        if rows > height or cols > width:
            raise ValueError(f"cannot pool a {height}x{width} patch grid into {rows}x{cols} tokens")
        pooled, positions = self._pool_grid(features.float(), grid, height, width, rows, cols)
        return pooled.to(features.dtype), positions

    def _motion_tokens(self, images, previous):
        # RGB difference is a documented fallback, not a native codec residual extractor.
        batch = images.shape[0]
        delta = torch.zeros_like(images) if previous is None else images - previous
        motion = self.motion(F.interpolate(delta, size=(16, 16), mode="bilinear", align_corners=False))
        if self.cfg.motion_layout == "grid":
            rows, cols = factor_grid(self.cfg.motion_tokens)
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, 16, device=images.device),
                torch.linspace(0, 1, 16, device=images.device),
                indexing="ij",
            )
            grid = torch.stack([xx.flatten(), yy.flatten()], -1)
            motion, motion_pos = self._pool_grid(motion.flatten(2).transpose(1, 2), grid, 16, 16, rows, cols)
            motion = motion + self.xy_embedding(motion_pos.to(motion.dtype)) + self.token_types[1]
            return motion, motion_pos
        motion = F.adaptive_avg_pool2d(motion, (1, self.cfg.motion_tokens)).flatten(2).transpose(1, 2)
        motion = motion + self.token_types[1]
        motion_pos = images.new_zeros(batch, self.cfg.motion_tokens, 2) + 0.5
        return motion, motion_pos

    def observe_step(self, images, time, state: StreamState, rule_embedding=None, frame_valid=None):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be [batch, 3, height, width]")
        frame_valid = (
            torch.ones(images.shape[0], device=images.device, dtype=torch.bool)
            if frame_valid is None
            else frame_valid
        )
        if not torch.isfinite(time).all():
            raise ValueError("nonfinite observation timestamp")
        out_of_order = frame_valid & state.memory.has_observation & (time <= state.memory.last_time)
        if out_of_order.any():
            raise ValueError("valid observation timestamps must be strictly increasing")
        x, positions = self._visual_tokens(images, state.previous_frame)
        previous_state = state.memory.working
        if self.cfg.memory_mode == "recent":
            previous_state = torch.zeros_like(previous_state)
        caches = []
        for block, cache in zip(self.blocks, state.caches):
            x, updated = block.step(x, positions, time, frame_valid, previous_state, cache)
            caches.append(updated)
        memory, aux = self.memory.step(x, time, frame_valid, state.memory)
        visual = torch.where(frame_valid[:, None, None], x, state.last_visual)
        previous_frame = (
            images
            if state.previous_frame is None
            else torch.where(frame_valid[:, None, None, None], images, state.previous_frame)
        )
        new_state = StreamState(memory, tuple(caches), visual, previous_frame)
        rule = torch.zeros_like(memory.working.mean(1)) if rule_embedding is None else rule_embedding
        event_mask = memory.valid.to(memory.events.dtype)
        event_summary = (memory.events.mean(2) * event_mask[..., None]).sum(1)
        event_summary = event_summary / event_mask.sum(1, keepdim=True).clamp_min(1)
        oldest = torch.where(memory.valid, memory.known_at, time[:, None]).min(1).values
        age = torch.log1p((time - oldest).clamp_min(0))
        gap = torch.log1p(
            torch.where(
                state.memory.has_observation,
                (time - state.memory.last_time).clamp_min(0),
                torch.zeros_like(time),
            )
        )
        clock = torch.stack([age, gap], -1).to(x.dtype)
        logits = self.alert_head(torch.cat([memory.working.mean(1), event_summary, rule, clock], -1)).squeeze(
            -1
        )
        aux["alert_logits"] = logits
        return new_state, aux

    def observe(self, frames, timestamps, frame_mask=None, rules=None, detach_every=0):
        if frames.ndim != 5 or timestamps.shape != frames.shape[:2]:
            raise ValueError("expected frames [B,T,3,H,W] and timestamps [B,T]")
        batch, steps = frames.shape[:2]
        state = self.initial_state(batch, frames.device)
        rule = self.encode_text(rules) if rules is not None else None
        results = []
        for step in range(steps):
            valid = None if frame_mask is None else frame_mask[:, step]
            state, aux = self.observe_step(frames[:, step], timestamps[:, step], state, rule, valid)
            results.append(aux)
            if detach_every and (step + 1) % detach_every == 0 and step + 1 < steps:
                state = state.detach()
        stacked = {
            name: torch.stack([r[name] for r in results], dim=1)
            for name in ("alert_logits", "write_logits", "utility_logits", "written")
        }
        return state, stacked

    def evidence(self, state, questions, query_time=None):
        query_time = state.memory.last_time if query_time is None else query_time
        if (query_time < state.memory.last_time - 1e-6).any():
            raise ValueError("query precedes ingested state; use a state snapshot at the query cutoff")
        tokens, info = self.reader(self.encode_text(questions), state.last_visual, state.memory, query_time)
        return self.projector(tokens), info

    def forward(self, batch, loss_weights=None, detach_every=0):
        weights = {"answer": 1.0, "alert": 1.0, "write": 0.2, "utility": 0.1, "contrast": 0.0}
        weights.update(loss_weights or {})
        state, aux = self.observe(
            batch["frames"], batch["timestamps"], batch.get("frame_mask"), batch.get("rules"), detach_every
        )
        prefix, evidence = self.evidence(state, batch["questions"])
        losses = {"answer": self.language.loss(prefix, batch["questions"], batch["answers"])}
        if weights.get("contrast", 0) and prefix.shape[0] > 1:
            # Answers must be less likely under another sample's video evidence (rolled within the batch).
            mismatched = self.language.loss(prefix.roll(1, dims=0), batch["questions"], batch["answers"])
            margin = weights.get("contrast_margin", 0.2)
            losses["contrast"] = F.relu(margin + losses["answer"] - mismatched)
        for key, prediction in [
            ("alert", "alert_logits"),
            ("write", "write_logits"),
            ("utility", "utility_logits"),
        ]:
            target = batch.get(key + "_labels")
            if target is not None:
                visible = target >= 0
                if "frame_mask" in batch:
                    visible = visible & batch["frame_mask"]
                if visible.any():
                    losses[key] = F.binary_cross_entropy_with_logits(
                        aux[prediction][visible].float(), target[visible].float()
                    )
        total = sum(weights.get(k, 0.0) * v for k, v in losses.items())
        return {"loss": total, "losses": losses, "state": state, "aux": aux, "evidence": evidence}

    @torch.no_grad()
    def answer(self, state, questions, max_new_tokens=32):
        prefix, info = self.evidence(state, questions)
        return self.language.generate(prefix, questions, max_new_tokens), info

    def parameter_report(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "components": {n: sum(p.numel() for p in m.parameters()) for n, m in self.named_children()},
        }
