"""Bounded recurrent state and four-part transition evidence capsules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class MemoryState:
    working: Tensor
    events: Tensor
    valid: Tensor
    observed_start: Tensor
    known_at: Tensor
    before_valid: Tensor
    priorities: Tensor
    pointer: Tensor
    last_time: Tensor
    has_observation: Tensor

    def detach(self):
        return MemoryState(*(getattr(self, k).detach() for k in self.__dataclass_fields__))


class TransitionMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.state_queries = nn.Parameter(torch.randn(cfg.state_slots, d) * 0.02)
        self.event_types = nn.Parameter(torch.randn(4, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d * cfg.ffn_ratio), nn.GELU(), nn.Linear(d * cfg.ffn_ratio, d))
        self.gate = nn.Linear(2 * d, d)
        self.write_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.utility_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.constant_(self.write_head[-1].bias, 1.0)

    def initial(self, batch, device, dtype):
        c = self.cfg
        return MemoryState(
            torch.zeros(batch, c.state_slots, c.dim, device=device, dtype=dtype),
            torch.zeros(batch, c.event_capacity, c.capsule_tokens, c.dim, device=device, dtype=dtype),
            torch.zeros(batch, c.event_capacity, device=device, dtype=torch.bool),
            torch.zeros(batch, c.event_capacity, device=device),
            torch.zeros(batch, c.event_capacity, device=device),
            torch.zeros(batch, c.event_capacity, device=device, dtype=torch.bool),
            torch.zeros(batch, c.event_capacity, device=device),
            torch.zeros(batch, device=device, dtype=torch.long),
            torch.zeros(batch, device=device),
            torch.zeros(batch, device=device, dtype=torch.bool),
        )

    def _read(self, queries, values, mask=None):
        q = self.norm(queries)
        x = queries + self.attn(q, values, values, attn_mask=mask, need_weights=False)[0]
        return x + self.ffn(self.norm(x))

    def step(self, current, time, frame_valid, old: MemoryState):
        cfg = self.cfg
        previous = old.working if cfg.memory_mode != "recent" else torch.zeros_like(old.working)
        context = torch.cat([previous, current], dim=1)
        proposed = self._read(previous + self.state_queries[None], context)
        if cfg.state_update == "bounded":
            # `proposed` is residual on `previous`, so mixing it with `previous` integrates without decay.
            # Remove the residual and normalize: the gate then forms a convex mix of bounded values.
            proposed = F.layer_norm(proposed - previous, (cfg.dim,))
        gate = torch.sigmoid(self.gate(torch.cat([previous, proposed], dim=-1)))
        new_working = previous * (1 - gate) + proposed * gate
        if cfg.normalize_memory:
            new_working = F.layer_norm(new_working, (cfg.dim,))
        new_working = torch.where(frame_valid[:, None, None], new_working, old.working)
        before, after, detail = previous.mean(1), new_working.mean(1), current.mean(1)
        inputs = torch.cat([before, after, detail], dim=-1)
        write_logits = self.write_head(inputs).squeeze(-1)
        utility_logits = self.utility_head(inputs).squeeze(-1)
        seeds = torch.stack([before, after - before, after, detail], dim=1) + self.event_types[None]
        evidence = torch.cat([previous, current, new_working], dim=1)
        # Each capsule part has explicit access: before->old, after->new, detail->current.
        n_old, n_cur = previous.shape[1], current.shape[1]
        mask = torch.ones(4, evidence.shape[1], device=current.device, dtype=torch.bool)
        mask[0, :n_old] = False
        mask[1, :] = False
        mask[2, n_old + n_cur :] = False
        mask[3, n_old : n_old + n_cur] = False
        capsule = self._read(seeds, evidence, mask)
        if cfg.normalize_memory:
            capsule = F.layer_norm(capsule, (cfg.dim,))
        if cfg.capsule_tokens == 1:
            capsule = capsule.mean(1, keepdim=True)
        probability = write_logits.sigmoid()
        wants_write = (probability >= cfg.write_threshold) if cfg.write_policy == "learned" else frame_valid
        wants_write = wants_write & frame_valid & (cfg.memory_mode != "recent")
        empty = ~old.valid
        has_empty = empty.any(1)
        empty_index = empty.long().argmax(1)
        if cfg.retention_policy == "priority":
            lowest = old.priorities.masked_fill(~old.valid, torch.inf).argmin(1)
            index = torch.where(has_empty, empty_index, lowest)
            minimum = old.priorities.gather(1, index[:, None]).squeeze(1)
            wants_write = wants_write & (has_empty | (utility_logits.sigmoid() > minimum))
        else:
            index = torch.where(has_empty, empty_index, old.pointer)
        hard = wants_write.to(current.dtype)
        weight = (
            hard + probability - probability.detach()
            if self.training and cfg.write_policy == "learned"
            else hard
        )
        # Padded steps cannot receive a surrogate write gradient.
        weight = weight * frame_valid.to(weight.dtype)
        slot = torch.nn.functional.one_hot(index, cfg.event_capacity).to(current.dtype) * weight[:, None]
        events = old.events * (1 - slot[..., None, None]) + capsule[:, None] * slot[..., None, None]
        changed = torch.nn.functional.one_hot(index, cfg.event_capacity).bool() & wants_write[:, None]
        start = torch.where(old.has_observation, old.last_time, time)
        new = MemoryState(
            new_working,
            events,
            old.valid | changed,
            torch.where(changed, start[:, None], old.observed_start),
            torch.where(changed, time[:, None], old.known_at),
            torch.where(changed, old.has_observation[:, None], old.before_valid),
            torch.where(changed, utility_logits.sigmoid()[:, None], old.priorities),
            torch.where(wants_write, (index + 1) % cfg.event_capacity, old.pointer),
            torch.where(frame_valid, time, old.last_time),
            old.has_observation | frame_valid,
        )
        return new, {
            "write_logits": write_logits,
            "utility_logits": utility_logits,
            "written": wants_write,
            "capsule": capsule,
        }


class EvidenceReader(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.queries = nn.Parameter(torch.randn(cfg.read_tokens, cfg.dim) * 0.02)
        self.time_embedding = nn.Sequential(nn.Linear(3, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(cfg.dim),
                        "attn": nn.MultiheadAttention(cfg.dim, cfg.heads, batch_first=True),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(cfg.dim),
                            nn.Linear(cfg.dim, cfg.dim * cfg.ffn_ratio),
                            nn.GELU(),
                            nn.Linear(cfg.dim * cfg.ffn_ratio, cfg.dim),
                        ),
                    }
                )
                for _ in range(cfg.reader_layers)
            ]
        )

    def forward(self, query, current, memory: MemoryState, query_time):
        eligible = memory.valid & (memory.known_at <= query_time[:, None] + 1e-6)
        scores = torch.einsum("bd,bcd->bc", query, memory.events.mean(2)) / self.cfg.dim**0.5
        scores = scores.masked_fill(~eligible, -torch.inf)
        k = min(self.cfg.read_events, self.cfg.event_capacity)
        indices = scores.topk(k, dim=1).indices
        b = torch.arange(query.shape[0], device=query.device)[:, None]
        selected = memory.events[b, indices]
        selected_valid = eligible[b, indices]
        if self.cfg.retrieval_gradient:
            # Straight-through retrieval: the forward value is unchanged (factor 1), but the scores receive
            # gradient through the softmax probability of each selected event, so retrieval can learn.
            finite = scores.masked_fill(~eligible, torch.finfo(scores.dtype).min)
            probability = finite.float().softmax(1).to(selected.dtype) * eligible.to(selected.dtype)
            chosen = probability.gather(1, indices)
            selected = selected * (1 + chosen - chosen.detach())[..., None, None]
        start, known = memory.observed_start[b, indices], memory.known_at[b, indices]
        timing = torch.stack(
            [
                torch.log1p((query_time[:, None] - known).clamp_min(0)),
                torch.log1p((known - start).clamp_min(0)),
                memory.before_valid[b, indices].float(),
            ],
            dim=-1,
        )
        selected = selected + self.time_embedding(timing.to(query.dtype))[:, :, None, :]
        history = selected.flatten(1, 2)
        history_valid = selected_valid.repeat_interleave(self.cfg.capsule_tokens, dim=1)
        if self.cfg.memory_mode == "recent":
            history_valid = torch.zeros_like(history_valid)
        context = torch.cat([current, memory.working, history], dim=1)
        visible = torch.cat(
            [
                torch.ones(
                    context.shape[0],
                    current.shape[1] + memory.working.shape[1],
                    device=query.device,
                    dtype=torch.bool,
                ),
                history_valid,
            ],
            dim=1,
        )
        x = self.queries[None].expand(query.shape[0], -1, -1) + query[:, None]
        for layer in self.layers:
            x = (
                x
                + layer["attn"](
                    layer["norm"](x), context, context, key_padding_mask=~visible, need_weights=False
                )[0]
            )
            x = x + layer["ffn"](x)
        return x, {"indices": indices, "valid": selected_valid, "known_at": known, "observed_start": start}
