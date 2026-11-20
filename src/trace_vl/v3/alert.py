"""Alert head: Jev-style classification for surveillance, answered every step without decoding.

Borrowed from TypeSafe AI's Jev (System One model, Sep 2026) for classification only: typed alert
questions are registered once, evaluated in parallel and in isolation against the current state, and
return a probability (plus a confidence for multi-class labels) instead of generated text:

    noul    "Is the scene abnormal?", "Is there a hazard?"        -> p in [0, 1]
    choice  "Which incident is this?" over labels (<= 255)       -> label, probabilities, confidence

The state is read through latent connections, not text: the current step's hidden states tapped from
several LM layers, the Gated DeltaNet recurrent states read as associative memories with a learned
per-question key (r = S^T k, as the GDN layer reads itself), a bounded memory of earlier steps' latent
tokens, and transition clocks (ages of active transitions, for duration rules such as "left > 10 min").
Open-ended questions stay with the language model (`StreamSession.ask`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

KINDS = ("noul", "choice")
MAX_OPTIONS = 255


@dataclass
class Question:
    id: str
    kind: str
    text: str
    options: list[str] = field(default_factory=list)  # labels for "choice"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"alert question kind must be one of {KINDS}")
        if self.kind == "choice" and not 2 <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(f"choice questions need 2..{MAX_OPTIONS} labels")
        if self.kind == "noul" and self.options:
            raise ValueError("noul questions take no options")


@dataclass
class EncodedQuestions:
    questions: list[Question]
    text: Tensor  # [Q, D] pooled LM hidden of each question text
    options: list[Tensor | None]  # per question [K, D] pooled label texts, or None


@torch.no_grad()
def encode_questions(backbone, questions: list[Question]) -> EncodedQuestions:
    """Encode each question and label text in its own LM pass (isolated, no stream cache)."""
    pooled = backbone.encode_texts([q.text for q in questions])
    options = [backbone.encode_texts(q.options) if q.options else None for q in questions]
    return EncodedQuestions(questions, pooled, options)


class DecisionMemory:
    """Bounded FIFO of earlier steps' latent tokens (step ends, gist/transition tokens) with times."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.tokens: Tensor | None = None
        self.times: Tensor | None = None

    def add(self, hidden: Tensor, t: float):
        hidden = hidden.detach()
        times = torch.full((hidden.shape[0],), float(t), device=hidden.device)
        self.tokens = hidden if self.tokens is None else torch.cat([self.tokens, hidden])[-self.capacity :]
        self.times = times if self.times is None else torch.cat([self.times, times])[-self.capacity :]


class DecisionLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_q, self.norm_c = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, q: Tensor, ctx: Tensor) -> Tensor:
        """q [Q, 1, h] (one query per batch item = isolation), ctx [Q, C, h]."""
        q = q + self.attn(self.norm_q(q), self.norm_c(ctx), self.norm_c(ctx), need_weights=False)[0]
        return q + self.ffn(q)


class AlertHead(nn.Module):
    def __init__(self, model_dim: int, cfg=None, state_heads: int = 16, state_k: int = 128, state_v: int = 128):
        super().__init__()
        from .config import AlertConfig

        cfg = cfg or AlertConfig()
        self.cfg = cfg
        hidden = cfg.hidden
        self.state_shape = (len(cfg.state_layers), state_heads, state_k, state_v)
        self.q_proj = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, hidden))
        self.ctx_proj = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, hidden))
        self.opt_proj = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, hidden))
        self.kind_embed = nn.Embedding(len(KINDS), hidden)
        self.layer_embed = nn.Embedding(max(1, len(cfg.taps)), hidden)
        self.source_embed = nn.Embedding(2, hidden)  # 0 = current step, 1 = memory
        self.pos = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, hidden))
        self.age = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, hidden))
        self.clock = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, hidden))
        self.state_key = nn.Linear(hidden, len(cfg.state_layers) * state_heads * state_k)
        self.state_out = nn.Sequential(nn.LayerNorm(state_heads * state_v), nn.Linear(state_heads * state_v, hidden))
        self.state_embed = nn.Embedding(max(1, len(cfg.state_layers)), hidden)
        self.layers = nn.ModuleList([DecisionLayer(hidden, cfg.heads) for _ in range(cfg.layers)])
        self.out_norm = nn.LayerNorm(hidden)
        self.noul = nn.Linear(hidden, 1)
        self.option_query = nn.Linear(hidden, hidden)
        self.option_scale = nn.Parameter(torch.tensor(math.log(1 / math.sqrt(hidden))))
        self.confidence = nn.Sequential(nn.Linear(hidden + 1, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, encoded: EncodedQuestions, step_taps: Tensor, step_xy: Tensor, memory: DecisionMemory | None,
                now: float, clock: Tensor, states: Tensor | None = None) -> list[dict]:  # fmt: skip
        """step_taps [T, n, D]: current step's hidden states at the tapped layers; step_xy [n, 2] in [0, 1];
        states [G, H, K, V]: Gated DeltaNet recurrent states, or None."""
        dtype = self.q_proj[1].weight.dtype
        if step_taps.dim() == 2:
            step_taps = step_taps[None]
        pos = self.pos(step_xy.to(dtype)) + self.source_embed.weight[0]
        parts = [self.ctx_proj(h.to(dtype)) + pos + self.layer_embed.weight[i] for i, h in enumerate(step_taps)]
        if memory is not None and memory.tokens is not None:
            age = torch.log1p((now - memory.times).clamp_min(0))[:, None].to(dtype)
            parts.append(self.ctx_proj(memory.tokens.to(dtype)) + self.age(age) + self.source_embed.weight[1])
        ctx = torch.cat(parts)  # [C, h]
        kinds = torch.tensor([KINDS.index(q.kind) for q in encoded.questions], device=ctx.device)
        q = self.q_proj(encoded.text.to(dtype)) + self.kind_embed(kinds) + self.clock(clock.to(ctx.device, dtype))[None]
        nq = q.shape[0]
        ctx_b = ctx[None].expand(nq, -1, -1)
        if states is not None and self.training and random.random() < self.cfg.state_dropout:
            states = None  # in training the state is only available at the clip end; learn to cope without it
        if states is not None:
            g, h_s, k_s, v_s = self.state_shape
            keys = F.normalize(self.state_key(q).view(nq, g, h_s, k_s).float(), dim=-1)  # GDN uses l2-normed keys
            read = torch.einsum("qghk,ghkv->qghv", keys, states.float().to(ctx.device))  # r = S^T k per head
            tokens = self.state_out(read.reshape(nq, g, h_s * v_s).to(dtype)) + self.state_embed.weight[None]
            ctx_b = torch.cat([ctx_b, tokens], dim=1)  # per-question context: isolation is preserved
        q = q[:, None, :]
        for layer in self.layers:
            q = layer(q, ctx_b)
        u = self.out_norm(q[:, 0])  # [Q, h]
        results = []
        for i, question in enumerate(encoded.questions):
            out = {"id": question.id, "kind": question.kind}
            if question.kind == "noul":
                logit = self.noul(u[i]).squeeze(-1)
                out.update(logits=logit, value=torch.sigmoid(logit))
            else:
                labels = self.opt_proj(encoded.options[i].to(dtype))  # [K, h]
                logits = (labels @ self.option_query(u[i])) * self.option_scale.exp()
                probs = logits.softmax(-1)
                conf_logit = self.confidence(torch.cat([u[i], probs.max()[None].detach()])).squeeze(-1)
                out.update(logits=logits, probabilities=probs, value=probs.argmax(), confidence=torch.sigmoid(conf_logit),
                           confidence_logit=conf_logit)  # fmt: skip
            results.append(out)
        return results


def alert_loss(results: list[dict], targets: dict) -> tuple[Tensor | None, dict]:
    """targets[id] = 0/1 (noul) or a label index (choice). Confidence learns whether the arg-max is right."""
    parts = {"noul": [], "choice": [], "confidence": []}
    for r in results:
        if r["id"] not in targets:
            continue
        target = targets[r["id"]]
        if r["kind"] == "noul":
            y = torch.tensor(float(target), device=r["logits"].device)
            parts["noul"].append(F.binary_cross_entropy_with_logits(r["logits"].float(), y))
        else:
            y = torch.tensor([int(target)], device=r["logits"].device)
            parts["choice"].append(F.cross_entropy(r["logits"].float()[None], y))
            correct = (r["probabilities"].argmax() == int(target)).float().detach()
            # BCE-with-logits on the pre-sigmoid confidence: plain BCE is not autocast-safe (PR #3 review).
            parts["confidence"].append(F.binary_cross_entropy_with_logits(r["confidence_logit"].float(), correct))
    weights = {"noul": 1.0, "choice": 1.0, "confidence": 0.2}
    losses, stats = [], {}
    for name, values in parts.items():
        if values:
            value = torch.stack(values).mean()
            losses.append(weights[name] * value)
            stats[f"alert_{name}"] = float(value)
    return (torch.stack(losses).sum() if losses else None), stats


def public(results: list[dict]) -> list[dict]:
    """JSON-friendly answers a caller can branch on."""
    out = []
    for r in results:
        if r["kind"] == "noul":
            out.append({"id": r["id"], "kind": "noul", "noul": round(float(r["value"]), 4)})
        else:
            out.append({"id": r["id"], "kind": "choice", "choice": int(r["value"]),
                        "probabilities": [round(float(p), 4) for p in r["probabilities"]],
                        "confidence": round(float(r["confidence"]), 4)})  # fmt: skip
    return out
