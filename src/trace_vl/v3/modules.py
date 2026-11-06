"""v3 modules: causal spatio-temporal adapter, transition encoder, alert/event heads, retention scorer.

All modules work on one step at a time (streaming) and keep their own small caches. Gates start at
zero so that, at initialization, the backbone's tokens pass through unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import CSTConfig, TransitionConfig


def grid_coords(h: int, w: int, device=None) -> Tensor:
    """Normalized (x, y) of each merged token in raster order: [h*w, 2]."""
    ys = torch.arange(h, device=device, dtype=torch.float32) / max(1, h - 1)
    xs = torch.arange(w, device=device, dtype=torch.float32) / max(1, w - 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


# ============================================================================ CST adapter
@dataclass
class CSTCache:
    """Per-layer K/V of past steps (adapter space) plus their times; also the memory bank."""

    keys: list[list[Tensor]] = field(default_factory=list)  # [layer][step] -> [H, N, dh]
    values: list[list[Tensor]] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    grid: tuple[int, int] | None = None
    memory: Tensor | None = None  # [M, d] adapter-space transition memory
    memory_times: Tensor | None = None  # [M]

    def detach(self):
        self.keys = [[k.detach() for k in layer] for layer in self.keys]
        self.values = [[v.detach() for v in layer] for layer in self.values]
        if self.memory is not None:
            self.memory = self.memory.detach()


class CSTLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_ratio: int, memory_read: bool):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        # Relative bias, separable: b = f(dx, dy) + g(log(1 + dt)). Locality and recency are learned.
        self.bias_xy = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, heads))
        self.bias_t = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, heads))
        self.memory_read = memory_read
        if memory_read:
            self.mem_norm = nn.LayerNorm(dim)
            self.mem_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.mem_time = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, dim))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_ratio * dim), nn.GELU(), nn.Linear(ffn_ratio * dim, dim))

    def _heads(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], self.heads, self.head_dim).transpose(0, 1)  # [H, N, dh]

    def forward(self, z: Tensor, coords: Tensor, t: float, past_k, past_v, past_t, memory, memory_t):
        n = z.shape[0]
        q, k, v = self.qkv(self.norm(z)).chunk(3, dim=-1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        keys = torch.cat([*past_k, k], dim=1)  # [H, (S+1)N, dh]
        values = torch.cat([*past_v, v], dim=1)
        steps = len(past_k) + 1
        dxy = coords[:, None, :] - coords[None, :, :]  # [N, N, 2]
        bias_xy = self.bias_xy(dxy).permute(2, 0, 1)  # [H, N, N]
        dts = torch.tensor([t - s for s in past_t] + [0.0], device=z.device, dtype=z.dtype).clamp_min(0)
        bias_t = self.bias_t(torch.log1p(dts)[:, None]).t()  # [H, S+1]
        bias = bias_xy.repeat(1, 1, steps) + bias_t.repeat_interleave(n, dim=1)[:, None, :]
        attended = F.scaled_dot_product_attention(q[None], keys[None], values[None], attn_mask=bias[None].to(q.dtype))
        z = z + self.proj(attended[0].transpose(0, 1).reshape(n, -1))
        if self.memory_read and memory is not None and memory.shape[0] > 0:
            age = torch.log1p((t - memory_t).clamp_min(0))[:, None].to(z.dtype)
            mem = (memory + self.mem_time(age))[None]
            read = self.mem_attn(self.mem_norm(z)[None], mem, mem, need_weights=False)[0][0]
            z = z + read
        z = z + self.ffn(self.ffn_norm(z))
        return z, k, v


class CSTAdapter(nn.Module):
    """Causal spatio-temporal adapter on merged visual tokens; output gate is zero at init."""

    def __init__(self, cfg: CSTConfig, model_dim: int):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, cfg.dim))
        self.layers = nn.ModuleList(
            [CSTLayer(cfg.dim, cfg.heads, cfg.ffn_ratio, cfg.memory_read) for _ in range(cfg.layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.dim)
        self.out = nn.Linear(cfg.dim, model_dim)
        # Zero gate (not a zero `out`): the gate still receives gradient through out(...) != 0.
        self.gate = nn.Parameter(torch.zeros(model_dim))

    def forward(self, x: Tensor, grid: tuple[int, int], t: float, cache: CSTCache) -> tuple[Tensor, Tensor]:
        """x: [N, model_dim] merged tokens of one step. Returns (adapted tokens, adapter state z)."""
        h, w = grid
        if cache.grid is not None and cache.grid != grid:
            cache.keys, cache.values, cache.times = [], [], []  # resolution change: restart the window
        cache.grid = grid
        if not cache.keys:
            cache.keys = [[] for _ in self.layers]
            cache.values = [[] for _ in self.layers]
        coords = grid_coords(h, w, x.device).to(x.dtype)
        z = self.inp(x)
        for i, layer in enumerate(self.layers):
            z, k, v = layer(z, coords, t, cache.keys[i], cache.values[i], cache.times, cache.memory, cache.memory_times)
            cache.keys[i] = (cache.keys[i] + [k])[-self.cfg.window :]
            cache.values[i] = (cache.values[i] + [v])[-self.cfg.window :]
        # The current step's K/V were appended; keep at most `window` past steps for the next call.
        cache.times = (cache.times + [t])[-self.cfg.window :]
        z = self.out_norm(z)
        return x + self.gate * self.out(z), z


# ============================================================================ transition encoder
@dataclass
class StepTokens:
    """What one step contributes to the LM."""

    tokens: Tensor  # [n, model_dim]
    rows: Tensor  # [n] merged-grid row of each token (for MRoPE)
    cols: Tensor  # [n]
    kinds: list[str]  # "frame" | "gist" | "before" | "after" | "delta"
    groups: list[int]  # transition slot id per token (-1 for frame/gist tokens)
    is_i_step: bool
    surprise: Tensor  # [N] token surprise in [0, 1]
    masses: Tensor  # [slots] surprise mass per slot
    aux: dict = field(default_factory=dict)


@dataclass
class TECache:
    prev_z: Tensor | None = None
    history: list[Tensor] = field(default_factory=list)  # past adapted LM-space tokens [N, d]
    steps_since_i: int = 10**9
    grid: tuple[int, int] | None = None
    next_slot_id: int = 0

    def detach(self):
        self.prev_z = None if self.prev_z is None else self.prev_z.detach()
        self.history = [x.detach() for x in self.history]


class SlotAttention(nn.Module):
    def __init__(self, dim: int, slots: int, iters: int):
        super().__init__()
        self.iters, self.scale = iters, dim**-0.5
        self.slots = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.norm_in, self.norm_slots = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.k, self.v, self.q = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.update = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, z: Tensor, weight: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """z [N, d], weight [N] in [0, 1], pos [N, d] positional code. Returns slots [S, d], attn [N, S]."""
        inputs = self.norm_in(z) + pos
        k, v = self.k(inputs), self.v(inputs)
        slots = self.slots
        attn = None
        for _ in range(self.iters):
            q = self.q(self.norm_slots(slots))
            attn = (k @ q.t() * self.scale).softmax(dim=-1)  # tokens compete for slots
            weights = attn * weight[:, None]
            updates = (weights / weights.sum(0, keepdim=True).clamp_min(1e-6)).t() @ v
            slots = self.update(updates, slots)
            slots = slots + self.mlp(slots)
        return slots, attn


class TransitionEncoder(nn.Module):
    """Surprise-gated latent P-frames: gist tokens plus before/after/delta tokens per active slot."""

    def __init__(self, cfg: TransitionConfig, adapter_dim: int, model_dim: int):
        super().__init__()
        self.cfg = cfg
        d = adapter_dim
        self.predictor = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.surprise_gate = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.constant_(self.surprise_gate[-1].bias, -2.0)  # start conservative: little surprise
        self.pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.slots = SlotAttention(d, cfg.slots, cfg.slot_iters)
        self.gist_queries = nn.Parameter(torch.randn(cfg.gist, d) * 0.02)
        self.gist_key = nn.Linear(d, d)
        # Transition tokens are convex combinations of real LM-space tokens (in-distribution for the LM);
        # delta adds a learned correction that starts at zero, and type embeddings start at zero.
        self.delta = nn.Sequential(nn.Linear(model_dim, d), nn.GELU(), nn.Linear(d, model_dim))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.type_embed = nn.Parameter(torch.zeros(3, model_dim))
        self.to_memory = nn.Linear(model_dim, adapter_dim)

    def forward(self, x: Tensor, z: Tensor, grid: tuple[int, int], cache: TECache, force_i: bool = False):
        """x: adapted LM-space tokens [N, D]; z: adapter state [N, d]. Returns StepTokens."""
        cfg = self.cfg
        h, w = grid
        n = x.shape[0]
        if cache.grid is not None and cache.grid != grid:
            cache.prev_z, cache.history, force_i = None, [], True
        cache.grid = grid
        coords = grid_coords(h, w, x.device).to(z.dtype)
        rows = torch.arange(h, device=x.device).repeat_interleave(w)
        cols = torch.arange(w, device=x.device).repeat(h)
        aux = {}
        if cache.prev_z is None:
            pred_err = torch.zeros(n, device=x.device, dtype=z.dtype)
            surprise = torch.ones(n, device=x.device, dtype=z.dtype)
            force_i = True
        else:
            predicted = cache.prev_z + self.predictor(cache.prev_z)
            target = z.detach()  # stop-gradient target: the adapter cannot collapse z to lower this loss
            aux["pred_loss"] = F.mse_loss(predicted, target)
            diff = z - predicted
            pred_err = diff.norm(dim=-1) / math.sqrt(z.shape[-1])
            surprise = torch.sigmoid(self.surprise_gate(torch.cat([z, diff, pred_err[:, None]], dim=-1))).squeeze(-1)
        pos = self.pos(coords)
        slots, attn = self.slots(z, surprise, pos)  # attn [N, S]
        weighted = attn * surprise[:, None]
        masses = weighted.sum(0)  # [S]
        share = weighted / weighted.sum(0, keepdim=True).clamp_min(1e-6)  # per-slot token weights [N, S]
        aux["surprise_mean"] = surprise.mean()
        aux["masses"] = masses
        is_i = force_i or cache.steps_since_i + 1 >= cfg.i_step_every or float(surprise.mean()) > cfg.scene_cut_surprise
        cache.history = (cache.history + [x])[-(cfg.before_lag + 1) :]
        cache.prev_z = z
        if is_i:
            cache.steps_since_i = 0
            return StepTokens(x, rows, cols, ["frame"] * n, [-1] * n, True, surprise, masses, aux)
        cache.steps_since_i += 1

        # Gist: G learned queries pool the step (convex combinations of real tokens).
        gist_attn = (self.gist_queries @ self.gist_key(z + pos).t() / math.sqrt(z.shape[-1])).softmax(-1)  # [G, N]
        gist = gist_attn @ x
        g_rows = (gist_attn @ rows.to(gist.dtype)).round()
        g_cols = (gist_attn @ cols.to(gist.dtype)).round()
        tokens, kinds, groups = [gist], ["gist"] * gist.shape[0], [-1] * gist.shape[0]
        t_rows, t_cols = [g_rows], [g_cols]

        # Active transition slots: mass above threshold, largest first, capped per step.
        order = masses.argsort(descending=True)
        active = [int(k) for k in order[: cfg.max_active] if float(masses[k]) >= cfg.mass_threshold]
        before_src = cache.history[0] if len(cache.history) > 1 else x
        trans_tokens = []
        for k in active:
            a = share[:, k]
            after = a @ x
            before = a @ before_src
            delta = (after - before) + self.delta(after - before)
            trio = torch.stack([before, after, delta]) + self.type_embed
            trans_tokens.append(trio)
            r, c = (a @ rows.to(a.dtype)).round(), (a @ cols.to(a.dtype)).round()
            t_rows.append(r.repeat(3))
            t_cols.append(c.repeat(3))
            kinds += ["before", "after", "delta"]
            groups += [cache.next_slot_id] * 3
            cache.next_slot_id += 1
        if trans_tokens:
            tokens.append(torch.cat(trans_tokens))
        aux["active_slots"] = len(active)
        aux["active_masses"] = [float(masses[k]) for k in active]
        aux["slot_memory"] = (
            self.to_memory(torch.stack([t[1] for t in trans_tokens])) if trans_tokens else None
        )
        return StepTokens(
            torch.cat(tokens),
            torch.cat([r.reshape(-1) for r in t_rows]).long(),
            torch.cat([c.reshape(-1) for c in t_cols]).long(),
            kinds,
            groups,
            False,
            surprise,
            masses,
            aux,
        )


# ============================================================================ heads
class AlertHead(nn.Module):
    """Rule-conditioned, non-autoregressive alert logits read at the end of every step.

    The rules are in the prompt, so the LM's state is already rule-aware; each rule's pooled prompt
    hidden state is its query. Transition ages feed duration rules ("left for > N min").
    """

    def __init__(self, model_dim: int, hidden: int = 512):
        super().__init__()
        self.state = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, hidden))
        self.rule = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, hidden))
        self.clock = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, hidden))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, step_hidden: Tensor, rules: Tensor, clock: Tensor) -> Tensor:
        """step_hidden [d] (hidden at the step's last token), rules [R, d], clock [4]. Returns [R] logits."""
        s = self.state(step_hidden)[None]
        r = self.rule(rules)
        return self.out(s * r + self.clock(clock)[None]).squeeze(-1)


class EventHead(nn.Module):
    """JEPA-style event embedding per step (cf. VL-JEPA): predicts the embedding of a description.

    Used for zero-shot rules (cosine to a rule's text embedding) and selective decoding (generate text
    only when the predicted embedding moves by more than a threshold).
    """

    def __init__(self, model_dim: int, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(),
                                 nn.Linear(model_dim, embed_dim))  # fmt: skip
        self.text = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, embed_dim))

    def forward(self, step_hidden: Tensor) -> Tensor:
        return F.normalize(self.net(step_hidden), dim=-1)

    def embed_text(self, pooled_text_hidden: Tensor) -> Tensor:
        return F.normalize(self.text(pooled_text_hidden), dim=-1)


class RetentionScorer(nn.Module):
    """Score of keeping a transition group in the full-attention KV when the budget is full."""

    FEATURES = 6

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(self.FEATURES, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @staticmethod
    def features(age: float, mass: float, drift: float, hits: float, is_active: float, is_gist: float):
        return torch.tensor([math.log1p(max(age, 0.0)), mass, drift, math.log1p(hits), is_active, is_gist])

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)
