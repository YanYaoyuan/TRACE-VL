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
from torch.utils.checkpoint import checkpoint

from .config import CSTConfig, TransitionConfig


def grid_coords(h: int, w: int, device=None) -> Tensor:
    """Normalized (x, y) of each merged token in raster order: [h*w, 2]."""
    ys = torch.arange(h, device=device, dtype=torch.float32) / max(1, h - 1)
    xs = torch.arange(w, device=device, dtype=torch.float32) / max(1, w - 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


def pool_regions(h: int, w: int, k: int) -> list[tuple[int, int, int, int]]:
    """Partition an h x w grid into exactly k non-overlapping, non-empty rectangles (row bands with balanced
    column splits), for any 1 <= k <= h * w; unsupported budgets raise instead of silently shrinking."""
    if not (isinstance(k, int) and 1 <= k <= h * w):
        raise ValueError(f"cannot pool a {h}x{w} grid into {k} regions (need 1 <= k <= {h * w})")
    bands = min(h, k, max(math.ceil(k / w), round(math.sqrt(k * h / w))))
    base, extra = divmod(k, bands)  # bands >= ceil(k / w) guarantees every band count <= w
    regions = []
    for i in range(bands):
        y0, y1 = (i * h) // bands, ((i + 1) * h) // bands
        count = base + (1 if i < extra else 0)
        for j in range(count):
            regions.append((y0, y1, (j * w) // count, ((j + 1) * w) // count))
    return regions


def pool_weights(h: int, w: int, k: int, device=None) -> tuple[Tensor, list[int], list[int]]:
    """Averaging weights [k, h*w] of the exact k-region partition, with region (row, col) centres."""
    regions = pool_regions(h, w, k)
    weights = torch.zeros(k, h * w, device=device, dtype=torch.float32)
    rows, cols = [], []
    for index, (y0, y1, x0, x1) in enumerate(regions):
        cells = [y * w + xx for y in range(y0, y1) for xx in range(x0, x1)]
        weights[index, cells] = 1.0 / len(cells)
        rows.append((y0 + y1 - 1) // 2)
        cols.append((x0 + x1 - 1) // 2)
    return weights, rows, cols


def exact_pool(x: Tensor, h: int, w: int, k: int) -> tuple[Tensor, Tensor, Tensor]:
    """Average-pool [h*w, D] grid tokens into exactly k region tokens with their (row, col) centres."""
    weights, rows, cols = pool_weights(h, w, k, x.device)
    pooled = (weights @ x.float()).to(x.dtype)
    return pooled, torch.tensor(rows, device=x.device), torch.tensor(cols, device=x.device)


@torch.no_grad()
def row_entropy(weights: Tensor) -> Tensor:
    """Normalized entropy of each row of [T, N] spatial weights (1 = uniform over the grid, 0 = one cell)."""
    w = weights.detach().float().clamp_min(0)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    n = w.shape[-1]
    return (-(w * w.clamp_min(1e-12).log()).sum(-1) / math.log(n)) if n > 1 else torch.ones(w.shape[0])


@torch.no_grad()
def token_stats(tokens: Tensor) -> dict:
    """Redundancy of emitted tokens [T, D]: mean off-diagonal cosine similarity and effective rank
    (exp entropy of the normalized singular values; T distinct directions -> up to T, identical tokens -> 1)."""
    t = tokens.detach().float()
    if t.shape[0] < 2:
        return {"cos_mean": 1.0, "effective_rank": 1.0}
    unit = F.normalize(t, dim=-1)
    sim = unit @ unit.t()
    off = sim[~torch.eye(t.shape[0], dtype=torch.bool, device=t.device)]
    sv = torch.linalg.svdvals(t)
    p = sv / sv.sum().clamp_min(1e-12)
    return {"cos_mean": float(off.mean()), "effective_rank": float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))}


@torch.no_grad()
def effective_coverage(weights: Tensor) -> float:
    """exp(entropy) / N of the mean normalized spatial weight map of a set of emitted tokens [T, N]:
    1.0 = the tokens jointly cover the grid uniformly; 1/N = everything comes from one cell (diagnostic)."""
    w = weights.detach().float().clamp_min(0)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    m = w.mean(0)
    entropy = -(m * m.clamp_min(1e-12).log()).sum()
    return float(entropy.exp() / m.numel())


def slot_part_pools(a: Tensor, x: Tensor, rows: Tensor, cols: Tensor, parts: int = 3, return_weights: bool = False):
    """Split one slot's token weights `a` [N] into `parts` groups of about equal weight (heaviest tokens first)
    and pool the real tokens of each group: `parts` convex combinations of real ViT tokens (in-distribution for
    the LM) instead of synthetic before/delta tokens. A group left empty by a dominant token falls back to the
    next-ranked single token. Returns (tokens [parts, D], rows [parts], cols [parts])."""
    order = a.argsort(descending=True)
    w = a[order]
    before = (w.cumsum(0) - w) / w.sum().clamp_min(1e-6)  # weight ranked above each token
    part = (before * parts).floor().clamp(max=parts - 1).long()
    toks, rs, cs, full = [], [], [], []
    for j in range(parts):
        idx = order[part == j]
        if idx.numel() == 0:
            idx = order[min(j, order.numel() - 1)].reshape(1)
        ww = a[idx] / a[idx].sum().clamp_min(1e-6)
        toks.append(ww @ x[idx])
        rs.append((ww @ rows[idx].to(ww.dtype)).round())
        cs.append((ww @ cols[idx].to(ww.dtype)).round())
        if return_weights:
            dense = torch.zeros_like(a)
            dense[idx] = ww
            full.append(dense)
    out = (torch.stack(toks), torch.stack(rs), torch.stack(cs))
    return (*out, torch.stack(full)) if return_weights else out


def match_norm(tokens: Tensor, ref_norm: Tensor, floor: float = 1e-3) -> Tensor:
    """Rescale tokens to the reference norm (scale detached), so synthesized tokens stay in the LM's range."""
    norms = tokens.norm(dim=-1, keepdim=True)
    scale = (ref_norm / norms.clamp_min(floor * ref_norm)).detach()
    return tokens * scale


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
        bias_xy = self.bias_xy(dxy).permute(2, 0, 1).to(q.dtype)  # [H, N, N], built in the compute dtype
        dts = torch.tensor([t - s for s in past_t] + [0.0], device=z.device, dtype=z.dtype).clamp_min(0)
        bias_t = self.bias_t(torch.log1p(dts)[:, None]).t().to(q.dtype)  # [H, S+1]
        bias = bias_xy.repeat(1, 1, steps) + bias_t.repeat_interleave(n, dim=1)[:, None, :]
        attended = F.scaled_dot_product_attention(q[None], keys[None], values[None], attn_mask=bias[None])
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
            args = (z, coords, t, cache.keys[i], cache.values[i], cache.times, cache.memory, cache.memory_times)
            if self.cfg.checkpoint and self.training and torch.is_grad_enabled():
                # Whole-clip training keeps every step's graph for the answer loss; the dense relative bias
                # [H, N, (S+1)N] alone was ~20 GB for 180 steps (PR #3, M1 OOM). Recompute it in backward.
                # Safe: cached past K/V tensors are never modified in place (lists are replaced, not edited).
                z, k, v = checkpoint(layer, *args, use_reentrant=False)
            else:
                z, k, v = layer(*args)
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
        # Transition tokens are built from real LM-space tokens and norm-matched to the step's frame tokens.
        # The delta token is anchored on the after token (delta = after + correction(after - before)):
        # a raw difference is exactly zero on static content, and zero-norm inputs make the LM's RMSNorm
        # gradients explode (~1/sqrt(eps); measured 1e12 on a static clip, PR #3 review).
        self.delta = nn.Sequential(nn.Linear(model_dim, d), nn.GELU(), nn.Linear(d, model_dim))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.type_embed = nn.Parameter(torch.zeros(3, model_dim))
        self.diagnostics = False  # extra per-token selector statistics in aux (no effect on outputs)
        self.to_memory = nn.Linear(model_dim, adapter_dim)

    def _emitted_stats(self, aux: dict, tokens: Tensor):
        if self.diagnostics:
            stats = token_stats(tokens)
            aux["diag_emitted_cos"], aux["diag_emitted_rank"] = stats["cos_mean"], stats["effective_rank"]

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
        if self.training and torch.is_grad_enabled():
            slots, attn = checkpoint(self.slots, z, surprise, pos, use_reentrant=False)  # attn [N, S]
        else:
            slots, attn = self.slots(z, surprise, pos)
        weighted = attn * surprise[:, None]
        masses = weighted.sum(0)  # [S]
        share = weighted / weighted.sum(0, keepdim=True).clamp_min(1e-6)  # per-slot token weights [N, S]
        aux["surprise_mean"] = surprise.mean()
        aux["masses"] = masses
        is_i = force_i or cache.steps_since_i + 1 >= cfg.i_step_every or float(surprise.detach().mean()) > cfg.scene_cut_surprise
        cache.history = (cache.history + [x])[-(cfg.before_lag + 1) :]
        cache.prev_z = z
        if is_i:
            cache.steps_since_i = 0
            return StepTokens(x, rows, cols, ["frame"] * n, [-1] * n, True, surprise, masses, aux)
        cache.steps_since_i += 1

        # Gist: G learned queries pool the step (convex combinations of real tokens).
        gist_attn = (self.gist_queries @ self.gist_key(z + pos).t() / math.sqrt(z.shape[-1])).softmax(-1)  # [G, N]
        ref_norm = x.detach().norm(dim=-1).mean()
        gist = match_norm(gist_attn @ x, ref_norm)
        g_rows = (gist_attn @ rows.to(gist.dtype)).round()
        g_cols = (gist_attn @ cols.to(gist.dtype)).round()
        tokens, kinds, groups = [gist], ["gist"] * gist.shape[0], [-1] * gist.shape[0]
        t_rows, t_cols = [g_rows], [g_cols]

        # Active transition slots: mass above threshold, largest first, capped per step.
        # Soft activity per slot. The hard threshold alone gives the gate no gradient once all slots are
        # below it (PR #3 A100 pilot); p_k keeps a learning signal for the budget and the task losses.
        p_active = torch.sigmoid((masses - cfg.mass_threshold) / cfg.activity_temperature)
        order = masses.argsort(descending=True)[: cfg.max_active]
        fixed = cfg.fixed_active > 0
        if fixed:  # fixed budget: top-k by mass, identical in training and inference, no gate to learn
            active = [int(k) for k in order[: cfg.fixed_active]]
        elif self.training and cfg.train_sampling:
            chosen = torch.bernoulli(p_active.detach())  # exploration: inactive-looking slots still get emitted
            active = [int(k) for k in order if chosen[k] > 0]
        else:
            active = [int(k) for k in order if float(p_active[k]) >= 0.5]  # == mass >= mass_threshold
        aux["activity_prob"] = p_active
        if fixed:  # constant: the budget loss is zero at target gist + 3 x fixed_active and has no gradient
            aux["expected_tokens"] = torch.tensor(float(cfg.gist + 3 * len(active)), device=x.device)
        else:
            aux["expected_tokens"] = cfg.gist + 3 * p_active[order].sum()  # differentiable P-step budget
        before_src = cache.history[0] if len(cache.history) > 1 else x
        trans_tokens, centroids, part_tokens, slot_weights = [], [], [], []
        for k in active:
            a = share[:, k]
            after = a @ x
            before = a @ before_src
            delta = after + self.delta(after - before)
            trio = match_norm(torch.stack([before, after, delta]) + self.type_embed, ref_norm)
            if self.training and not fixed:
                trio = trio * (1 + p_active[k] - p_active[k].detach())  # straight-through: value unchanged
            if cfg.p_step_mode == "after":
                pools, pr, pc, pw = slot_part_pools(a, x, rows, cols, return_weights=True)
                part_tokens.append((match_norm(pools + self.type_embed[1], ref_norm), pr, pc))
                slot_weights.append(pw)
            else:
                slot_weights.append(a[None].expand(3, -1))  # before/after/delta all pool the slot's region
            trans_tokens.append(trio)
            r, c = (a @ rows.to(a.dtype)).round(), (a @ cols.to(a.dtype)).round()
            centroids.append((float(r), float(c)))
            t_rows.append(r.repeat(3))
            t_cols.append(c.repeat(3))
            kinds += ["before", "after", "delta"]
            groups += [cache.next_slot_id] * 3
            cache.next_slot_id += 1
        if trans_tokens:
            tokens.append(torch.cat(trans_tokens))
        aux["active_slots"] = len(active)
        aux["max_mass"] = float(masses.max()) if masses.numel() else 0.0
        aux["active_masses"] = [float(masses[k]) for k in active]
        aux["active_centroids"] = centroids
        aux["slot_memory"] = (
            self.to_memory(torch.stack([t[1] for t in trans_tokens])) if trans_tokens else None
        )
        n_transition = sum(t.shape[0] for t in tokens)
        aux["transition_tokens"] = n_transition
        # Diagnostics (pilot-003 review): spatial coverage of what the LM sees on this P-step, and how peaked
        # the learned gist pools are (normalized entropy, 1 = uniform over the grid).
        if self.diagnostics:
            with torch.no_grad():
                logits = self.gist_queries @ self.gist_key(z + pos).t() / math.sqrt(z.shape[-1])
                ge = row_entropy(gist_attn)
                aux["diag_gist_entropy_min"] = float(ge.min())
                aux["diag_gist_logit_std"] = float(logits.float().std(-1).mean())  # spread across positions
                aux["diag_gist_token_cos"] = token_stats(gist)["cos_mean"]
                if active:
                    se = row_entropy(share[:, active].t())
                    aux["diag_slot_entropy_mean"], aux["diag_slot_entropy_min"] = float(se.mean()), float(se.min())
                    aux["diag_slot_attn_spread"] = float(attn[:, active].float().std(0).mean())
                if part_tokens:
                    pe = row_entropy(torch.cat(slot_weights))
                    aux["diag_part_entropy_mean"], aux["diag_part_entropy_min"] = float(pe.mean()), float(pe.min())
                if trans_tokens:
                    aux["diag_slot_token_cos"] = token_stats(torch.cat([t for t, _, _ in part_tokens] if part_tokens
                                                                       else trans_tokens))["cos_mean"]  # fmt: skip
        with torch.no_grad():
            g = gist_attn.detach().float()
            aux["gist_entropy"] = float((-(g * g.clamp_min(1e-12).log()).sum(-1) / math.log(n)).mean()) if n > 1 else 1.0
            aux["slot_coverage"] = effective_coverage(torch.cat(slot_weights)) if slot_weights else None
        if not cfg.emit_slots and cfg.p_step_mode in {"transition", "after"}:
            # Eval-time ablation: the trained LM sees the gist only (does it use the 12 slot tokens at all?).
            aux["coverage"] = effective_coverage(gist_attn)
            self._emitted_stats(aux, gist)
            return StepTokens(gist, g_rows.long(), g_cols.long(), ["gist"] * gist.shape[0], [-1] * gist.shape[0],
                              False, surprise, masses, aux)  # fmt: skip
        if cfg.p_step_mode == "after":
            aux["coverage"] = effective_coverage(torch.cat([gist_attn] + slot_weights))
            # After-only arm: same slots, same CST memory (from the trios above), same count (3 per slot), but
            # the LM sees real-token pools of each slot's region instead of before/after/delta.
            toks = [gist] + [t for t, _, _ in part_tokens]
            a_rows = [g_rows] + [r for _, r, _ in part_tokens]
            a_cols = [g_cols] + [c for _, _, c in part_tokens]
            a_kinds = ["gist"] * gist.shape[0] + ["after"] * (3 * len(part_tokens))
            a_groups = [-1] * gist.shape[0] + [g for g in groups[gist.shape[0] :]]
            self._emitted_stats(aux, torch.cat(toks))
            return StepTokens(torch.cat(toks), torch.cat([r.reshape(-1) for r in a_rows]).long(),
                              torch.cat([c.reshape(-1) for c in a_cols]).long(), a_kinds, a_groups, False,
                              surprise, masses, aux)  # fmt: skip
        if cfg.p_step_mode == "pooled":
            # B4: identical slot machinery (CST memory and clock inputs above are unchanged); only the
            # LM-facing P-step representation is replaced, by exactly K average-pooled frame regions.
            k = n_transition if cfg.pooled_match else cfg.pooled_tokens
            pooled, p_rows, p_cols = exact_pool(x, h, w, k)
            aux["coverage"] = effective_coverage(pool_weights(h, w, k, x.device)[0])
            self._emitted_stats(aux, pooled)
            pooled = match_norm(pooled, ref_norm)
            return StepTokens(pooled, p_rows, p_cols, ["gist"] * k, [-1] * k, False, surprise, masses, aux)
        aux["coverage"] = effective_coverage(torch.cat([gist_attn] + slot_weights))
        self._emitted_stats(aux, torch.cat(tokens))
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



# ============================================================================ transition tracks
class TransitionTracker:
    """Persistent tracks for the alert clocks: a transition emitted again at the same place with a similar
    after-token continues its track (onset kept); tracks not seen for `resolve_seconds` are resolved and
    dropped; at most `capacity` tracks are kept (least recently seen dropped first)."""

    def __init__(self, resolve_seconds: float = 12.0, capacity: int = 64, max_cells: float = 2.0, min_cos: float = 0.8):
        self.resolve_seconds, self.capacity, self.max_cells, self.min_cos = resolve_seconds, capacity, max_cells, min_cos
        self.tracks: list[dict] = []  # {"onset", "last", "row", "col", "vec"}

    def update(self, t: float, centroids: list[tuple[float, float]], vectors: Tensor | None) -> None:
        # Expire first: after an observation gap (dropped frames, sparse offline sampling) a stale track
        # must not be revived with its old onset (PR #3 review: age 100 s after a 100 s gap).
        self.tracks = [k for k in self.tracks if t - k["last"] <= self.resolve_seconds]
        if vectors is not None:
            vectors = F.normalize(vectors.detach().float(), dim=-1)
            for (row, col), vec in zip(centroids, vectors):
                best, best_cos = None, self.min_cos
                for track in self.tracks:
                    if math.hypot(track["row"] - row, track["col"] - col) > self.max_cells:
                        continue
                    cos = float(track["vec"] @ vec)
                    if cos >= best_cos:
                        best, best_cos = track, cos
                if best is None:
                    self.tracks.append({"onset": t, "last": t, "row": row, "col": col, "vec": vec})
                else:
                    best.update(last=t, row=row, col=col, vec=vec)
        self.tracks = [k for k in self.tracks if t - k["last"] <= self.resolve_seconds]
        if len(self.tracks) > self.capacity:
            self.tracks = sorted(self.tracks, key=lambda k: k["last"])[-self.capacity :]

    def clock(self, t: float, surprise_mean: float) -> Tensor:
        """[log1p max age, log1p mean age, log1p count, mean surprise] for the alert head."""
        ages = [t - k["onset"] for k in self.tracks]
        if not ages:
            return torch.tensor([0.0, 0.0, 0.0, surprise_mean])
        return torch.tensor([math.log1p(max(ages)), math.log1p(sum(ages) / len(ages)), math.log1p(len(ages)), surprise_mean])


# ============================================================================ heads
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
