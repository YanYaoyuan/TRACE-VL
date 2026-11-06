"""Incremental streaming session over the hybrid backbone (inference path).

Layout of the LM sequence (one user turn, as in Qwen's chat template):
    <head> [Monitoring rules: ...] (prompt, protected)
    per step: "<12.5 seconds>" <|vision_start|> visual tokens <|vision_end|>
    at a question (on a forked cache): "\nQuestion ... Answer with the option letter only." <tail>
`mode="native"` sends all 196 merged tokens per step (baseline); `mode="v3"` routes the step through
the CST adapter and the transition encoder (I-steps keep all tokens; P-steps send gist + transitions).
"""

from __future__ import annotations

import copy
import math
import time as clock
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from .backbone import Backbone
from .config import V3Config
from .kv import KVMemory
from .model import TraceV3Modules
from .modules import CSTCache, RetentionScorer, TECache
from .positions import PositionCounter

LETTERS = "ABCDEFGHIJ"


@dataclass
class StepResult:
    t: float
    tokens: int
    is_i_step: bool
    kv_tokens: int
    seconds_vision: float
    seconds_lm: float
    drift: float
    alert_logits: list[float] = field(default_factory=list)
    event: torch.Tensor | None = None
    active_slots: int = 0


class StreamSession:
    def __init__(self, backbone: Backbone, cfg: V3Config, modules: TraceV3Modules | None = None,
                 mode: str = "native", rules: list[str] | None = None):  # fmt: skip
        if mode not in {"native", "v3"}:
            raise ValueError("mode must be native or v3")
        if mode == "v3" and modules is None:
            raise ValueError("v3 mode needs TraceV3Modules")
        self.bb, self.cfg, self.modules, self.mode = backbone, cfg, modules, mode
        self.cache = backbone.new_cache()
        self.pos = PositionCounter()
        self.memory = KVMemory(cfg.memory)
        self.cst_cache, self.te_cache = CSTCache(), TECache()
        self.step_index = 0
        self.active: dict[int, float] = {}  # transition slot id -> onset time (for duration clocks)
        self.letter_ids = [backbone.ids(letter)[0] for letter in LETTERS]
        self.rules = rules or []
        self.rule_hidden = None
        self._prefill_prompt()

    # ------------------------------------------------------------------ prompt
    @torch.no_grad()
    def _prefill_prompt(self):
        pieces = [self.bb.head_text]
        spans = []
        if self.rules:
            pieces.append("Monitoring rules:\n")
            for i, rule in enumerate(self.rules):
                pieces.append(f"{i + 1}. ")
                spans.append(len(pieces))
                pieces.append(rule)
                pieces.append("\n")
            pieces.append("Video stream:\n")
        ids, starts = [], []
        for i, piece in enumerate(pieces):
            if i in spans:
                starts.append((len(ids), len(ids) + len(self.bb.ids(piece))))
            ids += self.bb.ids(piece)
        embeds = self.bb.embed_ids(ids)
        hidden = self.bb.lm(embeds, self.pos.text(len(ids)), self.cache)
        self.memory.append(len(ids), "prompt", -1.0, ("prompt",), protected=True)
        if spans:
            self.rule_hidden = torch.stack([hidden[0, a:b].mean(0) for a, b in starts])

    # ------------------------------------------------------------------ steps
    def _resize(self, frames: np.ndarray) -> np.ndarray:
        return self.bb.resize_step(frames)

    @torch.no_grad()
    def step(self, frames: np.ndarray, t: float) -> StepResult:
        """Consume one step: 2 RGB uint8 frames [2, H, W, 3] centred at stream time t (seconds)."""
        bb = self.bb
        started = clock.perf_counter()
        pixel_values, grid_thw = bb.preprocess(self._resize(frames))
        x = bb.encode(pixel_values, grid_thw)
        h, w = bb.merged_grid(grid_thw)
        info = None
        if self.mode == "v3":
            dtype = next(self.modules.parameters()).dtype
            adapted, z = self.modules.cst(x.to(dtype), (h, w), t, self.cst_cache)
            info = self.modules.transition(adapted, z, (h, w), self.te_cache)
            tokens, rows, cols = info.tokens.to(bb.dtype), info.rows, info.cols
            kinds = info.kinds
            groups = [("step", self.step_index) if g < 0 else ("slot", g) for g in info.groups]
            self._update_cst_memory(info, t)
        else:
            tokens = x
            rows = torch.arange(h, device=x.device).repeat_interleave(w)
            cols = torch.arange(w, device=x.device).repeat(h)
            kinds, groups = ["frame"] * x.shape[0], [("step", self.step_index)] * x.shape[0]
        seconds_vision = clock.perf_counter() - started

        ts_ids = bb.ids(f"<{t:.1f} seconds>") + [bb.vision_start]
        embeds = torch.cat([bb.embed_ids(ts_ids), tokens[None].to(bb.dtype), bb.embed_ids([bb.vision_end])], dim=1)
        positions = torch.cat(
            [self.pos.text(len(ts_ids), x.device), self.pos.points(rows, cols, h, w), self.pos.text(1, x.device)],
            dim=1,
        )
        before = bb.recurrent_state(self.cache)
        before = None if before is None else before.clone()
        started = clock.perf_counter()
        hidden = bb.lm(embeds, positions, self.cache)
        seconds_lm = clock.perf_counter() - started
        after = bb.recurrent_state(self.cache)
        drift = 0.0
        if before is not None and after is not None:
            drift = float((after.float() - before.float()).norm() / before.float().norm().clamp_min(1e-6))

        n_text = len(ts_ids)
        self.memory.append(n_text, "text", t, ("step", self.step_index))
        self.memory.append(tokens.shape[0], kinds, t, groups)
        self.memory.append(1, "text", t, ("step", self.step_index))
        self._score_groups(info, t, drift)
        mask = self.memory.keep_mask(t)
        if mask is not None:
            bb.evict(self.cache, mask)
            self.memory.apply(mask)

        result = StepResult(t, tokens.shape[0], info.is_i_step if info else True, len(self.memory),
                            seconds_vision, seconds_lm, drift)  # fmt: skip
        if self.mode == "v3":
            result.active_slots = int(info.aux.get("active_slots", 0))
            last = hidden[0, -1].to(next(self.modules.parameters()).dtype)
            result.event = self.modules.event(last)
            if self.rule_hidden is not None:
                logits = self.modules.alert(last, self.rule_hidden.to(last.dtype), self._clock(info, t).to(last))
                result.alert_logits = logits.float().tolist()
        self.step_index += 1
        return result

    def _update_cst_memory(self, info, t: float):
        slot_memory = info.aux.get("slot_memory")
        if slot_memory is None:
            return
        cache, cap = self.cst_cache, self.cfg.transition.memory_slots
        times = torch.full((slot_memory.shape[0],), t, device=slot_memory.device, dtype=slot_memory.dtype)
        cache.memory = slot_memory if cache.memory is None else torch.cat([cache.memory, slot_memory])[-cap:]
        cache.memory_times = times if cache.memory_times is None else torch.cat([cache.memory_times, times])[-cap:]
        for g in sorted({g for g in info.groups if g >= 0}):
            self.active[g] = t

    def _clock(self, info, t: float) -> torch.Tensor:
        ages = [t - onset for onset in self.active.values()]
        surprise = float(info.surprise.mean()) if info is not None else 0.0
        if not ages:
            return torch.tensor([0.0, 0.0, 0.0, surprise])
        return torch.tensor([math.log1p(max(ages)), math.log1p(sum(ages) / len(ages)), math.log1p(len(ages)), surprise])

    def _score_groups(self, info, t: float, drift: float):
        policy = self.cfg.memory.policy
        step_group = ("step", self.step_index)
        if policy == "drift":
            self.memory.scores[step_group] = drift
        elif policy == "tkm":
            # Learned retention when trained; before that, surprise mass (transitions) beats gist/frames.
            self.memory.scores[step_group] = -1.0
            if info is not None:
                scorer: RetentionScorer = self.modules.retention
                dtype = next(scorer.parameters()).dtype
                slot_ids = list(dict.fromkeys(g for g in info.groups if g >= 0))
                for g, mass in zip(slot_ids, info.aux.get("active_masses", [])):
                    feats = RetentionScorer.features(0.0, mass, drift, 0.0, 1.0, 0.0)
                    self.memory.scores[("slot", g)] = float(scorer(feats.to(info.tokens.device, dtype)))

    # ------------------------------------------------------------------ questions
    def _fork(self):
        return self.bb.fork(self.cache), copy.deepcopy(self.pos)

    @torch.no_grad()
    def _marker_step(self, cache, pos, frames: np.ndarray, t: float):
        """Append a full-token query image (e.g. the current frame with a red box) to a forked cache."""
        bb = self.bb
        pixel_values, grid_thw = bb.preprocess(self._resize(frames))
        x = bb.encode(pixel_values, grid_thw)
        h, w = bb.merged_grid(grid_thw)
        ts_ids = bb.ids(f"<{t:.1f} seconds>") + [bb.vision_start]
        embeds = torch.cat([bb.embed_ids(ts_ids), x[None].to(bb.dtype), bb.embed_ids([bb.vision_end])], dim=1)
        positions = torch.cat([pos.text(len(ts_ids), x.device), pos.grid(h, w, x.device), pos.text(1, x.device)], 1)
        bb.lm(embeds, positions, cache)

    @torch.no_grad()
    def ask(self, question: str, options: list[str], marker: tuple[np.ndarray, float] | None = None):
        """Score options by the next-token log-probability of their letters, on a forked cache.

        `marker=(frames, t)` adds a query image (current frame with the question's region drawn) on the
        fork only, so it never enters the stream memory.
        """
        cache, pos = self._fork()
        if marker is not None:
            self._marker_step(cache, pos, *marker)
            question = "The red box in the last image marks the area the question is about.\n" + question
        lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
        text = "\n" + question + "\n" + "\n".join(lines) + "\nAnswer with the option letter only." + self.bb.tail_text
        ids = self.bb.ids(text)
        hidden = self.bb.lm(self.bb.embed_ids(ids), pos.text(len(ids)), cache)
        logits = self.bb.logits(hidden[:, -1:])[0, -1].float()
        scores = logits.log_softmax(-1)[self.letter_ids[: len(options)]]
        return int(scores.argmax()), scores.tolist()

    @torch.no_grad()
    def describe(self, prompt: str = "Describe what is happening now in one sentence.", max_new_tokens: int = 40):
        """Greedy text on a forked cache (used by selective decoding; not needed for alerts)."""
        cache, pos = self._fork()
        ids = self.bb.ids("\n" + prompt + self.bb.tail_text)
        hidden = self.bb.lm(self.bb.embed_ids(ids), pos.text(len(ids)), cache)
        out = []
        for _ in range(max_new_tokens):
            token = int(self.bb.logits(hidden[:, -1:])[0, -1].argmax())
            if token == self.bb.tokenizer.eos_token_id or token in self.bb.tokenizer.all_special_ids:
                break
            out.append(token)
            hidden = self.bb.lm(self.bb.embed_ids([token]), pos.text(1), cache)
        return self.bb.tokenizer.decode(out, skip_special_tokens=True).strip()

    def stats(self) -> dict:
        return {
            "steps": self.step_index,
            "kv_tokens": len(self.memory),
            "kv_bytes": self.bb.kv_bytes(self.cache),
            "state_bytes": self.bb.state_bytes(self.cache),
            "evicted_tokens": self.memory.evicted_tokens,
            "kv_kinds": self.memory.counts(),
            "position": self.pos.state(),
        }


def selective_decode(previous: torch.Tensor | None, current: torch.Tensor, threshold: float = 0.15) -> bool:
    """VL-JEPA-style trigger: decode text only when the event embedding moved enough."""
    if previous is None:
        return True
    return float(1 - F.cosine_similarity(previous, current, dim=-1)) > threshold
