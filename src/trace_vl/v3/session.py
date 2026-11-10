"""Incremental streaming session over the hybrid backbone (inference path).

Layout of the LM sequence = Qwen3.5's public processor protocol for one video in a user turn
(`processor.apply_chat_template` with a video item; verified token-for-token by `trace-vl v3 check`):
    <head> <|vision_start|>                                   (prompt + OUTER video start, protected)
    per step: "<12.5 seconds>" <|vision_start|> visual tokens <|vision_end|>
    at a question, on a forked cache only: <|vision_end|> (OUTER video end) "\nQuestion ..." <tail>
The live stream never closes the outer video, so streaming continues after every question.
`mode="native"` sends all 196 merged tokens per step (baseline); `mode="v3"` routes the step through
the CST adapter and the transition encoder (I-steps keep all tokens; P-steps send gist + transitions).
Registered alert questions (Jev-style classification: noul / choice) are answered every step by the
alert head, latently connected to the backbone (tapped layers + Gated DeltaNet states), without decoding.
"""

from __future__ import annotations

import copy
import time as clock
from dataclasses import dataclass, field

import numpy as np
import torch

from .alert import DecisionMemory, Question, encode_questions, public
from .backbone import Backbone
from .config import V3Config
from .kv import KVMemory, RetentionBook, score_retention
from .model import TraceV3Modules, stream_step
from .modules import CSTCache, TECache, TransitionTracker
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
    alerts: list[dict] = field(default_factory=list)  # answers to registered alert questions
    seconds_alert: float = 0.0
    active_slots: int = 0


class StreamSession:
    def __init__(self, backbone: Backbone, cfg: V3Config, modules: TraceV3Modules | None = None,
                 mode: str = "native", questions: list[Question] | None = None, trace: bool = False):  # fmt: skip
        if mode not in {"native", "v3"}:
            raise ValueError("mode must be native or v3")
        if mode == "v3" and modules is None:
            raise ValueError("v3 mode needs TraceV3Modules")
        if questions and modules is None:
            raise ValueError("alert questions need TraceV3Modules (alert head)")
        self.bb, self.cfg, self.modules, self.mode = backbone, cfg, modules, mode
        self.cache = backbone.new_cache()
        self.pos = PositionCounter()
        self.memory = KVMemory(cfg.memory)
        self.book = RetentionBook()
        self.cst_cache, self.te_cache = CSTCache(), TECache()
        self.step_index = 0
        self.tracker = TransitionTracker()  # persistent transition tracks for the alert clocks (bounded)
        self.letter_ids = [backbone.ids(letter)[0] for letter in LETTERS]
        self.decision_memory = DecisionMemory(cfg.alert.memory_tokens)
        self.watching = encode_questions(backbone, questions) if questions else None
        self.last_context = None  # (taps, xy, t, clock) of the latest step, for decide()
        self.last_info = None
        self.evict_steps: set | None = None  # explicit eviction steps (Stage-2 training schedule); None = every N
        self.audit = False  # memory audit: counterfactual FIFO overlap and score correlations at each eviction
        self.audit_stats: dict[str, list[float]] = {"jaccard_vs_fifo": [], "score_recency_rho": [], "score_mass_rho": []}
        self.eviction_audit: dict = {"events": 0, "last_t": None, "last": None, "older_max": 0, "cap_violations": 0}
        self.trace_ids = [] if trace else None  # token ids as the processor would write them (checks)
        self.trace_query = None
        self._prefill_prompt()

    # ------------------------------------------------------------------ prompt
    @torch.no_grad()
    def _prefill_prompt(self):
        # The public processor wraps the whole video in an outer <|vision_start|> ... <|vision_end|> pair
        # around the per-patch blocks (PR #3 official-processor control).
        ids = self.bb.ids(self.bb.head_text) + [self.bb.vision_start]
        self.bb.lm(self.bb.embed_ids(ids), self.pos.text(len(ids)), self.cache)
        self.memory.append(len(ids), "prompt", -1.0, ("prompt",), protected=True)
        if self.trace_ids is not None:
            self.trace_ids += ids

    # ------------------------------------------------------------------ steps
    def _resize(self, frames: np.ndarray) -> np.ndarray:
        return self.bb.resize_step(frames)

    @torch.no_grad()
    def step(self, frames: np.ndarray | None, t: float, pixels=None) -> StepResult:
        """Consume one step: 2 RGB uint8 frames [2, H, W, 3] centred at stream time t (seconds).

        `pixels=(pixel_values, grid_thw)` feeds already-processed patches of one temporal patch instead
        (e.g. the processor's own output, for protocol checks).
        """
        bb = self.bb
        started = clock.perf_counter()
        pixel_values, grid_thw = pixels if pixels is not None else bb.preprocess(self._resize(frames))
        x = bb.encode(pixel_values, grid_thw)
        h, w = bb.merged_grid(grid_thw)
        info = None
        if self.mode == "v3":
            _, _, info = stream_step(self.modules, x, (h, w), t, self.cst_cache, self.te_cache)
            self.last_info = info  # latest StepTokens (demos read its weight map)
            tokens, rows, cols = info.tokens.to(bb.dtype), info.rows, info.cols
            kinds = info.kinds
            groups = [("step", self.step_index) if g < 0 else ("slot", g) for g in info.groups]
            self.tracker.update(t, info.aux.get("active_centroids", []), info.aux.get("slot_memory"))
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
        taps = self.cfg.alert.taps if self.modules is not None else ()
        hidden, tapped = bb.lm_taps(embeds, positions, self.cache, taps=taps)
        seconds_lm = clock.perf_counter() - started
        after = bb.recurrent_state(self.cache)
        drift = 0.0
        if before is not None and after is not None:
            drift = float((after.float() - before.float()).norm() / before.float().norm().clamp_min(1e-6))

        n_text = len(ts_ids)
        if self.trace_ids is not None:
            self.trace_ids += ts_ids + [bb.video_token] * tokens.shape[0] + [bb.vision_end]
        self.memory.append(n_text, "text", t, ("step", self.step_index))
        self.memory.append(tokens.shape[0], kinds, t, groups)
        self.memory.append(1, "text", t, ("step", self.step_index))
        self._score_groups(info, t, drift)
        if self.evict_steps is not None:
            evict_now = self.step_index in self.evict_steps
        else:
            evict_now = (self.step_index + 1) % self.cfg.memory.evict_every == 0
        mask = self.memory.keep_mask(t) if evict_now else None
        if self.audit and mask is not None and self.cfg.memory.policy in {"tkm", "mass"}:
            self._audit_eviction(t, mask)
        if mask is not None:
            bb.evict(self.cache, mask)
            self.memory.apply(mask)
            self.book.prune(set(self.memory.group))
            if self.audit:
                self._audit_cap(t)

        result = StepResult(t, tokens.shape[0], info.is_i_step if info else True, len(self.memory),
                            seconds_vision, seconds_lm, drift)  # fmt: skip
        if info is not None:
            result.active_slots = int(info.aux.get("active_slots", 0))
        if self.modules is not None:
            n_text = len(ts_ids)
            span = slice(n_text, n_text + tokens.shape[0])
            xy = torch.stack([cols.float() / max(1, w - 1), rows.float() / max(1, h - 1)], dim=-1)
            step_taps = tapped[:, 0, span]
            clock_feats = self._clock(info, t)
            self.last_context = (step_taps, xy, t, clock_feats)
            if self.watching is not None:
                started = clock.perf_counter()
                result.alerts = public(self._decide(self.watching))
                result.seconds_alert = clock.perf_counter() - started
            # Latent memory for later alerts: the step end plus this step's transition/gist tokens.
            keep = [i for i, kind in enumerate(kinds) if kind in {"gist", "after", "delta"}]
            latent = [hidden[0, -1:]] + ([hidden[0, span][keep]] if keep else [])
            self.decision_memory.add(torch.cat(latent), t)
        self.step_index += 1
        return result

    def _audit_eviction(self, t: float, mask: torch.Tensor):
        """Same memory state: which older tokens this policy keeps vs FIFO (Jaccard), and whether its group scores
        track recency (newest time) or surprise mass (Spearman rank correlations over the older groups)."""
        memory = self.memory
        recent_from = t - self.cfg.memory.window_seconds
        older = [i for i, (p, tt) in enumerate(zip(memory.protected, memory.time)) if not p and tt < recent_from]
        fifo = memory.keep_mask(t, policy="fifo")
        kept = {i for i in older if bool(mask[i])}
        kept_fifo = {i for i in older if fifo is None or bool(fifo[i])}
        union = kept | kept_fifo
        if union:
            self.audit_stats["jaccard_vs_fifo"].append(len(kept & kept_fifo) / len(union))
        groups = sorted({memory.group[i] for i in older if memory.group[i] in memory.scores})
        if len(groups) >= 3:
            newest = {g: max(memory.time[i] for i in older if memory.group[i] == g) for g in groups}
            scores = [memory.scores[g] for g in groups]
            self.audit_stats["score_recency_rho"].append(spearman(scores, [newest[g] for g in groups]))
            masses = [self.book.groups[g].mass if g in self.book.groups else 0.0 for g in groups]
            self.audit_stats["score_mass_rho"].append(spearman(scores, masses))

    def _audit_cap(self, t: float):
        """Older-memory cap at the eviction itself: the KV split right after the mask is applied, classified at the
        eviction (consumed step) time t. This is the clock the budget is enforced on."""
        acc = self.memory.accounting(t)
        ev = self.eviction_audit
        ev["events"] += 1
        ev["last_t"], ev["last"] = t, acc
        ev["older_max"] = max(ev["older_max"], acc["older"])
        if self.cfg.memory.policy != "full" and acc["older"] > self.cfg.memory.budget_tokens:
            ev["cap_violations"] += 1

    def audit_summary(self, t_question: float, t_step: float | None = None) -> dict:
        """Per-question audit row. Three clocks, labelled separately:
        - kv_evict_*: KV split right after the most recent eviction, ages at that eviction time (audit_evict_t);
          the older-memory cap applies here (audit_older_max_at_eviction, audit_older_cap_violations over all
          evictions so far).
        - kv_step_*: the current memory, ages at the last consumed step time (t_step).
        - kv_qtime_*: the current memory, ages at the question time; tokens that left the recent window after the
          last eviction count as older here, so kv_qtime_older can exceed the budget without any cap violation.
        audit_*_mean are cumulative session means over the eviction events up to this question (not per-event
        values); audit_*_sum / audit_*_n allow event-weighted pooling across streams."""
        out = {f"kv_qtime_{k}": v for k, v in self.memory.accounting(t_question).items()}
        if t_step is not None:
            out["audit_step_t"] = t_step
            out.update({f"kv_step_{k}": v for k, v in self.memory.accounting(t_step).items()})
        ev = self.eviction_audit
        out["audit_evict_t"] = ev["last_t"]
        out.update({f"kv_evict_{k}": v for k, v in (ev["last"] or {}).items()})
        out["audit_eviction_events"] = ev["events"]
        out["audit_older_max_at_eviction"] = ev["older_max"] if ev["events"] else None
        out["audit_older_cap_violations"] = ev["cap_violations"]
        for key, values in self.audit_stats.items():
            clean = [v for v in values if v == v]
            out[f"audit_{key}_mean"] = round(sum(clean) / len(clean), 4) if clean else None
            out[f"audit_{key}_sum"] = round(sum(clean), 6)
            out[f"audit_{key}_n"] = len(clean)
        out["audit_evictions"] = len(self.audit_stats["jaccard_vs_fifo"])
        out["audit_scope"] = "cumulative-session-at-question"
        return out

    def _clock(self, info, t: float) -> torch.Tensor:
        surprise = float(info.surprise.mean()) if info is not None else 0.0
        return self.tracker.clock(t, surprise)

    def _score_groups(self, info, t: float, drift: float):
        policy = self.cfg.memory.policy
        step_group = ("step", self.step_index)
        if policy == "drift":
            self.memory.scores[step_group] = drift
        elif policy in {"tkm", "mass"}:
            # Same bookkeeping as Stage 2 (train_stage2.observe_step): every older group, transition slots and
            # whole steps (gist / pooled / I-step) alike, is rescored with its current age before eviction.
            observe_step(self.book, info, t, self.step_index)
            scorer = self.modules.retention if self.modules is not None else None
            score_retention(self.book, self.memory, scorer, t)

    # ------------------------------------------------------------------ alerts (Jev-style classification)
    def _decide(self, encoded):
        if self.last_context is None:
            raise ValueError("no step observed yet")
        step_taps, xy, t, clock_feats = self.last_context
        states = self.bb.gdn_states(self.cache, self.cfg.alert.state_layers)
        return self.modules.alert(encoded, step_taps, xy, self.decision_memory, t, clock_feats, states)

    @torch.no_grad()
    def decide(self, questions: list[Question]) -> list[dict]:
        """Classify the stream so far for ad-hoc alert questions, in one pass, without decoding."""
        return public(self._decide(encode_questions(self.bb, questions)))

    # ------------------------------------------------------------------ questions (generative LM)
    def _fork(self):
        return self.bb.fork(self.cache), copy.deepcopy(self.pos)

    def _close_video(self, cache, pos) -> list[int]:
        """Close the outer video on a fork (never on the live stream)."""
        ids = [self.bb.vision_end]
        self.bb.lm(self.bb.embed_ids(ids), pos.text(1, self.bb.device), cache)
        return ids

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
        if marker is not None:  # a frame of the video: inside the outer wrapper, before it is closed
            self._marker_step(cache, pos, *marker)
            question = "The red box in the last image marks the area the question is about.\n" + question
        closing = self._close_video(cache, pos)
        lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
        text = "\n" + question + "\n" + "\n".join(lines) + "\nAnswer with the option letter only." + self.bb.tail_text
        ids = self.bb.ids(text)
        if self.trace_ids is not None:
            self.trace_query = closing + ids
        hidden = self.bb.lm(self.bb.embed_ids(ids), pos.text(len(ids)), cache)
        scores = self.bb.letter_logprobs(hidden[0, -1], self.letter_ids[: len(options)])
        return int(scores.argmax()), scores.tolist()

    @torch.no_grad()
    def ask_yesno(self, prompt: str) -> tuple[str, list[float]]:
        """Free-form yes/no prompt (e.g. OVO-Bench SSR/CRR): compare the fp32 log-probs of "Yes" and "No" as the
        first answer token on a forked cache."""
        cache, pos = self._fork()
        self._close_video(cache, pos)
        ids = self.bb.ids("\n" + prompt + self.bb.tail_text)
        hidden = self.bb.lm(self.bb.embed_ids(ids), pos.text(len(ids)), cache)
        candidates = [self.bb.ids("Yes")[0], self.bb.ids("No")[0]]
        scores = self.bb.letter_logprobs(hidden[0, -1], candidates)
        return ("Yes" if scores[0] >= scores[1] else "No"), scores.tolist()

    @torch.no_grad()
    def describe(self, prompt: str = "Describe what is happening now in one sentence.", max_new_tokens: int = 40):
        """Greedy text on a forked cache (explanations on demand; alerts do not need it)."""
        cache, pos = self._fork()
        self._close_video(cache, pos)
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
            "transition_tracks": len(self.tracker.tracks),
        }


def observe_step(book: RetentionBook, info, t: float, step_index: int):
    """Register this step's KV groups: ("step", k) with the step's mean surprise, ("slot", id) with its mass."""
    surprise = float(info.surprise.mean()) if info is not None else 0.0
    book.observe(("step", step_index), t, surprise, is_slot=False)
    if info is None:
        return
    slot_ids = list(dict.fromkeys(g for g in info.groups if g >= 0))
    for g, mass in zip(slot_ids, info.aux.get("active_masses", [])):
        book.observe(("slot", g), t, float(mass), is_slot=True)


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties (NaN when either side is constant)."""

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        r = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / (va * vb) ** 0.5 if va > 0 and vb > 0 else float("nan")

