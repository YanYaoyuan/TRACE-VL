"""Answer-only LoRA updates on disposable forks of a detached native stream.

One episode uses one parameter version. Prefix construction is question-blind
and detached; an optimizer update closes the episode, requiring a new prefix.
This is a reader-flow primitive, not full-prefix BPTT or a learned writer.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from trace_vl.complementary_memory.state_stream import NativeStream
from trace_vl.v3.backbone import functional_linear_cache
from trace_vl.v3.positions import PositionCounter


def _cache_signature(bb):
    return (bb.text_config.to_dict(), tuple(bb.full_layers), tuple(bb.linear_layers), bb.hidden, bb.device, bb.dtype,
            bb.video_token, bb.vision_start, bb.vision_end)


class EndpointReaderEpisode:
    def __init__(self, bb, head, policy="reservoir", *, recent=8, capacity=32, seed=0, prefix_bb=None):
        if torch.is_inference_mode_enabled():
            raise ValueError("reader episodes need no_grad prefixes, not inference-mode tensors")
        self.bb = bb
        self.prefix_bb = bb if prefix_bb is None else prefix_bb
        if prefix_bb is not None:
            if prefix_bb is bb or any(p.requires_grad for p in prefix_bb.model.parameters()):
                raise ValueError("independent prefix backbone must be fully frozen")
            if any(m.training for m in prefix_bb.model.modules()):
                raise ValueError("frozen prefix backbone requires eval mode")
            if _cache_signature(prefix_bb) != _cache_signature(bb):
                raise ValueError("prefix and reader cache/token configurations must match")
            storage = {p.untyped_storage().data_ptr() for p in prefix_bb.model.parameters() if p.numel()}
            if any(p.numel() and p.untyped_storage().data_ptr() in storage for p in bb.model.parameters()):
                raise ValueError("prefix and reader parameter storage must be independent")
            self.prefix_signature = _cache_signature(prefix_bb)
        self.trainable = [p for n, p in bb.model.named_parameters() if p.requires_grad and "lora_" in n]
        other = [n for n, p in bb.model.named_parameters() if p.requires_grad and "lora_" not in n]
        if not self.trainable or other:
            raise ValueError("reader flow requires only LM LoRA parameters trainable")
        if any(m.training for m in bb.model.modules()):
            raise ValueError("deterministic reader flow requires eval mode with trainable LoRA")
        self.versions = self._versions()
        self.closed = self.sealed = False
        self.last_end = -math.inf
        self.query_time = None
        with torch.no_grad():
            self.stream = NativeStream(self.prefix_bb, head, policy, recent=recent, capacity=capacity, seed=seed)

    def _versions(self):
        models = [self.bb.model] if self.prefix_bb is self.bb else [self.bb.model, self.prefix_bb.model]
        return tuple((id(p), p._version, p.requires_grad) for model in models for p in model.parameters())

    def _check(self):
        if torch.is_inference_mode_enabled():
            raise ValueError("reader training cannot run inside inference mode")
        if self.closed:
            raise RuntimeError("episode is closed; rebuild the source prefix after an optimizer update")
        if self._versions() != self.versions:
            self.closed = True
            raise RuntimeError("model parameters changed; cached prefix is stale and must be rebuilt")
        if any(m.training for m in self.bb.model.modules()):
            raise RuntimeError("model mode changed within the source episode")
        if self.prefix_bb is not self.bb and any(m.training for m in self.prefix_bb.model.modules()):
            self.closed = True
            raise RuntimeError("prefix model mode changed within the source episode")
        if self.prefix_bb is not self.bb and (
                _cache_signature(self.prefix_bb) != self.prefix_signature
                or _cache_signature(self.bb) != self.prefix_signature):
            self.closed = True
            raise RuntimeError("prefix/reader cache configuration changed within the source episode")

    def append(self, block):
        self._check()
        if self.sealed:
            raise RuntimeError("cannot append after sealing the endpoint")
        if (block.index != self.stream.seen or not math.isfinite(block.end)
                or block.end < 0 or block.end < self.last_end):
            raise ValueError("prefix steps must have contiguous indices and finite monotone times")
        with torch.no_grad():
            self.stream.append(block)
        self.last_end = block.end

    def seal(self, query_time):
        self._check()
        if (self.sealed or not self.stream.seen or not math.isfinite(query_time)
                or query_time < self.last_end):
            raise ValueError("a nonempty causal prefix and one valid endpoint are required")
        self.query_time = float(query_time)
        self.sealed = True

    def answer_loss(self, tail_ids, answer_ids, *, max_answer_tokens=512):
        """Teacher-force answer tokens (including caller-supplied EOS) only on a fork.

        `tail_ids` must end at the assistant-generation prompt and must not
        contain gold. Question/gold are never passed to append or the live LM.
        """
        self._check()
        if not self.sealed:
            raise RuntimeError("seal the complete source prefix before asking endpoint questions")
        if (not tail_ids or not answer_ids or type(max_answer_tokens) is not int or max_answer_tokens < 1
                or len(answer_ids) > max_answer_tokens
                or any(type(i) is not int or i < 0 for i in [*tail_ids, *answer_ids])):
            raise ValueError("nonempty integer token lists within the declared answer budget are required")
        snapshot = self.stream.snapshot()
        random_state = self.stream.rng.getstate()
        cache = self.stream.fork()
        tokens = [*tail_ids, *answer_ids[:-1]]
        counter = PositionCounter(self.stream.counter.pos)
        try:
            with torch.enable_grad(), functional_linear_cache():
                hidden = self.bb.lm(self.bb.embed_ids(tokens), counter.text(len(tokens), self.bb.device), cache)
                answer_hidden = hidden[:, len(tail_ids) - 1:]
                logits = self.bb.logits(answer_hidden).float()
                targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets)
                if not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError("answer loss must be finite and retain LM LoRA autograd")
        finally:
            self.stream.verify(snapshot)
            if self.stream.rng.getstate() != random_state:
                raise RuntimeError("question changed the live retention RNG")
        return loss, {"answer_tokens": len(answer_ids), "question_tail_tokens": len(tail_ids),
                      "query_time": self.query_time, "prefix_steps": self.stream.seen,
                      "prefix_gradients": False, "live_stream_unchanged": True}

    def train_step(self, questions, optimizer, *, max_grad_norm=1., max_answer_tokens=512,
                   synchronize_gradients=None):
        """Mean-question CE, one optimizer update per complete source episode.

        Questions are (tail token IDs, gold-answer token IDs including EOS).
        Question graphs are freed one by one; prefix state/KV stay detached.
        Any attempted update closes the episode, even on failure.
        """
        self._check()
        questions = list(questions)
        if not questions or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("questions and a finite positive gradient clip are required")
        params = [p for group in optimizer.param_groups for p in group["params"]]
        if len(params) != len(self.trainable) or {id(p) for p in params} != {id(p) for p in self.trainable}:
            raise ValueError("optimizer must own exactly the episode's trainable LoRA parameters")
        optimizer.zero_grad(set_to_none=True)
        losses, tokens = [], 0
        try:
            for tail, answer in questions:
                loss, meta = self.answer_loss(tail, answer, max_answer_tokens=max_answer_tokens)
                losses.append(float(loss.detach()))
                tokens += meta["answer_tokens"]
                with torch.enable_grad():
                    (loss / len(questions)).backward()
            if synchronize_gradients is not None:
                synchronize_gradients(self.trainable)
            norm = torch.nn.utils.clip_grad_norm_(self.trainable, max_grad_norm, error_if_nonfinite=True)
            if not bool(norm > 0):
                raise RuntimeError("answer loss produced no LoRA gradient")
            self._check()
            optimizer.step()
            return {"questions": len(questions), "answer_tokens": tokens,
                    "mean_question_answer_ce": sum(losses) / len(losses),
                    "gradient_norm_before_clip": float(norm), "optimizer_steps": 1,
                    "prefix_rebuild_required": True, "prefix_gradients": False}
        finally:
            self.closed = True
