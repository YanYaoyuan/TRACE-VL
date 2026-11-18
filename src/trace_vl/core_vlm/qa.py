"""Video-to-language answer loss for controlled core-VLM experiments.

This uses TRACE's explicit visual-embedding protocol, not Qwen's official native
video processor/chat path. Full and grid modes share observations and prompts.
There is no alert head, retention scorer, or recurrent LM cache. An optional
frozen original teacher anchors answer distributions in the preservation pilot. A trained LM adapter is part of the VLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from trace_vl.v3.positions import PositionCounter


@dataclass
class VisualStep:
    tokens: torch.Tensor  # [merged grid cells, LM hidden size]
    grid: tuple[int, int]
    end: float
    frame_times: tuple[float, float]  # two actual decoded frame timestamps


def validate_steps(steps: list[VisualStep], query_time: float, hidden: int):
    if not steps or not math.isfinite(query_time) or query_time < 0:
        raise ValueError("nonempty visual steps and a finite query time are required")
    previous_end = -1.0
    for step in steps:
        h, w = step.grid
        if type(h) is not int or type(w) is not int or min(h, w) < 1:
            raise ValueError("invalid merged grid")
        if step.tokens.ndim != 2 or tuple(step.tokens.shape) != (h * w, hidden):
            raise ValueError("visual tokens do not match grid/LM hidden size")
        if not math.isfinite(step.end) or not previous_end < step.end <= query_time:
            raise ValueError("steps must be strictly chronological and end before the query")
        times = step.frame_times
        if len(times) != 2 or any(not math.isfinite(t) or not 0 <= t <= step.end for t in times):
            raise ValueError("step must declare two causal decoded frame times")
        if times[0] > times[1]:
            raise ValueError("decoded frames must be chronological")
        previous_end = step.end


def grid_pool(tokens: torch.Tensor, grid: tuple[int, int], target: tuple[int, int]):
    """Nonoverlapping spatial bins; keep integer centres in the original RoPE grid.

Unlike adaptive pooling, adjacent bins never reuse patches when dimensions are
not divisible. No feature renormalization: the baseline is plain region means.
"""
    h, w = grid
    rows, cols = target
    if type(rows) is not int or type(cols) is not int or not (1 <= rows <= h and 1 <= cols <= w):
        raise ValueError("pool grid must fit inside the merged visual grid")
    image = tokens.reshape(h, w, -1)
    parts, rr, cc = [], [], []
    for r in range(rows):
        lo_r, hi_r = r * h // rows, (r + 1) * h // rows
        for c in range(cols):
            lo_c, hi_c = c * w // cols, (c + 1) * w // cols
            parts.append(image[lo_r:hi_r, lo_c:hi_c].mean((0, 1)))
            rr.append((lo_r + hi_r - 1) // 2)
            cc.append((lo_c + hi_c - 1) // 2)
    return (torch.stack(parts), torch.tensor(rr, device=tokens.device),
            torch.tensor(cc, device=tokens.device))


def question_and_answer(record: dict) -> tuple[str, str]:
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("a video understanding question is required")
    if "options" in record:
        options, gold = record["options"], record.get("answer")
        if (not isinstance(options, list) or not 2 <= len(options) <= 26
                or any(not isinstance(o, str) or not o.strip() for o in options)
                or type(gold) is not int or not 0 <= gold < len(options)):
            raise ValueError("invalid MC options/answer")
        text = question + "\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
        return text + "\nAnswer with the option letter only.", chr(65 + gold)
    answer = record.get("answer_text")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("free-text supervision requires answer_text")
    return question, answer


class VideoQALoss(nn.Module):
    """Register the LM (including its LoRA) and train on answer tokens only.

    The supplied Backbone owns preprocessing and frozen vision. Visual features
    may be computed with no_grad; language-model execution MUST keep autograd.
    A later architecture arm must transform these features before this same loss,
    with verified gradients from the answer loss to its added parameters.
    """

    def __init__(self, backbone, *, mode: str = "full", pool_grid: tuple[int, int] = (4, 5),
                 answer_end_token_id: int | None = None):
        super().__init__()
        if mode not in ("full", "grid"):
            raise ValueError("mode must be full or grid")
        self.bb = backbone
        self.lm_model = backbone.model
        self.mode, self.pool_grid = mode, pool_grid
        self.answer_end_token_id = answer_end_token_id

    def prefix(self, steps: list[VisualStep], record: dict):
        bb = self.bb
        validate_steps(steps, record["query_time"], bb.hidden)
        question, answer = question_and_answer(record)
        counter = PositionCounter()
        start = bb.ids(bb.head_text) + [bb.vision_start]
        parts, positions = [bb.embed_ids(start)], [counter.text(len(start), bb.device)]
        counts = []
        for step in steps:
            h, w = step.grid
            x = step.tokens.to(device=bb.device, dtype=bb.dtype)
            if self.mode == "grid":
                x, rows, cols = grid_pool(x, step.grid, self.pool_grid)
            else:
                rows = torch.arange(h, device=x.device).repeat_interleave(w)
                cols = torch.arange(w, device=x.device).repeat(h)
            timestamp = bb.ids(f"<{step.end:.1f} seconds>") + [bb.vision_start]
            parts.extend([bb.embed_ids(timestamp), x[None], bb.embed_ids([bb.vision_end])])
            positions.extend([counter.text(len(timestamp), bb.device),
                              counter.points(rows, cols, h, w), counter.text(1, bb.device)])
            counts.append(len(x))
        tail = [bb.vision_end] + bb.ids("\n" + question + bb.tail_text)
        parts.append(bb.embed_ids(tail))
        positions.append(counter.text(len(tail), bb.device))
        metadata = {"protocol": "trace-explicit-video-qa-v1", "mode": self.mode,
                    "visual_tokens_per_step": counts, "visual_tokens_total": sum(counts),
                    "frame_times": [t for step in steps for t in step.frame_times],
                    "step_ends": [step.end for step in steps]}
        return torch.cat(parts, 1), torch.cat(positions, 1), counter, answer, metadata

    def answer_logits(self, steps: list[VisualStep], record: dict):
        embeds, positions, counter, answer, metadata = self.prefix(steps, record)
        answer_ids = self.bb.ids(answer)
        if not answer_ids:
            raise ValueError("tokenizer produced an empty answer")
        if self.answer_end_token_id is not None:
            answer_ids = answer_ids + [self.answer_end_token_id]
        # Predict first answer token at the final prompt position, subsequent
        # tokens from preceding gold tokens. Prompt and visual tokens have no loss.
        start = embeds.shape[1] - 1
        if len(answer_ids) > 1:
            embeds = torch.cat([embeds, self.bb.embed_ids(answer_ids[:-1])], 1)
            positions = torch.cat([positions, counter.text(len(answer_ids) - 1, self.bb.device)], 1)
        hidden = self.bb.lm(embeds, positions, cache=None, use_cache=False)
        logits = self.bb.logits(hidden[:, start:]).float()
        targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
        return logits, targets, {**metadata, "answer_tokens": len(answer_ids), "sequence_tokens": embeds.shape[1]}

    def forward(self, steps: list[VisualStep], record: dict, *, teacher_logits=None,
                preservation_weight=0., preservation_temperature=2.):
        if isinstance(record, list):
            return self.batch_loss(steps, record, teacher_logits=teacher_logits,
                                   preservation_weight=preservation_weight,
                                   preservation_temperature=preservation_temperature)
        logits, targets, metadata = self.answer_logits(steps, record)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets)
        if not math.isfinite(preservation_weight) or preservation_weight < 0:
            raise ValueError("invalid preservation weight")
        if not math.isfinite(preservation_temperature) or preservation_temperature <= 0:
            raise ValueError("invalid preservation temperature")
        if preservation_weight and teacher_logits is None:
            raise ValueError("positive preservation weight requires teacher logits")
        kl = logits.new_zeros(())
        if teacher_logits is not None:
            if teacher_logits.shape != logits.shape or teacher_logits.requires_grad:
                raise ValueError("teacher must be detached and match answer positions/vocabulary")
            t = preservation_temperature
            kl = F.kl_div(F.log_softmax(logits / t, -1),
                          F.log_softmax(teacher_logits.float() / t, -1),
                          log_target=True, reduction="none").sum(-1).mean() * t * t
        return ce + preservation_weight * kl, {**metadata, "answer_ce": float(ce.detach()),
            "preservation_kl": float(kl.detach()), "teacher_used": teacher_logits is not None}

    def answer_logits_batch(self, steps, records):
        """Right-pad independent examples while preserving each native position.

        Only answer positions reach the vocabulary projection. Causal attention
        and the padding mask isolate samples, including hybrid linear layers.
        """
        if not records or len(steps) != len(records):
            raise ValueError("nonempty matching visual/record batches required")
        inputs, coordinates, targets, metadata, starts = [], [], [], [], []
        for visual, record in zip(steps, records, strict=True):
            embeds, positions, counter, answer, info = self.prefix(visual, record)
            ids = self.bb.ids(answer)
            if not ids:
                raise ValueError("tokenizer produced an empty answer")
            if self.answer_end_token_id is not None:
                ids = ids + [self.answer_end_token_id]
            starts.append(embeds.shape[1] - 1)
            if len(ids) > 1:
                embeds = torch.cat([embeds, self.bb.embed_ids(ids[:-1])], 1)
                positions = torch.cat([positions, counter.text(len(ids) - 1, self.bb.device)], 1)
            inputs.append(embeds[0])
            coordinates.append(positions)
            targets.append(torch.tensor(ids, device=embeds.device, dtype=torch.long))
            metadata.append({**info, "answer_tokens": len(ids), "sequence_tokens": embeds.shape[1]})
        length = max(x.shape[0] for x in inputs)
        embeds = torch.stack([F.pad(x, (0, 0, 0, length - len(x))) for x in inputs])
        positions = torch.stack([F.pad(p, (0, length - p.shape[1])) for p in coordinates], dim=1)
        attention = torch.arange(length, device=embeds.device)[None] < torch.tensor(
            [len(x) for x in inputs], device=embeds.device)[:, None]
        hidden = self.bb.lm(embeds, positions, cache=None, use_cache=False, attention_mask=attention)
        answer_hidden = torch.cat([hidden[i, start:start + len(target)]
                                   for i, (start, target) in enumerate(zip(starts, targets, strict=True))])
        logits = self.bb.logits(answer_hidden[None]).float()[0]
        return list(logits.split([len(t) for t in targets])), targets, metadata

    def batch_loss(self, steps, records, *, teacher_logits=None,
                   preservation_weight=0., preservation_temperature=2.):
        if not math.isfinite(preservation_weight) or preservation_weight < 0:
            raise ValueError("invalid preservation weight")
        if not math.isfinite(preservation_temperature) or preservation_temperature <= 0:
            raise ValueError("invalid preservation temperature")
        logits, targets, metadata = self.answer_logits_batch(steps, records)
        teachers = [None] * len(records) if teacher_logits is None else teacher_logits
        if len(teachers) != len(records):
            raise ValueError("teacher batch differs from student batch")
        ce, kl = [], []
        for value, target, teacher in zip(logits, targets, teachers, strict=True):
            ce.append(F.cross_entropy(value, target))
            divergence = value.new_zeros(())
            if teacher is not None:
                if teacher.requires_grad or teacher.shape != value.shape:
                    raise ValueError("teacher must be detached and match answer positions/vocabulary")
                t = preservation_temperature
                divergence = F.kl_div(F.log_softmax(value / t, -1),
                    F.log_softmax(teacher.float() / t, -1), log_target=True,
                    reduction="none").sum(-1).mean() * t * t
            kl.append(divergence)
        mean_ce, mean_kl = torch.stack(ce).mean(), torch.stack(kl).mean()
        return mean_ce + preservation_weight * mean_kl, {
            "answer_ce": float(mean_ce.detach()), "preservation_kl": float(mean_kl.detach()),
            "teacher_questions": sum(x is not None for x in teachers),
            "answer_tokens": sum(m["answer_tokens"] for m in metadata),
            "sequence_tokens": sum(m["sequence_tokens"] for m in metadata),
            "padded_sequence_tokens": len(records) * max(m["sequence_tokens"] for m in metadata)}

    @torch.no_grad()
    def generate(self, steps: list[VisualStep], record: dict, max_new_tokens: int = 16,
                 *, return_choice: bool = False):
        """Greedy diagnostic decoding, fresh state, no gold answer fed to the LM.

        Recomputes the prefix each token; intended for small validation, not the
        throughput benchmark. Production evaluation needs a verified cache path.
        """
        if max_new_tokens < 1:
            raise ValueError("positive generation length required")
        embeds, positions, counter, _, _ = self.prefix(steps, record)
        generated = []
        choice = None
        for _ in range(max_new_tokens):
            hidden = self.bb.lm(embeds, positions, cache=None, use_cache=False)
            logits = self.bb.logits(hidden[:, -1:])
            if return_choice and choice is None:
                candidates = [self.bb.ids(chr(65 + i)) for i in range(len(record["options"]))]
                if any(len(x) != 1 for x in candidates):
                    raise ValueError("first-token MC control requires single-token option letters")
                choice = int(logits[0, 0, [x[0] for x in candidates]].argmax().item())
            token = int(logits.argmax(-1).item())
            if token == self.answer_end_token_id:
                break
            generated.append(token)
            embeds = torch.cat([embeds, self.bb.embed_ids([token])], 1)
            positions = torch.cat([positions, counter.text(1, self.bb.device)], 1)
        return (generated, choice) if return_choice else generated
