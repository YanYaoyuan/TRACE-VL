"""Video-to-language answer loss for controlled core-VLM experiments.

This uses TRACE's explicit visual-embedding protocol, not Qwen's official native
video processor/chat path. Full and grid modes share observations and prompts.
There is no alert head, teacher distribution, retention scorer, or recurrent LM
cache in this clip-level training path. A trained LM adapter is part of the VLM.
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

    def __init__(self, backbone, *, mode: str = "full", pool_grid: tuple[int, int] = (4, 5)):
        super().__init__()
        if mode not in ("full", "grid"):
            raise ValueError("mode must be full or grid")
        self.bb = backbone
        self.lm_model = backbone.model
        self.mode, self.pool_grid = mode, pool_grid

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

    def forward(self, steps: list[VisualStep], record: dict):
        embeds, positions, counter, answer, metadata = self.prefix(steps, record)
        answer_ids = self.bb.ids(answer)
        if not answer_ids:
            raise ValueError("tokenizer produced an empty answer")
        # Predict first answer token at the final prompt position, subsequent
        # tokens from preceding gold tokens. Prompt and visual tokens have no loss.
        start = embeds.shape[1] - 1
        if len(answer_ids) > 1:
            embeds = torch.cat([embeds, self.bb.embed_ids(answer_ids[:-1])], 1)
            positions = torch.cat([positions, counter.text(len(answer_ids) - 1, self.bb.device)], 1)
        hidden = self.bb.lm(embeds, positions, cache=None, use_cache=False)
        logits = self.bb.logits(hidden[:, start:]).float()
        targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets)
        return loss, {**metadata, "answer_tokens": len(answer_ids), "sequence_tokens": embeds.shape[1]}
