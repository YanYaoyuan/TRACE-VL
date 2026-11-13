"""Qwen native processor/forward control under a declared, shared frame budget.

This preserves the HF chat template, timestamp strings and M-RoPE coordinates.
It does not claim to reproduce a benchmark's native sampling protocol: frames
are selected explicitly by core_vlm.data and resizing/sampling is disabled here.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch

from trace_vl.v3.positions import PositionCounter

from .qa import VideoQALoss, VisualStep, grid_pool, question_and_answer, validate_steps


@dataclass
class NativeLayout:
    inputs: dict
    frame_times: list[list[float]]
    question: str
    audit: dict


def prepare_native(processor, images: np.ndarray, times: np.ndarray, record: dict, audit: dict,
                   *, allow_native_timestamp_approximation=False):
    """No gold answer is sent to the processor. No silent 24-FPS fallback.

    Current HF Qwen timestamps are index/fps, not arbitrary PTS. By default refuse
    >50 ms discrepancies. An explicitly declared native-index/FPS comparison may
    retain the official approximation, recording its error; frame selection and
    causality still use actual PTS. It must not claim exact PTS prompt timestamps.
    """
    from transformers.video_utils import VideoMetadata

    fps = float(audit["source_fps"])
    indices = audit["frame_indices"]
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("native timestamps require a measured positive FPS")
    if images.ndim != 5 or images.shape[1] != 2 or images.dtype != np.uint8:
        raise ValueError("native inputs require uint8 [steps,2,H,W,3]")
    if times.shape != (len(images), 2) or len(indices) != 2 * len(images):
        raise ValueError("frame metadata does not match decoded images")
    if (not np.isfinite(times).all() or (times < 0).any()
            or (np.diff(times.reshape(-1)) <= 0).any()
            or times.max() > record["query_time"]):
        raise ValueError("native observations must be strictly chronological and causal")
    if hashlib.sha256(images.tobytes()).hexdigest() != audit["decoded_rgb_sha256"]:
        raise ValueError("decoded image identity changed")
    timestamp_error = float(np.abs(np.array(indices) / fps - times.reshape(-1)).max())
    if timestamp_error > .05 and not allow_native_timestamp_approximation:
        raise ValueError(f"index/FPS timestamps differ from decoded PTS by {timestamp_error:.6f}s")
    question, _ = question_and_answer(record)
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    metadata = VideoMetadata(total_num_frames=audit["source_frame_count"], fps=fps,
                             frames_indices=list(indices))
    inputs = processor(text=[prompt], videos=[images.reshape(-1, *images.shape[2:])],
                       video_metadata=[metadata], do_sample_frames=False, do_resize=False,
                       return_tensors="pt")
    inputs.pop("video_metadata", None)
    if inputs["video_grid_thw"].shape != (1, 3) or int(inputs["video_grid_thw"][0, 0]) != len(images):
        raise ValueError("processor changed the number of temporal patches")
    report = {**audit, "processor": type(processor).__name__, "timestamp_max_error_seconds": timestamp_error,
              "timestamp_policy": "native-index-fps" if allow_native_timestamp_approximation else "strict",
              "timestamp_approximation_over_50ms": timestamp_error > .05,
              "timestamp_text_seconds": (np.array(indices).reshape(-1, 2).mean(1) / fps).tolist(),
              "input_ids_sha256": hashlib.sha256(inputs["input_ids"].numpy().tobytes()).hexdigest(),
              "protocol": "qwen-hf-native-shared-observation-v1"}
    return NativeLayout(dict(inputs), times.tolist(), question, report)


@torch.no_grad()
def encode_native(backbone, layout: NativeLayout):
    grid = layout.inputs["video_grid_thw"]
    x = backbone.encode(layout.inputs["pixel_values_videos"], grid)
    h, w = backbone.merged_grid(grid)
    if x.shape != (len(layout.frame_times) * h * w, backbone.hidden):
        raise ValueError("native visual features do not match the frame/grid layout")
    return [VisualStep(block, (h, w), float(pair[-1]), tuple(pair))
            for block, pair in zip(x.split(h * w), layout.frame_times, strict=True)]


@torch.no_grad()
def check_native_forward(backbone, layout, steps, record):
    """Verify the full-token bridge against the actual HF multimodal forward.

    This checks next-token logits, not merely identical argmax labels. The strict
    CPU float32 regression additionally checks every prompt token and answer CE.
    """
    net = NativeVideoQALoss(backbone, mode="full")
    inputs = {k: v.to(backbone.device) for k, v in layout.inputs.items()}
    direct = backbone.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    embeds, positions, _, _, _ = net.prefix(steps, record)
    hidden = backbone.lm(embeds, positions, cache=None, use_cache=False)
    packed = backbone.logits(hidden[:, -1:])[:, -1].float()
    gap = float((direct - packed).abs().max())
    agree = bool(torch.equal(direct.argmax(-1), packed.argmax(-1)))
    limit = 1e-5 if backbone.dtype == torch.float32 else .05
    if not math.isfinite(gap) or gap > limit or not agree:
        raise RuntimeError(f"native forward mismatch: max_logit_gap={gap}, argmax_equal={agree}")
    return {"max_logit_abs_difference": gap, "argmax_equal": agree, "tolerance": limit}


class NativeVideoQALoss(VideoQALoss):
    """Full mode preserves native packing; grid replaces visual spans only.

    Positions for pooled tokens are selected at bin centres in the original grid.
    Text, timestamps, question and other tokens keep their native coordinates.
    Answer gradients can reach features and LM adapters as in VideoQALoss.
    """

    def prefix(self, steps, record):
        bb = self.bb
        layout = record["native_layout"]
        validate_steps(steps, record["query_time"], bb.hidden)
        question, answer = question_and_answer(record)
        if question != layout.question or [list(s.frame_times) for s in steps] != layout.frame_times:
            raise ValueError("question/observations do not match native layout")
        inputs = layout.inputs
        ids = inputs["input_ids"].to(bb.device)
        if ids.shape[0] != 1 or not torch.all(inputs["attention_mask"] == 1):
            raise ValueError("this packing path requires one unpadded example")
        types = inputs["mm_token_type_ids"].to(bb.device)
        mask = ids == bb.video_token
        if not torch.equal(mask, types == 2):
            raise ValueError("video placeholder/type mismatch")
        grid = inputs["video_grid_thw"].to(bb.device)
        positions, _ = bb.model.model.get_rope_index(
            ids, mm_token_type_ids=types, video_grid_thw=grid,
            attention_mask=inputs["attention_mask"].to(bb.device))
        positions = positions[:, 0]
        text = bb.embed_ids(ids[0].tolist())
        video_indices = mask[0].nonzero().flatten()
        counts = [s.grid[0] * s.grid[1] for s in steps]
        if len(video_indices) != sum(counts) or int(grid[0, 0]) != len(steps):
            raise ValueError("native placeholder count does not match features")
        chunks = video_indices.split(counts)
        parts, pos, emitted, cursor = [], [], [], 0
        for step, indices in zip(steps, chunks, strict=True):
            lo, hi = int(indices[0]), int(indices[-1]) + 1
            if hi - lo != len(indices):
                raise ValueError("native visual spans are not contiguous")
            h, w = step.grid
            merge = bb.visual.spatial_merge_size
            if (h, w) != (int(grid[0, 1]) // merge, int(grid[0, 2]) // merge):
                raise ValueError("native spatial grid mismatch")
            x = step.tokens.to(device=bb.device, dtype=bb.dtype)
            visual_pos = positions[:, lo:hi]
            if self.mode == "grid":
                x, rr, cc = grid_pool(x, step.grid, self.pool_grid)
                visual_pos = visual_pos[:, rr * w + cc]
            parts.extend([text[:, cursor:lo], x[None]])
            pos.extend([positions[:, cursor:lo], visual_pos])
            cursor = hi
            emitted.append(len(x))
        parts.append(text[:, cursor:])
        pos.append(positions[:, cursor:])
        counter = PositionCounter(int(positions.max()) + 1)
        info = {"protocol": "qwen-hf-native-shared-observation-v1", "mode": self.mode,
                "visual_tokens_per_step": emitted, "visual_tokens_total": sum(emitted),
                "frame_times": [t for s in steps for t in s.frame_times], "step_ends": [s.end for s in steps],
                "native_input_ids_sha256": layout.audit.get("input_ids_sha256"),
                "timestamp_text_seconds": layout.audit.get("timestamp_text_seconds")}
        return torch.cat(parts, 1), torch.cat(pos, 1), counter, answer, info
