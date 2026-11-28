"""Native packing control for frozen-ingestion / adapted-query execution."""
from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import torch

from trace_vl.complementary_memory.state_stream import block_inputs, replay, score
from trace_vl.v3.positions import PositionCounter

from .endpoint_reader import _cache_signature
from .native import NativeVideoQALoss, prepare_native


@torch.no_grad()
def staged_prefix_parity(query_bb, prefix_bb, head, tail, blocks, raw, frames,
                         query, indices, times, metadata):
    """Compare native-packed ingestion then query with chunked ingestion then query.

    This is a two-stage model control, not ordinary single-model native parity.
    Actual gold is never passed to the processor or used for scoring.
    """
    if _cache_signature(query_bb) != _cache_signature(prefix_bb):
        raise ValueError("prefix and reader cache/token configurations must match")
    if any(p.requires_grad for p in prefix_bb.model.parameters()):
        raise ValueError("ingestion backbone must be fully frozen")
    if any(m.training for bb in (query_bb, prefix_bb) for m in bb.model.modules()):
        raise ValueError("staged parity requires eval mode")
    n = len(blocks)
    if not n or len(raw) != n or len(frames) != n or not tail:
        raise ValueError("matching nonempty native steps and question tail required")
    images = np.stack(frames)
    audit = {"source_fps": float(Fraction(metadata["streams"][0]["avg_frame_rate"])),
             "source_frame_count": len(metadata["frames"]),
             "frame_indices": [int(i) for i in indices[:2 * n]],
             "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    safe = {**query, "answer": 0}
    layout = prepare_native(prefix_bb.processor, images, np.asarray(times[:n]), safe, audit,
                            allow_native_timestamp_approximation=True)
    native = NativeVideoQALoss(prefix_bb, mode="grid", pool_grid=(4, 4))
    embeds, positions, *_ = native.prefix(raw, {**safe, "native_layout": layout})
    cut = embeds.shape[1] - len(tail)
    if cut <= 0:
        raise ValueError("native layout contains no ingestion prefix")
    counter = PositionCounter()
    parts = [(prefix_bb.embed_ids(head), counter.text(len(head), prefix_bb.device))]
    parts += [block_inputs(prefix_bb, block, counter) for block in blocks]
    manual, coords = torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
    tail_positions = counter.text(len(tail), query_bb.device)
    if (not torch.equal(manual, embeds[:, :cut]) or not torch.equal(coords, positions[:, :cut])
            or not torch.equal(tail_positions, positions[:, cut:])):
        raise RuntimeError("staged stream packing differs from native packing")
    query_tail = query_bb.embed_ids(tail)
    if not torch.equal(query_tail, embeds[:, cut:]):
        raise ValueError("prefix and query token embeddings differ")
    letters = [query_bb.ids(chr(65 + i)) for i in range(len(query["options"]))]
    if any(len(ids) != 1 or ids != prefix_bb.ids(chr(65 + i)) for i, ids in enumerate(letters)):
        raise ValueError("identical single-token answer options required")
    letter_ids = [ids[0] for ids in letters]
    cache = prefix_bb.new_cache()
    prefix_bb.lm(embeds[:, :cut], positions[:, :cut], cache)
    hidden = query_bb.lm(query_tail, positions[:, cut:], cache)[0, -1]
    direct = query_bb.lm_head.weight[letter_ids].float() @ hidden.float()
    stream = replay(prefix_bb, head, blocks)
    before, rng = stream.snapshot(), stream.rng.getstate()
    actual = score(query_bb, stream.fork(), stream.counter.pos, tail, len(letters))
    stream.verify(before)
    if stream.rng.getstate() != rng:
        raise RuntimeError("query changed live ingestion RNG")
    cached = torch.tensor(actual["option_logits_fp32"], dtype=torch.float32, device=direct.device)
    if not bool(torch.isfinite(direct).all()):
        raise RuntimeError("nonfinite staged native logits")
    gap = float((direct - cached).abs().max())
    tolerance = 3e-5 if query_bb.dtype == torch.float32 else .5
    same = int(direct.argmax()) == int(cached.argmax())
    top2 = direct.topk(2).values
    if gap > tolerance or (not same and float(top2[0] - top2[1]) > 2 * gap):
        raise RuntimeError(f"staged native/chunked query mismatch: gap={gap}, same_choice={same}")
    return {"mode": "staged_native_prefix_query", "steps": n, "exact_packing": True,
            "max_letter_logit_gap": gap, "same_choice": same, "tolerance": tolerance,
            "native_prefix_tokens": cut, "question_tail_tokens": len(tail),
            "gold_used": False, "live_stream_unchanged": True}
