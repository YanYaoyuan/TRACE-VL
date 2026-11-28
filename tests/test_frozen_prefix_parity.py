import numpy as np
import pytest
import torch
from test_endpoint_reader import bb as bb
from test_frozen_ingest_reader import frozen_copy

from trace_vl.complementary_memory.state_stream import Block
from trace_vl.core_vlm import frozen_prefix_parity as parity
from trace_vl.core_vlm.native import NativeLayout, encode_native
from trace_vl.core_vlm.qa import grid_pool, question_and_answer


def fixture_layout(bb, monkeypatch):
    query = {"question": "Action?", "options": ["first", "second"], "answer": 1, "query_time": 1.5}
    ids = torch.tensor([[5, 1, *([3] * 16), 2, 8, 1, *([3] * 16), 2, 6, 7]])
    generator = torch.Generator().manual_seed(28)
    layout = NativeLayout({"input_ids": ids, "attention_mask": torch.ones_like(ids),
                           "mm_token_type_ids": torch.where(ids == 3, 2, 0),
                           "video_grid_thw": torch.tensor([[2, 8, 8]]),
                           "pixel_values_videos": torch.randn(128, 3 * 2 * 16 * 16, generator=generator)},
                          [[0., .5], [1., 1.5]], question_and_answer(query)[0], {})
    seen_gold = []

    def prepare(processor, images, times, record, audit, **kwargs):
        seen_gold.append(record["answer"])
        assert np.array_equal(times, layout.frame_times)
        return layout

    monkeypatch.setattr(parity, "prepare_native", prepare)
    with torch.no_grad():
        raw = encode_native(bb, layout)
    blocks = []
    for i, step in enumerate(raw):
        tokens, rows, cols = grid_pool(step.tokens, step.grid, (4, 4))
        blocks.append(Block(i, step.end, () if i == 0 else (8,), tokens.detach(), rows, cols, step.grid))
    frames = [np.zeros((2, 128, 128, 3), dtype=np.uint8) for _ in raw]
    metadata = {"streams": [{"avg_frame_rate": "2/1"}], "frames": [{}, {}, {}, {}]}
    return ([5], [6, 7], blocks, raw, frames, query, np.arange(4), np.asarray(layout.frame_times), metadata), seen_gold


@pytest.mark.parametrize("adapt_query", [False, True])
def test_staged_native_matches_chunked_for_same_and_different_query_adapters(bb, monkeypatch, adapt_query):
    args, seen_gold = fixture_layout(bb, monkeypatch)
    prefix = frozen_copy(bb)
    prefix.processor = object()
    if adapt_query:
        with torch.no_grad():
            for name, parameter in bb.model.named_parameters():
                if "lora_B" in name:
                    parameter.add_(.1)
    result = parity.staged_prefix_parity(bb, prefix, *args)
    assert result["mode"] == "staged_native_prefix_query"
    assert result["exact_packing"] and result["live_stream_unchanged"]
    assert result["max_letter_logit_gap"] <= 3e-5
    assert not result["gold_used"] and seen_gold == [0]
    assert all(p.grad is None for p in prefix.model.parameters())


def test_staged_native_parity_refuses_corrupt_block_coordinates(bb, monkeypatch):
    args, _ = fixture_layout(bb, monkeypatch)
    prefix = frozen_copy(bb)
    prefix.processor = object()
    original = parity.block_inputs

    def wrong_coordinates(*args):
        embeds, positions = original(*args)
        return embeds, positions + 1

    monkeypatch.setattr(parity, "block_inputs", wrong_coordinates)
    with pytest.raises(RuntimeError, match="packing differs"):
        parity.staged_prefix_parity(bb, prefix, *args)


def test_staged_native_uses_no_actual_gold_answer(bb, monkeypatch):
    args, seen_gold = fixture_layout(bb, monkeypatch)
    prefix = frozen_copy(bb)
    prefix.processor = object()
    before = parity.staged_prefix_parity(bb, prefix, *args)
    changed = [*args]
    changed[5] = {**args[5], "answer": 0}
    after = parity.staged_prefix_parity(bb, prefix, *changed)
    assert before == after and seen_gold == [0, 0]
