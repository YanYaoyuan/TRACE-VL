import math

import pytest
import torch

from trace_vl.complementary_memory.state_stream import NativeStream, block_inputs
from trace_vl.core_vlm.native import NativeLayout
from trace_vl.core_vlm.qa import VisualStep
from trace_vl.delayed_history.mechanics import DelayedReader, DenseArrival, feature_block
from trace_vl.delayed_history.reference import mixed_native_prefix
from trace_vl.v3.positions import PositionCounter

HEAD, TAIL = [5], [6, 7]


@pytest.fixture
def bb():
    from test_core_vlm_native import tiny_example

    bb, _, _ = tiny_example()
    bb.text_config = bb.model.config.text_config
    bb.layer_types = list(bb.text_config.layer_types)
    bb.linear_layers, bb.full_layers = [0], [1]
    bb.vision_start, bb.vision_end = 1, 2
    bb.ids = lambda text: [20 + ord(text) % 20]
    with torch.no_grad():
        yield bb


def arrivals(n, grid=(7, 9)):
    gen = torch.Generator().manual_seed(29)
    return [DenseArrival(i, float(i + 1), (8, 9), torch.randn(math.prod(grid), 32, generator=gen), grid)
            for i in range(n)]


def layout_for(steps):
    grid = steps[0].grid
    ids = HEAD[:]
    for step in steps:
        ids.extend([*step.stamp, 1, *([3] * math.prod(grid)), 2])
    ids.extend(TAIL)
    ids = torch.tensor([ids])
    times = [[s.end - .5, s.end] for s in steps]
    layout = NativeLayout(dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                               mm_token_type_ids=torch.where(ids == 3, 2, 0),
                               video_grid_thw=torch.tensor([[len(steps), grid[0] * 2, grid[1] * 2]])),
                          times, "Synthetic?", {})
    raw = [VisualStep(s.tokens, s.grid, s.end, tuple(t)) for s, t in zip(steps, times, strict=True)]
    row = {"question": "Synthetic?", "answer_text": "unused", "query_time": steps[-1].end,
           "native_layout": layout}
    return raw, row


@pytest.mark.parametrize("history_grid", [(2, 2), (4, 4)])
def test_mixed_prefix_matches_independent_hf_native_coordinates_and_uncached_logits(bb, history_grid):
    items = arrivals(12)
    reader = DelayedReader(bb, HEAD, capacity=32, history_grid=history_grid)
    for step in items:
        reader.append(step)
    assert reader.history.seen == 4 and reader.history.audit()["recent"] == []
    raw, row = layout_for(items)
    grids = [history_grid] * 4 + [(4, 4)] * 8
    expected, expected_positions, _ = mixed_native_prefix(bb, raw, row, grids)
    counter = PositionCounter()
    pieces = [(bb.embed_ids(HEAD), counter.text(len(HEAD)))]
    pieces.extend(block_inputs(bb, feature_block(step, grid), counter) for step, grid in zip(items, grids, strict=True))
    pieces.append((bb.embed_ids(TAIL), counter.text(len(TAIL))))
    actual, positions = torch.cat([x for x, _ in pieces], 1), torch.cat([p for _, p in pieces], 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(positions, expected_positions)
    prediction, audit = reader.answer(TAIL, 3, query_time=items[-1].end)
    hidden = bb.lm(expected, expected_positions, cache=None, use_cache=False)[0, -1]
    direct = bb.lm_head.weight[[bb.ids(c)[0] for c in "ABC"]].float() @ hidden.float()
    torch.testing.assert_close(torch.tensor(prediction["option_logits_fp32"]), direct, atol=1e-4, rtol=1e-4)
    assert audit["readout_kv_tokens"] == expected.shape[1] - len(TAIL)
    assert audit["next_position"] == int(expected_positions[0, -len(TAIL)])
    assert reader.history.seen == 4
    assert reader.answer(TAIL, 3, query_time=items[-1].end)[0] == prediction


def test_visual_budget_boundaries_and_metadata_cost_are_explicit(bb):
    fine = DelayedReader(bb, HEAD, capacity=32, history_grid=(4, 4), seed=17)
    coarse = DelayedReader(bb, HEAD, capacity=128, history_grid=(2, 2), seed=17)
    state, feature_bound = {}, {}
    for step in arrivals(145):
        for name, reader in (("fine", fine), ("coarse", coarse)):
            reader.append(step)
            a = reader.audit()
            assert a["history_state_bytes"] == state.setdefault(name, a["history_state_bytes"])
            if step.index == 7:
                feature_bound[name] = a["recent_feature_bytes"]
            if step.index >= 7:
                assert a["recent_feature_bytes"] == feature_bound[name]
    af, ac = fine.audit(), coarse.audit()
    assert af["readout_visual_tokens"] == ac["readout_visual_tokens"] == 640
    assert len(af["history_indices"]) == 32 and len(ac["history_indices"]) == 128
    _, f = fine.answer(TAIL, 3, query_time=145.)
    _, c = coarse.answer(TAIL, 3, query_time=145.)
    assert f["readout_recent_indices"] == c["readout_recent_indices"] == list(range(137, 145))
    assert f["next_position"] == c["next_position"]
    assert c["readout_kv_tokens"] > f["readout_kv_tokens"]
    assert c["readout_kv_bytes"] > f["readout_kv_bytes"]


def test_original_reservoir_owner_law_is_preserved_but_state_timing_is_separate(bb):
    delayed = DelayedReader(bb, HEAD, capacity=3, history_grid=(4, 4), seed=29)
    original = NativeStream(bb, HEAD, "reservoir", recent=8, capacity=3, seed=29)
    for step in arrivals(20):
        delayed.append(step)
        original.append(feature_block(step, (4, 4)))
        assert delayed.audit()["history_indices"] == original.audit()["history"]
    assert delayed.history.seen == 12 and original.seen == 20
    before = delayed.history.snapshot()
    delayed.answer(TAIL, 3, query_time=20.)
    delayed.history.verify(before)
    assert delayed.history.seen == 12  # recent is never committed by a question


def test_future_time_and_caller_mutation_cannot_change_observed_prefix(bb):
    items = arrivals(10)
    reader = DelayedReader(bb, HEAD, capacity=32)
    for step in items:
        reader.append(step)
    first = reader.answer(TAIL, 3, query_time=10.)[0]
    items[-1].tokens.add_(1000)
    assert reader.answer(TAIL, 3, query_time=10.)[0] == first
    with pytest.raises(ValueError, match="causal"):
        reader.answer(TAIL, 3, query_time=9.)
    with pytest.raises(ValueError, match="chronological"):
        reader.append(items[-1])
    before = reader.audit()
    assert reader.audit() == before


def test_nondivisible_dense_pooling_needs_area_weights(bb):
    h, w = 7, 9
    values = (torch.arange(h)[:, None].float() ** 2 + torch.arange(w)[None].float() ** 3).reshape(-1, 1).repeat(1, 32)
    step = DenseArrival(0, 1., (8, 9), values, (h, w))
    fine, coarse = feature_block(step, (4, 4)), feature_block(step, (2, 2))
    cells = fine.tokens.reshape(4, 4, 32)
    weighted, naive = [], []
    for r in range(2):
        for c in range(2):
            weights = torch.tensor([((i + 1) * h // 4 - i * h // 4) * ((j + 1) * w // 4 - j * w // 4)
                                    for i in range(2*r, 2*r+2) for j in range(2*c, 2*c+2)]).float()
            patch = cells[2*r:2*r+2, 2*c:2*c+2].reshape(4, 32)
            weighted.append((patch * weights[:, None]).sum(0) / weights.sum())
            naive.append(patch.mean(0))
    torch.testing.assert_close(torch.stack(weighted), coarse.tokens, atol=1e-4, rtol=1e-6)
    assert (torch.stack(naive) - coarse.tokens).abs().max() > 1
