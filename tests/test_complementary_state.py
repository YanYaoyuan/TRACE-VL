import math

import pytest
import torch

pytest.importorskip("transformers")
from test_core_vlm_native import tiny_example  # noqa: E402

from trace_vl.complementary_memory import analysis  # noqa: E402
from trace_vl.complementary_memory.state_stream import (  # noqa: E402
    Block,
    NativeStream,
    block_inputs,
    replay,
    score,
    segment_layout,
    state_tensors,
)
from trace_vl.v3.positions import PositionCounter  # noqa: E402

HEAD, TAIL = [5, 6], [10, 11, 12]


@pytest.fixture
def bb():
    backbone, _, _ = tiny_example()
    backbone.text_config = backbone.model.config.text_config
    backbone.layer_types = list(backbone.text_config.layer_types)
    backbone.linear_layers, backbone.full_layers = [0], [1]
    backbone.vision_start, backbone.vision_end = 1, 2
    backbone.ids = lambda text: [20 + ord(text) % 20]
    with torch.no_grad():
        yield backbone


def blocks(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [Block(i, float(i), (8, 9 + i % 3), torch.randn(4, 32, generator=generator),
                  torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), (2, 2)) for i in range(n)]


def feed(stream, items):
    for item in items:
        stream.append(item)
    return stream


def logits(bb, stream, **kw):
    return torch.tensor(score(bb, stream.fork(**kw), stream.counter.pos, TAIL, 3)["option_logits_fp32"])


def test_cached_steps_match_one_uncached_forward(bb):
    items = blocks(6)
    stream = feed(NativeStream(bb, HEAD, "full", recent=2), items)
    counter = PositionCounter()
    parts = [(bb.embed_ids(HEAD), counter.text(2))] + [block_inputs(bb, b, counter) for b in items]
    parts.append((bb.embed_ids(TAIL), counter.text(3)))
    hidden = bb.lm(torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1),
                   cache=None, use_cache=False)[0, -1]
    expected = bb.lm_head.weight[[bb.ids(c)[0] for c in "ABC"]] @ hidden
    torch.testing.assert_close(logits(bb, stream), expected, atol=1e-4, rtol=1e-4)


def test_forks_and_interventions_never_change_live_stream(bb):
    items = blocks(12)
    stream = feed(NativeStream(bb, HEAD, "reservoir", recent=3, capacity=4, seed=1), items)
    donor = replay(bb, HEAD, items[-3:], positions_start=stream.counter.pos - 3 * 6)
    assert donor.counter.pos == stream.counter.pos
    before, donor_state = stream.snapshot(), [t.clone() for t in state_tensors(bb, donor.cache)]
    first = logits(bb, stream)
    for history in (True, False):
        for source in (None, donor):
            logits(bb, stream, history=history, state_from=source)
    stream.verify(before)
    assert torch.equal(first, logits(bb, stream))
    assert all(torch.equal(a, b) for a, b in zip(donor_state, state_tensors(bb, donor.cache), strict=True))
    stream.append(blocks(13)[-1])
    with pytest.raises(RuntimeError, match="changed the live"):
        stream.verify(before)


def test_factors_change_only_the_declared_component(bb):
    items = blocks(12)
    stream = feed(NativeStream(bb, HEAD, "reservoir", recent=3, capacity=4), items)
    donor = replay(bb, HEAD, items[-3:])
    # A tiny GDN can forget the omitted prefix exactly, so natural replay is
    # not guaranteed to produce a different state. This test checks the swap
    # mechanism, not memory quality: create an explicitly distinct donor.
    for layer_index in bb.linear_layers:
        layer = donor.cache.layers[layer_index]
        for name in ("recurrent_states", "conv_states"):
            states = getattr(layer, name)
            for key, value in states.items():
                states[key] = value.clone() + 0.25
    full, no_kv = stream.fork(), stream.fork(history=False)
    swapped = stream.fork(state_from=donor)
    assert all(torch.equal(a, b) for a, b in zip(state_tensors(bb, full), state_tensors(bb, no_kv), strict=True))
    assert bb.kv_length(no_kv) == 2 + 3 * 8 and bb.kv_length(full) == 2 + 7 * 8
    assert torch.equal(swapped.layers[1].keys, full.layers[1].keys)
    assert all(torch.equal(a, b) for a, b in zip(state_tensors(bb, swapped), state_tensors(bb, donor.cache),
                                                 strict=True))
    assert not torch.equal(swapped.layers[0].recurrent_states[0], full.layers[0].recurrent_states[0])


def test_before_expiry_every_visual_arm_is_the_same(bb):
    items = blocks(3)
    streams = [feed(NativeStream(bb, HEAD, p, recent=3, capacity=4), items) for p in ("window", "reservoir", "full")]
    fresh = replay(bb, HEAD, items)
    for stream in streams:
        torch.testing.assert_close(logits(bb, stream), logits(bb, fresh))
        torch.testing.assert_close(logits(bb, stream, history=False, state_from=fresh), logits(bb, fresh))


def test_kv_is_bounded_and_recurrent_bytes_are_constant(bb):
    window = NativeStream(bb, HEAD, "window", recent=3)
    reservoir = NativeStream(bb, HEAD, "reservoir", recent=3, capacity=4, seed=2)
    state = None
    for item in blocks(30):
        window.append(item)
        reservoir.append(item)
        a, b = window.audit(), reservoir.audit()
        assert a["kv_steps"] <= 3 and b["kv_steps"] <= 7
        state = state or a["state_bytes"]
        assert a["state_bytes"] == b["state_bytes"] == state
    assert reservoir.audit()["recent"] == [27, 28, 29]
    assert len(reservoir.history) == 4 and not set(reservoir.history) & set(reservoir.recent)


def test_native_prompt_segmentation():
    ids = [5, 1, 8, 1, 3, 3, 2, 9, 1, 3, 3, 2, 2, 6, 7]
    assert segment_layout(ids, 3, 1, 2, [8]) == ([5, 1], [2, 6, 7])
    with pytest.raises(ValueError, match="timestamp"):
        segment_layout(ids, 3, 1, 2, [7])
    with pytest.raises(ValueError, match="no video"):
        segment_layout([1, 2], 3, 1, 2, [])


def row(group, delay, margins):
    predictions = {a: {"option_logits_fp32": [margins.get(a, 0.), 0.]} for a in (
        "text_only", "recent_replay", "window_stream", "full_stream", "snapshot_replay")}
    for seed in (1, 2):
        for factor in analysis.FACTORS:
            predictions[f"reservoir{seed}:{factor}"] = {"option_logits_fp32": [margins.get(factor, 0.), 0.]}
    return {"source_group": group, "delay_seconds": delay, "gold": 0, "predictions": predictions}


def test_analysis_signs_seed_average_and_outcomes():
    assert analysis.question_metrics([0., 0.], 1)["nll"] == pytest.approx(math.log(2))
    rows = [row(str(i), d, {"S1K1": 2 + .01 * i}) for i in range(10) for d in (0, 300)]
    result = analysis.analyze(rows, (1, 2))
    kv = result["300"]["contrasts"]["history_kv_given_state"]["nll"]
    assert kv["mean"] < 0 and kv["reading"] == "helps"
    decision = analysis.outcome(result)
    assert decision["decision"].startswith("go_B:")
    assert decision["interaction_nll_300s"] == "kv_worth_more_with_carried_state"
    flat = analysis.outcome(analysis.analyze([row(str(i), d, {}) for i in range(5) for d in (0, 300)], (1, 2)))
    assert flat["decision"].startswith("hold") and not flat["state_conditional_claim_supported"]
    damaged = [row(str(i), d, {"window_stream": -3.}) for i in range(5) for d in (0, 300)]
    assert analysis.outcome(analysis.analyze(damaged, (1, 2)))["decision"].startswith("reader_unfit")
    with pytest.raises(ValueError, match="one question"):
        analysis.analyze(rows + rows[:1], (1, 2))
