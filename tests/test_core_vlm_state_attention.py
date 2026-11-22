import json

import pytest
import torch
from test_core_vlm_native import tiny_example

from trace_vl.core_vlm.native import NativeVideoQALoss, encode_native
from trace_vl.core_vlm.state_attention import (
    READERS,
    OnlineBoxMatcher,
    StateAttention,
    StatePairBuffer,
    region_feature,
)
from trace_vl.core_vlm.state_experiment import batch_logits, fit, natural_manifest, verify_reader


def test_pair_buffer_only_uses_previous_observations_of_same_track():
    buf = StatePairBuffer(2)
    buf.observe(1, {"left": torch.tensor([1.]), "other": torch.tensor([10.])})
    buf.observe(2, {"other": torch.tensor([20.])})
    buf.observe(3, {"left": torch.tensor([3.])})
    before = buf.read(3)
    assert [(p.track, p.t_before, p.t_after) for p in before] == [("other", 1, 2), ("left", 1, 3)]
    buf.observe(4, {"left": torch.tensor([4.])})
    assert [(p.t_before, p.t_after) for p in buf.read(4)] == [(1, 3), (3, 4)]
    assert before[1].before.item() == 1
    with pytest.raises(ValueError, match="causal"):
        buf.read(3)
    with pytest.raises(ValueError, match="increasing"):
        buf.observe(4, {})


def test_region_feature_can_follow_a_track_to_another_grid_cell():
    tokens = torch.arange(4.).reshape(-1, 1)
    assert region_feature(tokens, (2, 2), (0, 0, .5, .5)).item() == 0
    assert region_feature(tokens, (2, 2), (.75, .75, 1, 1)).item() == 3


def test_online_matcher_keeps_ids_on_box_reordering_and_expires_stale_tracks():
    matcher = OnlineBoxMatcher()
    a, b = torch.tensor([1., 0.]), torch.tensor([0., 1.])
    first = matcher.update(0, [a, b])
    second = matcher.update(1, [b, a])
    assert list(second) == list(reversed(first))
    assert not matcher.update(2, [])
    assert set(matcher.update(4, [a, b])).isdisjoint(first)


def test_symmetric_keys_and_antisymmetric_values_under_fixed_query():
    reader = StateAttention(12, 8, "antisymmetric")
    pairs = torch.randn(2, 3, 2, 12)
    q = torch.randn(2, 12)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    keys, values = reader.key_value(pairs)
    reverse_keys, reverse_values = reader.key_value(pairs.flip(-2))
    torch.testing.assert_close(keys, reverse_keys, rtol=0, atol=0)
    torch.testing.assert_close(values, -reverse_values, rtol=0, atol=0)
    direction, weights = reader.read(q, pairs, mask)
    reverse, reverse_weights = reader.read(q, pairs.flip(-2), mask)
    torch.testing.assert_close(direction, -reverse, rtol=0, atol=0)
    assert torch.equal(weights, reverse_weights)
    assert weights[0, 2] == 0 and direction[1].count_nonzero() == 0
    identical = pairs[..., :1, :].expand_as(pairs)
    assert reader.key_value(identical)[1].count_nonzero() == 0


@pytest.mark.parametrize("variant", READERS)
def test_frozen_real_qwen_decoder_learns_only_reader_and_restores_empty_behavior(variant):
    bb, layout, row = tiny_example()
    bb.model.requires_grad_(False)
    steps = encode_native(bb, layout)
    embed, pos, *_ = NativeVideoQALoss(bb).prefix(steps, row)
    pairs = torch.randn(2, 2, bb.hidden)
    views = [{"embeds": embed[0].detach(), "positions": pos, "pairs": pairs,
              "answer": 0, "choices": [20, 21]},
             {"embeds": embed[0].detach()[:-1], "positions": pos[:, :-1], "pairs": pairs.flip(-2),
              "answer": 1, "choices": [20, 21]}]
    reader = StateAttention(bb.hidden, 16, variant)
    before = batch_logits(bb, views, None, 0).detach()
    assert verify_reader(bb, reader, 0, views)["zero_gate_equal"]
    metrics = list(fit(bb, reader, 0, [{"split": "train", "views": views}],
                       {"seed": 17, "updates": 3, "lr": .01, "batch": 2}))
    assert len(metrics) == 3 and reader.gate.abs().item() > 0
    assert any(p.grad is not None and p.grad.count_nonzero() for p in reader.parameters())
    assert all(p.grad is None for p in bb.model.parameters())
    assert torch.equal(batch_logits(bb, views, reader, 0, empty=True), before)
    assert torch.equal(batch_logits(bb, views, None, 0), before)


def test_natural_manifest_rejects_future_frames_and_train_val_leakage(tmp_path):
    def make_row(key, split):
        local = tmp_path / f"{key}.mp4"
        local.write_text(key)
        return {"id": key, "source_group": key, "split": split, "question": "Placed or removed?",
                "options": ["placed", "removed"],
                "views": [{"video": local.name, "span": [i, i+1], "query_time": i+2,
                           "answer": i, "tracks": [{}] * 8} for i in range(2)]}
    rows = [make_row("train", "train"), make_row("val", "val")]
    path = tmp_path / "rows.jsonl"

    def write():
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write()
    assert len(natural_manifest(path)) == 2
    rows[1]["views"][0]["query_time"] = .5
    write()
    with pytest.raises(ValueError, match="end at or before"):
        natural_manifest(path)
    rows[1]["views"][0]["query_time"] = 2
    rows[1]["views"][0]["video"] = "train.mp4"
    write()
    with pytest.raises(ValueError, match="reused"):
        natural_manifest(path)
