import math

import pytest
import torch
from test_complementary_state import HEAD, TAIL, blocks

from trace_vl.b0_window40 import analysis, run
from trace_vl.complementary_memory.state_stream import NativeStream, score


@pytest.fixture
def bb():
    from test_core_vlm_native import tiny_example

    backbone, _, _ = tiny_example()
    backbone.text_config = backbone.model.config.text_config
    backbone.layer_types = list(backbone.text_config.layer_types)
    backbone.linear_layers, backbone.full_layers = [0], [1]
    backbone.vision_start, backbone.vision_end = 1, 2
    backbone.ids = lambda text: [20 + ord(text) % 20]
    with torch.no_grad():
        yield backbone


@torch.no_grad()
def test_window40_retains_exact_newest_steps_and_query_forks_are_isolated(bb):
    narrow = NativeStream(bb, HEAD, "window", recent=8)
    wide = NativeStream(bb, HEAD, "window", recent=40)
    state_bytes = None
    for block in blocks(47):
        narrow.append(block)
        wide.append(block)
        assert narrow.audit()["kv_steps"] == min(block.index + 1, 8)
        assert wide.audit()["kv_steps"] == min(block.index + 1, 40)
        state_bytes = state_bytes or wide.audit()["state_bytes"]
        assert narrow.audit()["state_bytes"] == wide.audit()["state_bytes"] == state_bytes
        assert narrow.counter.pos == wide.counter.pos
    assert list(wide.recent) == list(range(7, 47)) and not wide.history
    before = wide.snapshot()
    first = score(bb, wide.fork(), wide.counter.pos, TAIL, 3)
    assert score(bb, wide.fork(), wide.counter.pos, TAIL, 3) == first
    wide.verify(before)


def test_reproduction_checks_options_finiteness_threshold_and_choice_changes():
    old = {"choice": 0, "option_logits_fp32": [.1, 0.]}
    current = {"choice": 1, "option_logits_fp32": [0., .1]}
    result = run.reproduction(current, old, 0)
    assert result["max_logit_gap"] == .1 and result["choice_changed"]
    assert result["gold_nll_change"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="mismatch"):
        run.reproduction({"choice": 0, "option_logits_fp32": [1., 0.]}, old, 0)
    with pytest.raises(ValueError, match="finite"):
        run.reproduction({"choice": 0, "option_logits_fp32": [math.nan, 0.]}, old, 0)


def examples(seed_margins=(2., 2., 2.)):
    rows, refs = [], {}
    for i in range(4):
        for delay in (0, 15, 60, 180, 300):
            identity = f"{i}-{delay}"
            row = {"id": identity, "source_group": str(i), "gold": 0, "delay_seconds": delay,
                   "predictions": {name: {"option_logits_fp32": [0., 0.]} for name in ("window8", "window40")}}
            old = {**row, "predictions": {"window_stream": {"option_logits_fp32": [0., 0.]}}}
            old["predictions"].update({f"reservoir{s}:S1K1": {"option_logits_fp32": [m, 0.]}
                                       for s, m in zip(analysis.SEEDS, seed_margins, strict=True)})
            rows.append(row)
            refs[identity] = old
    return rows, refs


def test_paired_analysis_preserves_primary_sign_and_source_seed_averaging():
    rows, refs = examples()
    value = analysis.analyze(rows, refs)
    assert analysis.outcome(value)["primary_300s"] == "favours_reservoir"
    assert value["300"]["contrasts"]["reservoir_minus_window40"]["nll"]["sources"] == 4
    rows, refs = examples((2., 0., -2.))
    value = analysis.analyze(rows, refs)
    expected = sum(math.log1p(math.exp(-m)) for m in (2., 0., -2.)) / 3
    assert value["300"]["arms"]["reservoir"]["nll"] == pytest.approx(expected)
    assert expected > math.log(2)
    assert analysis.outcome(value)["primary_300s"] == "favours_window40"
    with pytest.raises(ValueError, match="duplicate"):
        analysis.analyze(rows + rows[:1], refs)
