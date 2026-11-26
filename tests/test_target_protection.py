import itertools
import math
from collections import Counter

import pytest
import torch
from test_complementary_state import HEAD, TAIL, blocks

from trace_vl.complementary_memory.state_stream import NativeStream, score, state_tensors
from trace_vl.target_protection import analysis
from trace_vl.target_protection.stream import TargetProtectedStream


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


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_original_path_is_exact_until_first_protected_admission(bb, seed):
    baseline = NativeStream(bb, HEAD, "reservoir", seed=seed)
    protected = TargetProtectedStream(bb, HEAD, seed=seed)
    items = blocks(46)
    for block in items[:45]:
        baseline.append(block)
        protected.append(block)
    protected.protect(list(protected.recent))
    assert not protected.protected
    assert baseline.owners == protected.owners and baseline.history == protected.history
    assert baseline.rng.getstate() == protected.rng.getstate()
    assert all(torch.equal(a, b) for a, b in zip(
        state_tensors(bb, baseline.cache), state_tensors(bb, protected.cache), strict=True))
    assert score(bb, baseline.fork(), baseline.counter.pos, TAIL, 3) == score(
        bb, protected.fork(), protected.counter.pos, TAIL, 3)
    protected.append(items[45])
    assert protected.protected == [37] and protected.old_seen == baseline.old_seen
    assert protected.audit()["kv_steps"] == 40 and len(protected.history) == 31


def test_unique_budgets_protection_and_question_isolation(bb):
    stream = TargetProtectedStream(bb, HEAD, seed=19)
    byte_count = None
    for block in blocks(75):
        transient = stream.audit()["kv_steps"] + 1
        stream.append(block)
        if block.index == 44:
            stream.protect(list(stream.recent))
        audit = stream.audit()
        assert transient <= 41 and audit["kv_steps"] == min(block.index + 1, 40)
        byte_count = byte_count or audit["state_bytes"]
        assert audit["state_bytes"] == byte_count
    assert stream.audit()["protected"] == list(range(37, 45))
    assert stream.audit()["reservoir_capacity"] == len(stream.history) == 24
    assert stream.old_seen == 75 - 8 - 8
    before, logical = stream.snapshot(), (stream.audit(), stream.rng.getstate())
    first = score(bb, stream.fork(), stream.counter.pos, TAIL, 3)
    assert score(bb, stream.fork(), stream.counter.pos, TAIL, 3) == first
    stream.verify(before)
    assert (stream.audit(), stream.rng.getstate()) == logical


def test_early_protection_and_snapshot_validation(bb):
    stream = TargetProtectedStream(bb, HEAD)
    items = blocks(14)
    for block in items[:8]:
        stream.append(block)
    with pytest.raises(ValueError, match="snapshot"):
        stream.protect([0] * 8)
    with pytest.raises(ValueError, match="snapshot"):
        stream.protect(list(range(1, 9)))
    stream.protect(list(range(8)))
    with pytest.raises(ValueError, match="snapshot"):
        stream.protect(list(range(8)))
    for block in items[8:]:
        stream.append(block)
        assert stream.audit()["kv_steps"] == block.index + 1
    with pytest.raises(ValueError, match="consecutive"):
        stream.append(items[0])


class Choices:
    def __init__(self, values, bounds):
        self.values, self.bounds = iter(values), iter(bounds)

    def randrange(self, bound):
        assert bound == next(self.bounds)
        result = next(self.values)
        assert 0 <= result < bound
        return result


def test_uniform_thinning_then_update_by_exhaustive_rng_choices():
    # Uniform size-two samples of three candidates, then a protected admission
    # shrinks to one, followed by candidate 3. All 24 equally likely paths.
    counts = Counter()
    for pair in itertools.combinations(range(3), 2):
        for eviction, insertion in itertools.product(range(2), range(4)):
            stream = object.__new__(TargetProtectedStream)
            stream.target_steps, stream.protected = frozenset({99}), []
            stream.capacity, stream.old_seen, stream.history = 2, 3, list(pair)
            stream.rng = Choices([eviction, insertion], [2, 4])
            dropped = stream._departed(99)
            assert dropped in pair and stream.old_seen == 3 and stream.protected == [99]
            stream._departed(3)
            assert stream.old_seen == 4 and len(stream.history) == 1
            counts[stream.history[0]] += 1
    assert counts == {0: 6, 1: 6, 2: 6, 3: 6}


def examples(margins=(2., 0., -2.)):
    rows, refs, windows = [], {}, {}
    for group in range(4):
        for delay in (0, 15, 60, 180, 300):
            identity = f"{group}-{delay}"
            row = {"id": identity, "source_group": str(group), "gold": 0, "delay_seconds": delay,
                   "predictions": {f"protected{s}": {"option_logits_fp32": [m, 0.]}
                                   for s, m in zip(analysis.SEEDS, margins, strict=True)}}
            old = {**row, "predictions": {f"reservoir{s}:S1K1": {"option_logits_fp32": [0., 0.]}
                                        for s in analysis.SEEDS}}
            rows.append(row)
            refs[identity] = old
            windows[identity] = {**row, "predictions": {"window40": {"option_logits_fp32": [0., 0.]}}}
    return rows, refs, windows


def test_metric_seed_averaging_sign_and_no_automatic_training():
    rows, refs, windows = examples()
    result = analysis.analyze(rows, refs, windows)
    expected = sum(math.log1p(math.exp(-m)) for m in (2., 0., -2.)) / 3
    assert result["300"]["arms"]["protected"]["nll"] == pytest.approx(expected)
    assert expected > math.log(2)
    assert analysis.outcome(result)["primary_300s"] == "favours_reservoir"
    rows, refs, windows = examples((2., 2., 2.))
    result = analysis.analyze(rows, refs, windows)
    assert analysis.outcome(result)["primary_300s"] == "favours_protection"
    assert not analysis.outcome(result)["automatic_training"]
    assert not analysis.outcome(result)["relevant_improvement_005_excluded"]
    with pytest.raises(ValueError, match="duplicate"):
        analysis.analyze(rows + rows[:1], refs, windows)
