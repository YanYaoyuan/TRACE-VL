import hashlib

import pytest
import torch

from trace_vl.core_vlm.longstream import StoredStep
from trace_vl.core_vlm.persistent_writers import PersistentWriter


def observation(i, tokens=None):
    if tokens is None:
        tokens = torch.ones(16, 4)
    return StoredStep(tokens, (4, 4), (float(i), i+.5), (2*i, 2*i+1), hashlib.sha256(str(i).encode()).hexdigest())


@pytest.mark.parametrize('policy', ['time_coverage', 'spatial_change'])
def test_bounded_long_stream_owns_old_values_and_queries_do_not_write(policy):
    bank = PersistentWriter(policy, 8)
    for i in range(300):
        incoming = observation(i)
        bank.append(incoming)
        incoming.tokens.fill_(999)
        assert len(bank.history) <= 8 and len(bank.recent) <= 4
    before = bank.audit()
    assert before['seen'] == 300 and before['old_seen'] == 296
    assert before['history_steps']+before['recent_steps'] == 12
    assert before['feature_and_fixed_slot_metadata_bytes'] == 12*(16*4*4+96)
    assert before['persistent_auxiliary_feature_bytes'] == 0
    assert bank.history[0].indices == (0, 1)
    assert bank.history[-1].indices == (590, 591)
    for target in (0, 50, 299.5, 100, 10, 299.5):
        read = bank.read(target)
        assert len(read) <= 8 and all(s.end <= target for s in read)
        assert all(torch.equal(s.tokens, torch.ones(16, 4)) for s in read)
        assert bank.audit() == before


def test_spatial_change_preserves_rare_return_event_even_when_mean_feature_is_identical():
    a = torch.zeros(16, 2)
    a[:8, 0] = 1
    a[8:, 1] = 1
    b = a.flip(0)
    assert torch.equal(a.mean(0), b.mean(0))
    bank = PersistentWriter('spatial_change', 8)
    for i in range(150):
        bank.append(observation(i, b if i == 20 else a))
    assert (40, 41) in {s.indices for s in bank.history}
    retained = next(s for s in bank.history if s.indices == (40, 41))
    assert torch.equal(retained.tokens, b)
    # Later A observations did not replace the saved B by their current values.
    assert all(torch.equal(s.tokens, a) for s in bank.recent)


def test_time_coverage_keeps_interior_past_in_a_constant_stream():
    bank = PersistentWriter('time_coverage', 8)
    for i in range(500):
        bank.append(observation(i))
    ends = [s.end for s in bank.history]
    assert ends[0] == .5 and ends[-1] == 495.5
    assert any(150 < t < 350 for t in ends)
    assert max(b-a for a, b in zip(ends, ends[1:])) < 180


@pytest.mark.parametrize('policy', ['time_coverage', 'spatial_change'])
def test_invalid_observation_is_rejected_without_mutating_memory(policy):
    bank = PersistentWriter(policy, 4)
    bank.append(observation(0))
    before = bank.audit()
    for bad in (observation(0), observation(1, torch.full((16, 4), float('nan'))),
                observation(1, torch.ones(15, 4)), observation(1, torch.ones(16, 5))):
        with pytest.raises(ValueError):
            bank.append(bad)
        assert bank.audit() == before
    for target, count in ((float('nan'), 8), (-1, 8), (2, 9), (2, 0)):
        with pytest.raises(ValueError):
            bank.read(target, count)
        assert bank.audit() == before
