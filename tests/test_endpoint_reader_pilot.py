import copy

import numpy as np
import pytest

from trace_vl.core_vlm.train_endpoint_reader import causal_indices, training_sources


def records():
    return [dict(id=f'{s}-{i}', source_id=s, source_group_id='group-' + s,
                 split='candidate_train_pilot', endpoint_rule='actual_full_video_container_end',
                 video=s + '.mp4', video_sha256=s, query_time=t,
                 question='Native question', answer='Native answer',
                 evidence_intervals=None, memory_necessity_verified=False)
            for s, t in [('a', 400.), ('b', 600.), ('c', 500.)] for i in range(2)]


def test_source_schedule_is_question_blind_and_keeps_atomic_metadata_groups():
    original = records()
    selected = training_sources(original)
    altered = copy.deepcopy(list(reversed(original)))
    for row in altered:
        row.update(question='unrelated question', answer='another gold answer')
    assert [[r['id'] for r in group] for group in selected] == [
        [r['id'] for r in group] for group in training_sources(altered)]
    assert [group[0]['source_id'] for group in selected] == ['a', 'b', 'c']
    assert all(len({r['source_id'] for r in group}) == 1 for group in selected)


@pytest.mark.parametrize('change', [
    {'split': 'reserved'}, {'source_group_id': 'group-b'}, {'query_time': 300.},
    {'evidence_intervals': [[0, 1]]}, {'memory_necessity_verified': True},
    {'answer': ''}, {'id': 'b-0'},
])
def test_mixed_holdout_groups_inconsistent_endpoints_and_bad_qa_fail_closed(change):
    rows = records()
    rows[0].update(change)
    with pytest.raises(ValueError):
        training_sources(rows)


def test_causal_sampling_preserves_endpoint_and_handles_container_tail_offset():
    pts = np.arange(0., 10., .04)
    endpoint = 10.333
    indices = causal_indices(pts, endpoint)
    assert len(indices) % 2 == 0 and np.all(np.diff(indices) > 0)
    assert np.all(pts[indices] <= endpoint)
    assert indices[-1] < len(pts)
    # Large future changes cannot influence the selected causal prefix.
    extended = np.concatenate([pts, np.arange(11., 20., .04)])
    np.testing.assert_array_equal(indices, causal_indices(extended, endpoint))
    assert endpoint == 10.333
    with pytest.raises(ValueError, match='strict finite'):
        causal_indices([0., .5, .5, 1.], 1.)
