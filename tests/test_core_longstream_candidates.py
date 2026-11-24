import copy

import pytest

from trace_vl.core_vlm.longstream_delayed_candidates import choose_smoke, validate_rows


def cohort():
    return [{"id": f"q{i}", "source_group": f"used{i % 28}", "query_time": 60.0 + i,
             "task": "Event Understanding" if i < 26 else "Prospective Reasoning",
             "temporal_clue_type": "Prior", "frames_required": "multiple"} for i in range(34)]


def test_candidate_population_and_semantic_strata_cannot_silently_drop_or_relabel_targets():
    rows = cohort()
    allowed = {r["source_group"] for r in rows}
    validate_rows(rows, allowed)
    with pytest.raises(ValueError, match="all34"):
        validate_rows(rows[:-1], allowed)
    with pytest.raises(ValueError, match="all34"):
        validate_rows(rows, allowed - {"used0"})
    wrong = copy.deepcopy(rows)
    wrong[-1]["task"] = "Event Understanding"
    with pytest.raises(ValueError, match="semantic strata"):
        validate_rows(wrong, allowed)
    for changes in ({"query_time": 59.9}, {"query_time": float("nan")}, {"frames_required": "single"},
                    {"temporal_clue_type": "Concurrent"}, {"evidence": [0, 1]}):
        wrong = copy.deepcopy(rows)
        wrong[0].update(changes)
        with pytest.raises(ValueError, match="candidate metadata"):
            validate_rows(wrong, allowed)


def test_smoke_deterministically_covers_longest_and_distinct_multiple_capture_source():
    sources = [
        {"source_group": "long", "planned_stream_encoder_calls": 1040, "queries": [{"target_id": "a"}] * 5},
        {"source_group": "multi", "planned_stream_encoder_calls": 692,
         "queries": [{"target_id": "b"}] * 5 + [{"target_id": "c"}] * 5},
        {"source_group": "short", "planned_stream_encoder_calls": 432, "queries": [{"target_id": "d"}] * 5},
    ]
    assert choose_smoke(sources) == choose_smoke(sources[::-1]) == ["long", "multi"]
    with pytest.raises(ValueError, match="multi-target"):
        choose_smoke([sources[0], sources[2]])
