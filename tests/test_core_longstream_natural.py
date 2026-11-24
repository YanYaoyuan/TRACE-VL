import pytest

from trace_vl.core_vlm.longstream_natural import prefix_calls, smoke_groups, validate_rows


def test_natural_metadata_guard_preserves_existing_development_boundary():
    row = {
        "id": "q",
        "source_group": "used",
        "query_time": 608.0,
        "temporal_clue_type": "Prior",
        "frames_required": "multiple",
    }
    validate_rows([row], {"used"})
    with pytest.raises(ValueError, match="development boundary"):
        validate_rows([row], {"reserved"})
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows([row, row], {"used"})
    for changes in (
        {"query_time": 599.9},
        {"query_time": float("nan")},
        {"temporal_clue_type": "Concurrent"},
        {"frames_required": "single"},
        {"evidence": [0, 4]},
    ):
        with pytest.raises(ValueError, match="metadata selection"):
            validate_rows([{**row, **changes}], {"used"})


def test_encoder_count_respects_fractional_source_times_and_causal_boundary():
    times = [[0.0, 0.49], [1.0, 1.49], [2.0, 2.49]]
    assert prefix_calls(times, 1.0) == 1
    assert prefix_calls(times, 1.49) == 2
    assert prefix_calls(times, 2.48) == 2


def test_smoke_includes_longest_prefix_and_shared_source_multiple_arrivals():
    sources = [
        {"source_group": "long", "queries": [{"query_time": 920.0}]},
        {"source_group": "multi", "queries": [{"query_time": 651.0}, {"query_time": 665.0}]},
        {"source_group": "short", "queries": [{"query_time": 608.0}]},
    ]
    assert smoke_groups(sources) == ["long", "multi"]
    with pytest.raises(ValueError, match="multiple arrivals"):
        smoke_groups([sources[0], sources[2]])
