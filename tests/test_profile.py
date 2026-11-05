from trace_vl.config import ModelConfig
from trace_vl.profiling import profile_stream


def test_stream_resource_state_plateaus(tmp_path):
    config = {
        "model": ModelConfig(event_capacity=3, cache_frames=2).to_dict(),
        "evaluation": {"device": "cpu"},
    }
    report = profile_stream(config, tmp_path / "profile.json", lengths=[4, 17])
    a, b = report["rows"]
    assert a["logical_state_bytes"] == b["logical_state_bytes"]
    assert a["retained_events"] == b["retained_events"] == 3
    assert a["cached_frames_per_layer"] == b["cached_frames_per_layer"] == [2, 2]
