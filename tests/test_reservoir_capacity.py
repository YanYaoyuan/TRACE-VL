import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from test_complementary_state import HEAD, TAIL, blocks

from trace_vl.complementary_memory.state_stream import NativeStream, score
from trace_vl.reservoir_capacity import analysis
from trace_vl.reservoir_capacity.cost import estimate_cost
from trace_vl.reservoir_capacity.supervisor import BudgetExceeded, budget, run_jobs


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


def test_actual_capacity_boundaries_and_fork_isolation(bb):
    streams = {c: NativeStream(bb, HEAD, "reservoir", recent=8, capacity=c, seed=17) for c in (32, 128)}
    state = {}
    for item in blocks(145):
        for cap, stream in streams.items():
            assert stream.audit()["kv_steps"] + 1 <= cap + 9
            stream.append(item)
            audit = stream.audit()
            assert audit["kv_steps"] == min(item.index + 1, cap + 8)
            assert len(audit["history"]) == min(max(0, item.index + 1 - 8), cap)
            assert not set(audit["recent"]) & set(audit["history"])
            assert audit["state_bytes"] == state.setdefault(cap, audit["state_bytes"])
        if item.index == 39:
            assert streams[32].audit() == streams[128].audit()
            assert score(bb, streams[32].fork(), streams[32].counter.pos, TAIL, 3) == score(
                bb, streams[128].fork(), streams[128].counter.pos, TAIL, 3)
    assert streams[32].audit()["kv_steps"] == 40 and streams[128].audit()["kv_steps"] == 136
    for stream in streams.values():
        before, logical = stream.snapshot(), (stream.audit(), stream.rng.getstate())
        first = score(bb, stream.fork(), stream.counter.pos, TAIL, 3)
        assert score(bb, stream.fork(), stream.counter.pos, TAIL, 3) == first
        stream.verify(before)
        assert (stream.audit(), stream.rng.getstate()) == logical


def examples():
    rows, windows, protected = [], {}, {}
    for group in range(3):
        for delay in (0, 15, 60, 180, 300):
            identity = f"{group}-{delay}"
            row = {"id": identity, "source_group": str(group), "gold": 0, "delay_seconds": delay,
                   "predictions": {f"r{cap}s{s}": {"option_logits_fp32": [m, 0.]}
                                   for cap in (32, 128) for s, m in zip(analysis.SEEDS, ((0., 0., 0.) if cap == 32 else (2., 0., -2.)), strict=True)}}
            rows.append(row)
            windows[identity] = {**row, "predictions": {"window40": {"option_logits_fp32": [9., 0.]}}}
            protected[identity] = {**row, "predictions": {f"protected{s}": {"option_logits_fp32": [9., 0.]} for s in analysis.SEEDS}}
    return rows, windows, protected


def test_fresh_metric_level_seed_mean_and_secondary_references():
    rows, windows, protected = examples()
    result = analysis.analyze(rows, windows, protected)
    expected = sum(math.log1p(math.exp(-m)) for m in (2., 0., -2.)) / 3 - math.log(2)
    assert result["300"]["contrasts"]["R128_minus_R32"]["nll"]["mean"] == pytest.approx(expected)
    assert expected > 0 and analysis.outcome(result)["primary_300s"] == "favours_R32"
    assert set(result["60"]["arms"]) == {"R32", "R128"}
    assert "saved_window40" in result["15"]["arms"]
    assert not analysis.outcome(result)["automatic_training"]
    with pytest.raises(ValueError, match="duplicate"):
        analysis.analyze(rows + rows[:1], windows, protected)


def smoke_cost_inputs():
    base = {"sources": [{"source_group": str(i), "queries": [{"planned_encoder_calls": 300} for _ in range(5)]} for i in range(77)],
            "smoke_source_groups": ["0", "76"]}
    inputs = {"id": "test-input", "reference": base}
    sources = [{"completed": True, "input_id": inputs["id"], "source_group": g, "live_streams": 6,
                "steps": 300, "source_wall_seconds": 100.,
                "seconds": dict(vision=10., packing_audit=1., lm_r32=20., lm_r128=25.,
                                answer_r32=2., answer_r128=3., question_verification=4.)} for g in base["smoke_source_groups"]]
    workers = {f"smoke-{i}": {"exit_code": 0, "wall_seconds": 120.} for i in range(2)}
    return inputs, sources, workers


def test_cost_uses_exact_remaining_shards_and_enforces_ceiling():
    inputs, sources, workers = smoke_cost_inputs()
    result = estimate_cost(inputs, sources, workers, 4)
    assert sum(s["sources"] for s in result["remaining_shards"]) == 75
    assert sum(s["questions"] for s in result["remaining_shards"]) == 375
    assert result["startup_allowance_seconds"] == 20
    assert result["within_ceiling"] and result["conservative_wall_seconds"] >= result["component_wall_seconds"]
    for source in sources:
        source["source_wall_seconds"] *= 100
        source["seconds"] = {k: v * 100 for k, v in source["seconds"].items()}
    assert not estimate_cost(inputs, sources, workers, 4)["within_ceiling"]
    sources[0]["seconds"]["lm_r128"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        estimate_cost(inputs, sources, workers, 4)


def test_phase_alarm_covers_work_outside_worker_loop():
    with pytest.raises(BudgetExceeded):
        with budget(.05):
            time.sleep(.5)
    with budget(.2):
        pass  # timer and signal handlers restored after an expired phase


def test_timeout_cleans_owned_group_and_leaves_unrelated_job(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    receipt = tmp_path / "jobs.json"
    try:
        with pytest.raises(BudgetExceeded):
            with budget(.3) as deadline:
                run_jobs([("owned", [sys.executable, "-c", "import time;time.sleep(30)"], [])],
                         os.environ.copy(), tmp_path / "logs", deadline, receipt)
        data = json.loads(receipt.read_text())
        assert data["status"] == "held_or_failed" and unrelated.poll() is None
        assert not Path(f"/proc/{data['workers']['owned']['pid']}").exists()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_failed_worker_stops_peer_and_records_completed_timing(tmp_path):
    receipt = tmp_path / "jobs.json"
    with pytest.raises(RuntimeError, match="exited 2"):
        with budget(5) as deadline:
            run_jobs([("failure", [sys.executable, "-c", "raise SystemExit(2)"], []),
                      ("peer", [sys.executable, "-c", "import time;time.sleep(30)"], [])],
                     os.environ.copy(), tmp_path / "logs", deadline, receipt)
    data = json.loads(receipt.read_text())
    assert data["workers"]["failure"]["exit_code"] == 2
    assert not Path(f"/proc/{data['workers']['peer']['pid']}").exists()


def test_successful_jobs_leave_phase_alarm_armed_for_reporting(tmp_path):
    with pytest.raises(BudgetExceeded):
        with budget(.6) as deadline:
            run_jobs([("fast", [sys.executable, "-c", "pass"], [])], os.environ.copy(),
                     tmp_path / "logs", deadline, tmp_path / "jobs.json")
            time.sleep(1)


def test_poll_timeout_shields_cleanup_from_later_alarm(tmp_path):
    # The monotonic deadline intentionally wins, then the alarm would fire
    # during cleanup. A TERM-ignoring worker requires the final KILL.
    receipt = tmp_path / "jobs.json"
    command = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
    try:
        with pytest.raises(BudgetExceeded):
            with budget(.7) as deadline:
                run_jobs([("stubborn", [sys.executable, "-c", command], [])], os.environ.copy(),
                         tmp_path / "logs", deadline - .4, receipt, cleanup_grace=.5)
        pid = json.loads(receipt.read_text())["workers"]["stubborn"]["pid"]
        assert not Path(f"/proc/{pid}").exists()
    finally:
        if receipt.exists():
            data = json.loads(receipt.read_text())
            pid = data.get("workers", {}).get("stubborn", {}).get("pid")
            if pid and Path(f"/proc/{pid}").exists():
                os.killpg(pid, 9)
