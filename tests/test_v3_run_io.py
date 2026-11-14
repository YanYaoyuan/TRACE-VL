import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest


def _worker(args):
    path, identity = args
    for _ in range(20):
        ensure_run_manifest(Path(path), identity)
        assert json.loads(Path(path).read_text()) == identity
    return True


def test_four_processes_publish_and_read_one_manifest(tmp_path):
    identity = {"id": "same-run", "configuration": "large-manifest" * 10000}
    path = tmp_path / "run.json"
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert all(pool.map(_worker, [(str(path), identity)] * 4))
    before = path.stat().st_mtime_ns
    ensure_run_manifest(path, identity)
    assert path.stat().st_mtime_ns == before  # a resume must never rewrite/truncate it


@pytest.mark.parametrize("existing", ["", "{", "null", '{"id":"other"}', '{"id":"same","config":2}'])
def test_corrupt_or_conflicting_identity_is_preserved(tmp_path, existing):
    path = tmp_path / "run.json"
    path.write_text(existing)
    with pytest.raises(ValueError):
        ensure_run_manifest(path, {"id": "same", "config": 1})
    assert path.read_text() == existing


def test_failed_publish_keeps_previous_file_and_cleans_only_own_temp(tmp_path, monkeypatch):
    path = tmp_path / "predictions.jsonl"
    path.write_text("original\n")
    other = tmp_path / "other-worker.tmp"
    other.write_text("being written")

    def fail(*args):
        raise OSError("injected rename failure")

    monkeypatch.setattr("trace_vl.v3.run_io.os.replace", fail)
    with pytest.raises(OSError):
        atomic_text(path, "replacement\n")
    assert path.read_text() == "original\n"
    assert other.read_text() == "being written"
    assert not list(tmp_path.glob(".predictions.jsonl.*.tmp"))


def test_concurrent_evaluator_resume_preserves_active_unit_and_valid_reports(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from trace_vl.v3 import harness
    from trace_vl.v3.config import V3Config
    from trace_vl.v3.train_stage2 import evaluate_streams

    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"id": "stream1", "video": "video.mp4", "split": "val", "duration": 3,
                                   "questions": [{"id": "q1", "time": 2}]}) + "\n")
    out = tmp_path / "eval"
    row = {"id": "q1", "stream": "stream1", "correct": 1, "delay": 1, "kv_tokens": 32, "run": "same"}
    harness.write_unit(out / "streams", "stream1", [row], "same")
    active = out / "streams" / "another-worker.tmp"
    active.write_text("unpublished")
    monkeypatch.setattr(harness, "run_fingerprint", lambda *args, **kwargs: {"id": "same"})

    def run(index):
        return evaluate_streams(V3Config(), str(records), str(tmp_path), str(out), mode="native", shard=(index, 4))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert all(r["accuracy"] == 1 for r in results)
    assert active.read_text() == "unpublished"
    assert json.loads((out / "run.json").read_text()) == {"id": "same"}
    assert json.loads((out / "predictions.jsonl").read_text()) == row
    assert json.loads((out / "summary.json").read_text())["questions"] == 1
