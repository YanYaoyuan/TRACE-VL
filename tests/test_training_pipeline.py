import json
from collections import OrderedDict
from threading import Event

import pytest
import torch

from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.prepare import digest, write_json
from trace_vl.longstream_training.cache import LEGACY_CACHE_CODE, SEMANTIC_KEYS, reuse_entry
from trace_vl.longstream_training.pipeline import ordered_parallel, prefetch
from trace_vl.longstream_training.train import gradient_norm


def test_parallel_probe_runs_ahead_but_preserves_source_order():
    ahead = Event()

    def work(i):
        if i == 0:
            assert ahead.wait(2)
        else:
            ahead.set()
        return i

    assert list(ordered_parallel(work, range(6), workers=2, depth=3)) == list(range(6))


def test_prefetch_bounds_residency_propagates_failure_and_closes_decoder():
    closed, produced = Event(), []

    def source():
        try:
            for i in range(1000):
                produced.append(i)
                yield i
        finally:
            closed.set()

    with prefetch(source, depth=2) as iterator:
        assert next(iterator) == 0
    assert closed.is_set() and len(produced) <= 5

    def broken():
        yield 1
        raise ValueError("decoder error")

    with prefetch(broken) as iterator:
        assert next(iterator) == 1
        with pytest.raises(ValueError, match="decoder error"):
            next(iterator)


def test_legacy_cache_reuse_preserves_all_tensor_and_causal_metadata(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    destination = tmp_path / "new.pt"
    identity = {k: k for k in SEMANTIC_KEYS}
    old = {**identity, "code_sha256": LEGACY_CACHE_CODE}
    old["id"] = digest(old)
    write_json(source / "inputs.json", old)
    identity.update(code_sha256="new-execution-code", id="new-input-id")
    row = {"id": "question"}
    entry = {
        "input_id": old["id"],
        "id": row["id"],
        "features": torch.randn(36, 16, 8),
        "times": [[0.0, 0.5], [1.0, 1.5]],
        "rgb_sha256": ["a", "b"],
        "indices": [[0, 1], [2, 3]],
    }
    path = source / f"{digest(row['id'])}.pt"
    torch.save(entry, path)
    write_json(path.with_suffix(".json"), {"input_id": old["id"], "id": row["id"], "sha256": sha256(path)})
    assert reuse_entry(source, destination, row, identity)
    result = torch.load(destination, weights_only=True)
    assert result.pop("input_id") == identity["id"]
    entry.pop("input_id")
    assert torch.equal(result.pop("features"), entry.pop("features")) and result == entry
    assert json.loads(destination.with_suffix(".json").read_text())["reused_from"]["input_id"] == old["id"]
    with pytest.raises(ValueError, match="semantics"):
        reuse_entry(source, destination, row, {**identity, "sampling_hz": 4})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        reuse_entry(source, destination, row, identity)


def test_gradient_logging_preserves_gradients_and_matches_original_reduction():
    parameters = [torch.nn.Parameter(torch.randn(4, 7)) for _ in range(4)]
    for p in parameters:
        p.grad = torch.randn_like(p)
    before = [p.grad.clone() for p in parameters]
    expected = sum(float(p.grad.float().square().sum()) for p in parameters) ** 0.5
    assert gradient_norm(parameters) == pytest.approx(expected, rel=1e-12)
    assert all(torch.equal(p.grad, g) for p, g in zip(parameters, before, strict=True))


def test_layout_cache_is_answer_blind_and_recomputes_for_new_observations(monkeypatch):
    from test_core_vlm_native import tiny_example

    from trace_vl.core_vlm.native import NativeLayout
    from trace_vl.longstream_training import train

    bb, _, _ = tiny_example()
    entry = {
        "features": torch.randn(8, 16, 32),
        "times": [[i, i + 0.5] for i in range(8)],
        "indices": [[2 * i, 2 * i + 1] for i in range(8)],
        "hw": [64, 64],
        "grid": [4, 4],
        "fps": 2.0,
        "frame_count": 100,
    }
    row = {"question": "What happened?", "options": ["a", "b"], "query_time": 20.0, "answer": 0}
    calls = []

    def prepare(processor, images, times, record, audit, **kwargs):
        calls.append(record)
        return NativeLayout({"pixel_values_videos": torch.zeros(100)}, times.tolist(), "prompt", audit)

    monkeypatch.setattr(train, "prepare_native", prepare)
    layouts = OrderedDict()
    a, record, indices = train.sample(bb, None, entry, row, layouts)
    b, changed, indices2 = train.sample(bb, None, entry, {**row, "answer": 1}, layouts)
    assert len(calls) == 1 and calls[0]["answer"] == 0
    assert record["native_layout"] is changed["native_layout"] and changed["answer"] == 1
    assert "pixel_values_videos" not in record["native_layout"].inputs
    assert indices == indices2 and all(torch.equal(x.tokens, y.tokens) for x, y in zip(a, b, strict=True))
    train.sample(bb, None, {**entry, "times": [[i + 0.1, i + 0.6] for i in range(8)]}, row, layouts)
    assert len(calls) == 2
