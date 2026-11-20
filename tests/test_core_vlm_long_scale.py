import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from trace_vl.core_vlm import batch_benchmark, long_scale
from trace_vl.core_vlm.train import Samples


def row(id_, path, byte_hash, group, question="Q", split="train"):
    return {"id": id_, "video": path, "media_sha256": byte_hash, "source_group": group,
            "source_aliases": [group], "question": question, "options": ["a", "b"],
            "answer": 0, "split": split}


def test_expansion_prefers_unseen_sources_and_deduplicates_questions():
    baseline = [row("old", "old.mp4", "o", "old-source")]
    fresh = [
        row("a", "a.mp4", "a", "a-source"),
        row("b", "b.mp4", "b", "b-source"),
        row("c", "c.mp4", "c", "old-source"),
        row("clone", "clone.mp4", "o", "renamed"),
        row("held-out", "held-out.mp4", "v", "val-source", split="val"),
    ]
    unused = [row("old-clone", "old.mp4", "o", "old-source"),
              row("old-new-question", "old.mp4", "o", "old-source", question="New question?")]
    added = long_scale.added_questions(baseline, {"old.mp4": "o"}, fresh, unused, 3, 17)
    assert {r["id"] for r in added[:2]} == {"a", "b"}
    assert added[2]["id"] == "old-new-question"
    assert long_scale.added_questions(baseline, {"old.mp4": "o"}, reversed(fresh), unused, 3, 17) == added
    with pytest.raises(ValueError, match="only 3 additional"):
        long_scale.added_questions(baseline, {"old.mp4": "o"}, fresh, unused, 4, 17)


def test_global_32_prefers_fastest_layout_passing_both_arms():
    trials = {
        f"{arm}-{layout}": {"passed": True, "question_ids_by_update": [["fixed"]],
                            "initial_lora": "same", "first_loss": 1.,
                            "peak_reserved_gib_max": 14., "median_seconds_per_update": t}
        for arm in ("plain", "temporal")
        for layout, t in zip(long_scale.LAYOUTS, (22., 17., 15.), strict=True)
    }
    recommendation = batch_benchmark.recommend(trials, layouts=long_scale.LAYOUTS, per_rank=8)
    assert (recommendation["microbatch_size"], recommendation["accumulation"],
            recommendation["decode_workers"], recommendation["effective_batch_size"]) == (8, 1, 4, 32)
    trials["temporal-mb8-parallel"]["peak_reserved_gib_max"] = 35.
    recommendation = batch_benchmark.recommend(trials, layouts=long_scale.LAYOUTS, per_rank=8)
    assert recommendation["layout"] == "mb4-parallel" and recommendation["accumulation"] == 2


def test_parallel_decode_preserves_order_and_uses_cache(monkeypatch):
    samples = Samples(None, SimpleNamespace(decode_workers=4))
    samples.cache = OrderedDict(cached="cached-payload")
    decoded = []
    monkeypatch.setattr(samples, "decode", lambda r: decoded.append(r["id"]) or r["id"].upper())
    monkeypatch.setattr(samples, "get", lambda r, payload=None: (
        r["id"], samples.cache.get(r["id"], payload)))
    ids = ["b", "cached", "a", "a"]
    assert samples.batch([{"id": k} for k in ids]) == [
        ("b", "B"), ("cached", "cached-payload"), ("a", "A"), ("a", "A")]
    assert sorted(decoded) == ["a", "b"]


def test_prepare_only_does_not_start_gpu_and_queue_freezes_batch(tmp_path, monkeypatch):
    out = tmp_path / "run"
    cfg = {"gpus": [0, 1, 2, 3], "minimum_free_gib": 30}
    monkeypatch.setattr(long_scale, "prepare", lambda *a: dict(cfg))
    work = []
    monkeypatch.setattr(long_scale, "run_jobs", lambda *a: work.append("gpu-check"))
    monkeypatch.setattr(long_scale, "run_training", lambda *a: work.append(a[0]))

    def benchmark(*args, **kwargs):
        work.append("benchmark")
        assert kwargs["per_rank"] == 8
        rec = dict(layout="mb8-parallel", candidate_seconds={"mb8-parallel": 34.0},
                   microbatch_size=8, accumulation=1, decode_workers=4)
        args[1].mkdir()
        (args[1] / "batch-report.json").write_text(json.dumps(rec))
        return rec

    monkeypatch.setattr(batch_benchmark, "run", benchmark)
    argv = ["long-scale", "--previous", str(tmp_path), "--out", str(out)]
    monkeypatch.setattr(long_scale.sys, "argv", [*argv, "--prepare-only"])
    long_scale.main()
    assert not work
    monkeypatch.setattr(long_scale.sys, "argv", argv)
    long_scale.main()
    long_scale.main()
    assert work.count("benchmark") == 1
    assert work[2]["microbatch_size"] == 8 and work[2]["decode_workers"] == 4
    assert json.loads((out / "long-scale-plan.json").read_text())["presentations_per_arm"] == 160_000
