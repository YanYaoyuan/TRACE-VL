"""Stage-2 memory pilot (CPU): frozen source-video splits, training completion/identity, eviction-schedule parity,
three-state decision, data sufficiency and machine-checked smoke authorization (PR #3 review of 958424e)."""

import json

import pytest

from trace_vl.v3.config import V3Config, load_v3_config
from trace_vl.v3.memory_pilot import (
    EXPECTED_RUNS,
    data_problems,
    expected_questions,
    make_splits,
    memory_report,
    smoke_checks,
    verify_smoke,
)
from trace_vl.v3.pilot import training_complete, write_completion
from trace_vl.v3.train_stage2 import eviction_steps


# ---------------------------------------------------------------------------- splits
def test_splits_exclude_stage1_videos_and_are_video_disjoint(tmp_path):
    rows = [{"id": f"q{v}-{k}", "video": f"v{v}.mp4", "question": "?", "options": ["a", "b"], "answer": 0}
            for v in range(200) for k in range(2)]  # fmt: skip
    (tmp_path / "mc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "s1.jsonl").write_text("".join(json.dumps({"video": f"v{v}.mp4"}) + "\n" for v in range(20)))
    m = make_splits([str(tmp_path / "mc.jsonl")], [str(tmp_path / "s1.jsonl")], str(tmp_path / "out"),
                    max_videos={"train": 50, "dev": 10, "test": 10})  # fmt: skip
    split = {s: [json.loads(x) for x in (tmp_path / "out" / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "dev", "test")}
    videos = {s: {r["video"] for r in rows} for s, rows in split.items()}
    assert not (videos["train"] & videos["dev"]) and not (videos["train"] & videos["test"]) and not (videos["dev"] & videos["test"])
    assert all(int(v[1:-4]) >= 20 for vs in videos.values() for v in vs)  # Stage-1 videos excluded
    assert len(videos["train"]) == 50 and len(videos["dev"]) <= 10 and m["excluded_videos"] == 20


# ---------------------------------------------------------------------------- completion and identity
CONFIG = {"train": {"steps": 3}, "stage2": {"steps": 3}, "lr": 1e-4}
IDENTITY = {"revision": "r1", "streams_sha256": "s1", "media": "m1", "init": {"modules.pt": "i1"}, "steps": 3}


def trained(out, steps=3, weights=True, record=True, extra=IDENTITY):
    (out / "lora").mkdir(parents=True)
    (out / "modules.pt").write_bytes(b"modules-final")
    (out / "lora" / "adapter_config.json").write_text("{}")
    if weights:
        (out / "lora" / "adapter_model.safetensors").write_bytes(b"lora-final")
    (out / "metrics.jsonl").write_text("".join(json.dumps({"step": k}) + "\n" for k in range(1, steps + 1)))
    if record:
        write_completion(out, steps, CONFIG, extra)


def test_stage2_completion_rejects_invalid_or_stale_outputs(tmp_path):
    good = tmp_path / "good"
    trained(good)
    assert training_complete(good, CONFIG, 3, extra=IDENTITY) == (True, "")  # valid unchanged rerun skips
    interrupted = tmp_path / "interrupted"
    trained(interrupted, record=False)  # final save interrupted before the record
    assert not training_complete(interrupted, CONFIG, 3, extra=IDENTITY)[0]
    with pytest.raises(FileNotFoundError):
        trained(tmp_path / "noweights", weights=False)  # a completion record cannot be written without weights
    corrupt = tmp_path / "corrupt"
    trained(corrupt)
    (corrupt / "modules.pt").write_bytes(b"invalid checkpoint")  # corrupt / replaced module
    assert "differ" in training_complete(corrupt, CONFIG, 3, extra=IDENTITY)[1]
    stale = tmp_path / "stale"
    trained(stale)
    (stale / "lora" / "adapter_model.safetensors").write_bytes(b"lora-step-200")  # stale adapter
    assert not training_complete(stale, CONFIG, 3, extra=IDENTITY)[0]
    assert not training_complete(good, {**CONFIG, "lr": 9e-4}, 3, extra=IDENTITY)[0]  # changed learning rate
    for key, value in (("streams_sha256", "other"), ("init", {"modules.pt": "other"}), ("revision", "r2")):
        assert not training_complete(good, CONFIG, 3, extra={**IDENTITY, key: value})[0]  # changed streams / init / code
    assert not training_complete(good, CONFIG, 4, extra=IDENTITY)[0]
    (good / "metrics.jsonl").write_text("".join(json.dumps({"step": 1}) + "\n" for _ in range(3)))  # 3 rows, not 1..3
    assert not training_complete(good, CONFIG, 3, extra=IDENTITY)[0]


# ---------------------------------------------------------------------------- eviction schedule
def test_train_schedule_matches_the_trainers_chunk_ends_including_question_cuts():
    times = [i + 0.25 for i in range(24)]
    steps = eviction_steps(times, [{"id": "q", "time": 3.5}], 8, 0.5)
    assert sorted(i + 1 for i in steps) == [4, 12, 20, 24]  # the review's example (1-based)
    assert sorted(i + 1 for i in eviction_steps(times, [], 8, 0.5)) == [8, 16, 24]  # = evict_every 8 only without questions


# ---------------------------------------------------------------------------- three-state decision
def write_run(root, name, rows):
    (root / name).mkdir(parents=True)
    (root / name / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def rows_for(correct, kept, kv=1700, evicted=11000, beyond=True):
    out = []
    for s in range(30):
        for q in range(5):
            age = 5.0 if (q == 0 or not beyond) else 60.0 + q
            out.append({"stream": f"s{s}", "id": f"s{s}-q{q}", "delay": age, "evidence_age": age,
                        "correct": int(correct(s, q)), "evidence_kept": 1.0 if age <= 16 else kept,
                        "kv_tokens": kv, "kv_mb": 3.0, "evicted_tokens": evicted, "streamed_tokens": 13000,
                        "stream_seconds_elapsed": 20.0})  # fmt: skip
    return out


def all_runs(root, tkm, mass, fifo, **kw):
    for name in EXPECTED_RUNS:
        fn = {"s2_tkm": tkm, "s2_mass": mass, "s2_fifo": fifo}.get(name, mass)
        write_run(root, name, rows_for(fn[0], fn[1], **kw))


def test_supported_negative_and_invalid_are_distinguished(tmp_path):
    good, bad = (lambda s, q: q != 4, 0.7), (lambda s, q: q in (0, 1), 0.3)
    all_runs(tmp_path / "a", good, (lambda s, q: q in (0, 1), 0.5), (lambda s, q: q in (0, 1, 2), 0.3))
    assert memory_report(str(tmp_path / "a"), 16.0, expected_runs=EXPECTED_RUNS)["decision"]["status"] == "supported"
    all_runs(tmp_path / "b", bad, bad, bad)  # valid test, no gain
    d = memory_report(str(tmp_path / "b"), 16.0, expected_runs=EXPECTED_RUNS)["decision"]
    assert d["status"] == "negative" and "not a contribution" in d["meaning"]
    all_runs(tmp_path / "c", good, bad, bad, evicted=0, beyond=False)  # no delayed questions, no evictions
    d = memory_report(str(tmp_path / "c"), 16.0, expected_runs=EXPECTED_RUNS)["decision"]
    assert d["status"] == "invalid" and "inconclusive" in d["meaning"] and "not a contribution" not in d["meaning"]
    assert "enough_beyond_window" in d["invalid_reasons"] and "evictions" in d["invalid_reasons"]
    all_runs(tmp_path / "d", good, bad, bad)
    import shutil

    shutil.rmtree(tmp_path / "d" / "s1_full")
    write_run(tmp_path / "d", "unrelated", rows_for(*bad))  # an unrelated directory cannot stand in for a missing arm
    d = memory_report(str(tmp_path / "d"), 16.0, expected_runs=EXPECTED_RUNS)["decision"]
    assert d["status"] == "invalid" and "missing run s1_full" in d["invalid_reasons"]
    all_runs(tmp_path / "e", good, bad, bad)  # incomplete coverage against the expected question set
    ids = {f"s{s}-q{q}" for s in range(31) for q in range(5)}
    assert memory_report(str(tmp_path / "e"), 16.0, expected_runs=EXPECTED_RUNS, expected_ids=ids)["decision"]["status"] == "invalid"


# ---------------------------------------------------------------------------- data sufficiency
def test_actual_evidence_age_counts_clipped_questions_as_within_window(tmp_path):
    cfg = load_v3_config("configs/v3/memory_pilot_b4.yaml")[0]
    stream = {"id": "s", "split": "dev", "duration": 100.0, "clips": [],
              "questions": [{"id": "far", "time": 90.0, "evidence": [[10.0, 30.0]], "delay": 60.0},
                            {"id": "clipped", "time": 99.0, "evidence": [[90.0, 95.0]], "delay": 4.0}]}  # fmt: skip
    (tmp_path / "dev.jsonl").write_text(json.dumps(stream) + "\n")
    ages = expected_questions(tmp_path / "dev.jsonl", "dev", 1, cfg)
    assert ages == {"far": 60.0, "clipped": 4.0}  # a nominally delayed question clipped at the stream end is within the window
    rep = {"streams": {"train": {"streams": 10}}, "dev_selected": {"streams": 40, "beyond_window": 1}}
    assert data_problems(rep, "pilot") and not data_problems({**rep, "dev_selected": {"streams": 2, "beyond_window": 2}}, "smoke")


# ---------------------------------------------------------------------------- smoke authorization
def smoke_report(evicted=10.0):
    return {"runs": {name: {"evicted_tokens_mean": evicted} for name in EXPECTED_RUNS}, "decision": {"invalid_reasons": []}}


def test_smoke_requires_a_real_retention_signal():
    data = {"dev_selected": {"beyond_window": 5}}
    ok = [{"grad_norm_retention": 0.2, "update_norm_retention": 1e-4, "ret_pos": 3, "ret_neg": 9}]
    assert all(smoke_checks(smoke_report(), ok, data).values())
    decay_only = [{"grad_norm_retention": 0.0, "update_norm_retention": 1e-6, "ret_pos": 0, "ret_neg": 9}]
    checks = smoke_checks(smoke_report(), decay_only, data)  # weight decay moves weights without a learning signal
    assert not checks["retention_gradient_finite_nonzero"] and not checks["retention_labels_both_classes"]
    assert not smoke_checks(smoke_report(evicted=0.0), ok, data)["evictions"]
    report = smoke_report()
    report["runs"].pop("s2_full")
    assert not smoke_checks(report, ok, data)["expected_runs_complete"]


def test_pilot_authorization_requires_a_matching_passed_smoke(tmp_path):
    expected = {"revision": "r", "config": {"a": 1}, "init": {"m": "x"}, "data_sources": {"mc": "h"}}
    path = tmp_path / "smoke-ok.json"
    assert verify_smoke(path, expected)  # missing
    path.write_text("")
    assert verify_smoke(path, expected)  # empty
    path.write_text(json.dumps({"passed": False, **expected}))
    assert verify_smoke(path, expected) == ["smoke did not pass"]
    path.write_text(json.dumps({"passed": True, **expected}))
    assert verify_smoke(path, expected) == []
    for key, value in (("init", {"m": "other"}), ("data_sources", {"mc": "other"}), ("revision", "old")):
        path.write_text(json.dumps({"passed": True, **{**expected, key: value}}))
        assert verify_smoke(path, expected) == [f"smoke {key} differs from the intended pilot"]


def test_memory_pilot_config_loads_and_evict_every_is_validated():
    cfg, _ = load_v3_config("configs/v3/memory_pilot_b4.yaml")
    assert cfg.transition.p_step_mode == "pooled" and cfg.memory.policy == "tkm" and cfg.memory.budget_tokens == 1024
    with pytest.raises(ValueError, match="evict_every"):
        V3Config.from_dict({"memory": {"evict_every": 0}})
