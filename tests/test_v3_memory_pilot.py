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
    expected = {"protocol": "memory-pilot-v2", "revision": "r", "config": {"a": 1}, "init": {"m": "x"}, "data_sources": {"mc": "h"}}
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


# ---------------------------------------------------------------------------- integration regressions (review of 3eedb89)
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

LIB = Path(__file__).resolve().parents[1] / "scripts" / "memory_pilot_lib.sh"


def run_branch(tmp_path, check_exit: int, existing: bool):
    """Source the runner's actual maybe_train with stub completion_check / run_training."""
    target = tmp_path / "train"
    if existing:
        target.mkdir()
        (target / "old.txt").write_text("previous attempt")
    script = f"""
set -euo pipefail
source "{LIB}"
completion_check() {{ return {check_exit}; }}
run_training() {{ echo trained > "{target}/marker"; }}
maybe_train "{target}"
"""
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return target, proc.stdout


def test_runner_trains_when_incomplete_and_skips_when_complete(tmp_path):
    for case in ("fresh", "incomplete", "complete"):
        (tmp_path / case).mkdir()
    target, out = run_branch(tmp_path / "fresh", 1, existing=False)
    assert (target / "marker").is_file() and "skipped" not in out  # fresh (check exits 1) -> trains
    target, out = run_branch(tmp_path / "incomplete", 1, existing=True)
    assert (target / "marker").is_file() and not (target / "old.txt").exists()  # incomplete -> kept aside, retrained
    assert list((tmp_path / "incomplete").glob("train.incomplete-*/old.txt"))
    target, out = run_branch(tmp_path / "complete", 0, existing=True)
    assert "skipped" in out and not (target / "marker").exists() and (target / "old.txt").is_file()  # complete -> skip


@pytest.mark.parametrize("steps", [2, 300])
def test_stage2_completion_with_the_real_memory_pilot_config(tmp_path, steps):
    import inspect

    from trace_vl.v3 import train_stage2
    from trace_vl.v3.memory_pilot import STAGE2_CHECKPOINT_STEPS

    cfg = load_v3_config("configs/v3/memory_pilot_b4.yaml")[0].to_dict()
    assert cfg["train"]["checkpoint_steps"] == [100, 300, 1000]  # inherited Stage-1 snapshot schedule
    out = tmp_path / "train"
    trained(out, steps=steps, record=False)
    with pytest.raises(FileNotFoundError):  # Stage-1 semantics unchanged: snapshots required by default
        write_completion(out, steps, cfg, IDENTITY)
    write_completion(out, steps, cfg, IDENTITY, checkpoint_steps=STAGE2_CHECKPOINT_STEPS)  # Stage-2 contract
    assert training_complete(out, cfg, steps, extra=IDENTITY, checkpoint_steps=STAGE2_CHECKPOINT_STEPS) == (True, "")
    assert "checkpoint_steps=()" in inspect.getsource(train_stage2.train_stage2)  # the trainer writes the same contract
    assert STAGE2_CHECKPOINT_STEPS == ()


def test_missing_prediction_in_a_present_arm_is_invalid_not_a_crash(tmp_path):
    base = [{"stream": "s0", "id": q, "evidence_age": 60.0, "correct": 1, "evidence_kept": 0.5, "kv_tokens": 100,
             "kv_mb": 1.0, "evicted_tokens": 5, "streamed_tokens": 50, "stream_seconds_elapsed": 1.0} for q in ("q0", "q1")]  # fmt: skip
    for name in EXPECTED_RUNS:
        write_run(tmp_path, name, base[:1] if name == "s2_mass" else base)
    r = memory_report(str(tmp_path), 16.0, expected_runs=EXPECTED_RUNS, expected_ids={"q0", "q1"})
    assert r["decision"]["status"] == "invalid" and r["paired"] == {}
    assert any("s2_mass" in reason for reason in r["decision"]["invalid_reasons"])
    assert not smoke_checks(r, [{"grad_norm_retention": 1.0, "ret_pos": 1, "ret_neg": 1}],
                            {"dev_selected": {"beyond_window": 5}})["expected_runs_complete"]


# ---------------------------------------------------------------------------- smoke selection (A100 smoke data-gate stop)
def selection_streams():
    """Early short streams with only immediate questions, then long streams with delayed questions, plus a test-split
    stream (must never be selected from dev) and a long stream whose delayed question is clipped at its end."""
    rows = []
    for i in range(3):  # short leftovers: 60 s, question right after the clip (age 0)
        rows.append({"id": f"short{i}", "split": "dev", "duration": 60.0, "clips": [],
                     "questions": [{"id": f"short{i}-q", "time": 50.0, "evidence": [[20.0, 50.0]], "delay": 0.0}]})
    for i in range(4):  # eligible: 300 s, evidence ends at 100 s, asked at 200 s (age 100 > 16)
        rows.append({"id": f"long{i}", "split": "dev", "duration": 300.0, "clips": [],
                     "questions": [{"id": f"long{i}-q", "time": 200.0, "evidence": [[60.0, 100.0]], "delay": 100.0}]})
    rows.append({"id": "clipped", "split": "dev", "duration": 300.0, "clips": [],  # nominal delay 120 s, actual age 5 s
                 "questions": [{"id": "clipped-q", "time": 299.0, "evidence": [[260.0, 294.0]], "delay": 120.0}]})
    rows.append({"id": "testlong", "split": "test", "duration": 300.0, "clips": [],
                 "questions": [{"id": "t-q", "time": 200.0, "evidence": [[60.0, 100.0]], "delay": 100.0}]})
    return rows


def test_smoke_selection_is_deterministic_eligible_and_split_isolated(tmp_path):
    from trace_vl.v3.memory_pilot import PILOT_RULE, SMOKE_RULE, data_report, select_streams

    cfg = load_v3_config("configs/v3/memory_pilot_b4.yaml")[0]
    src = tmp_path / "dev.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in selection_streams()))
    a = select_streams(src, "dev", cfg, SMOKE_RULE, 2, tmp_path / "a.jsonl")
    b = select_streams(src, "dev", cfg, SMOKE_RULE, 2, tmp_path / "b.jsonl")
    ids = [x["id"] for x in a["selected"]]
    assert ids == [x["id"] for x in b["selected"]] and a["sha256"] == b["sha256"]  # deterministic
    assert set(ids) <= {"long0", "long1", "long2", "long3"} and len(ids) == 2  # never short or clipped or test
    assert a["candidates"] == 8 and a["eligible"] == 4  # the test-split stream is not even a candidate
    assert a["excluded"] == {"too_short": 3, "no_beyond_window_question": 4}  # 3 short (age 0) + clipped (age 5)
    assert all(age > 16 for x in a["selected"] for age in x["ages"])
    pilot = select_streams(src, "dev", cfg, PILOT_RULE, 2, tmp_path / "p.jsonl")
    assert [x["id"] for x in pilot["selected"]] == ["short0", "short1"]  # pilot population unchanged: first n
    (tmp_path / "streams").mkdir()
    for split in ("train", "dev", "test"):
        (tmp_path / "streams" / f"{split}.jsonl").write_text("")
    (tmp_path / "m.json").write_text(json.dumps({"splits": {}}))
    rep = data_report(tmp_path / "streams", tmp_path / "m.json", None, 16.0, cfg, dev_path=tmp_path / "a.jsonl",
                      train_path=tmp_path / "a.jsonl")  # fmt: skip
    assert rep["dev_selected"]["streams"] == 2 and rep["dev_selected"]["beyond_window"] == 2
    assert not data_problems(rep, "smoke")


def test_too_few_eligible_streams_stop_before_training(tmp_path):
    from trace_vl.v3.memory_pilot import SMOKE_RULE, data_report, select_streams

    cfg = load_v3_config("configs/v3/memory_pilot_b4.yaml")[0]
    rows = [r for r in selection_streams() if r["id"] in ("short0", "short1", "clipped", "long0")]
    src = tmp_path / "dev.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sel = select_streams(src, "dev", cfg, SMOKE_RULE, 2, tmp_path / "sel.jsonl")
    assert sel["eligible"] == 1 and sel["selected_count"] == 1
    (tmp_path / "streams").mkdir()
    for split in ("train", "dev", "test"):
        (tmp_path / "streams" / f"{split}.jsonl").write_text("")
    (tmp_path / "m.json").write_text(json.dumps({"splits": {}}))
    rep = data_report(tmp_path / "streams", tmp_path / "m.json", None, 16.0, cfg, dev_path=tmp_path / "sel.jsonl",
                      train_path=tmp_path / "sel.jsonl")  # fmt: skip
    problems = data_problems(rep, "smoke")
    assert any("selected dev streams 1 < 2" in p for p in problems) and any("usable train streams 1 < 2" in p for p in problems)


# ---------------------------------------------------------------------------- dev-only expansion (pilot dev streams 25 < 30)
def build_old_split(tmp_path):
    rows = [{"id": f"q{v}-{k}", "video": f"v{v}.mp4", "question": "?", "options": ["a", "b"], "answer": 0}
            for v in range(600) for k in range(2)]  # fmt: skip
    (tmp_path / "mc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "s1.jsonl").write_text("".join(json.dumps({"video": f"v{v}.mp4"}) + "\n" for v in range(20)))
    make_splits([str(tmp_path / "mc.jsonl")], [str(tmp_path / "s1.jsonl")], str(tmp_path / "old"),
                max_videos={"train": 200, "dev": 10, "test": 20})  # fmt: skip
    return [str(tmp_path / "mc.jsonl")], [str(tmp_path / "s1.jsonl")]


def test_dev_expansion_keeps_train_and_test_byte_identical(tmp_path):
    from trace_vl.v3.memory_pilot import expand_dev_split

    records, exclude = build_old_split(tmp_path)
    m = expand_dev_split(records, exclude, tmp_path / "old", tmp_path / "new")
    for split in ("train", "test"):
        assert (tmp_path / "new" / f"{split}.jsonl").read_bytes() == (tmp_path / "old" / f"{split}.jsonl").read_bytes()
    old_dev = {json.loads(x)["video"] for x in (tmp_path / "old" / "dev.jsonl").read_text().splitlines()}
    new_dev = {json.loads(x)["video"] for x in (tmp_path / "new" / "dev.jsonl").read_text().splitlines()}
    assert len(new_dev) == m["dev_available_videos"] > len(old_dev) == 10  # the reproduced shortage: capped dev
    train_test = {json.loads(x)["video"] for s in ("train", "test") for x in (tmp_path / "new" / f"{s}.jsonl").read_text().splitlines()}
    assert not new_dev & train_test and all(int(v[1:-4]) >= 20 for v in new_dev)  # no leakage, Stage-1 excluded
    small = expand_dev_split(records, exclude, tmp_path / "old", tmp_path / "small", dev_cap=15)
    small_dev = {json.loads(x)["video"] for x in (tmp_path / "small" / "dev.jsonl").read_text().splitlines()}
    assert len(small_dev) == 15 and small_dev <= new_dev  # prefix rule: a larger cap only adds videos
    again = expand_dev_split(records, exclude, tmp_path / "old", tmp_path / "again", dev_cap=15)
    assert again["splits"]["dev"]["sha256"] == small["splits"]["dev"]["sha256"]  # deterministic
    assert m["previous"]["dev"]["videos"] == 10 and m["dev_rule"] == "dev-hash-prefix-v1"


def test_dev_expansion_refuses_tampered_or_mismatched_reuse(tmp_path):
    from trace_vl.v3.memory_pilot import expand_dev_split

    records, exclude = build_old_split(tmp_path)
    with pytest.raises(ValueError, match="seed or fractions"):
        expand_dev_split(records, exclude, tmp_path / "old", tmp_path / "x", seed=8)
    (tmp_path / "old" / "test.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="recorded hash"):
        expand_dev_split(records, exclude, tmp_path / "old", tmp_path / "y")


def test_pilot_dev_shortage_is_rejected_and_stale_prep_refused(tmp_path):
    from trace_vl.v3.memory_pilot import check_prep_request

    rep = {"streams": {"train": {"streams": 164}}, "train_used": {"streams": 164},
           "dev_selected": {"streams": 25, "beyond_window": 300}}  # fmt: skip
    assert data_problems(rep, "pilot") == ["selected dev streams 25 < 30"]  # the A100 pilot stop, unchanged gate
    request = {"mode": "pilot", "caps": "3000 400 400", "reuse": "/old/data", "dev_cap": None}
    check_prep_request(tmp_path / "data" / "prep-request.json", request)
    check_prep_request(tmp_path / "data" / "prep-request.json", request)  # same request: allowed (resume)
    with pytest.raises(ValueError, match="new output directory"):
        check_prep_request(tmp_path / "data" / "prep-request.json", {**request, "dev_cap": 400})
