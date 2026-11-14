"""Recent-window control (experiments.md §14.4): step selection, fresh-state isolation, checks and report."""

from __future__ import annotations

import pytest

from trace_vl.v3.data import write_jsonl
from trace_vl.v3.ovo_transfer import BACKWARD
from trace_vl.v3.recent_window import (
    CHECKPOINTS,
    CONTROLS,
    NEW_ARMS,
    answer_fresh,
    check_arms,
    contrast,
    flow_check,
    recent_window_report,
    run_fresh_stream,
    window_steps,
)
from trace_vl.v3.train_stage2 import chunk_plan, stream_times

GAP = 0.5


class _Memory(list):
    evicted_tokens = 0


class _Session:
    """Deterministic fake: the answer depends only on the steps this session was fed. `leak` is state shared by
    every session of one factory (what a leaking backbone or module cache would be)."""

    def __init__(self, leak: list | None = None):
        self.fed, self.leak, self.step_index, self.memory = [], leak, 0, _Memory()

    def step(self, frames, t, pixels=None):
        assert frames is None and pixels == ("px", t)  # preprocessed pixels of exactly this step
        self.fed.append(t)
        self.memory.append(t)
        self.step_index += 1
        if self.leak is not None:
            self.leak.append(t)

    def ask(self, question, options):
        base = sum(self.fed) + 10 * len(self.fed) + (len(self.leak) if self.leak is not None else 0)
        scores = [-((base + 7 * i) % 5) - 0.25 * i for i in range(len(options))]
        return max(range(len(options)), key=lambda i: scores[i]), scores


def _stream(duration=40.0, q_times=(3.0, 12.6, 30.0, 39.9)):
    times = stream_times({"duration": duration}, 1.0, GAP, 600.0)
    questions = [{"id": f"q{i}", "time": t, "question": "?", "options": ["a", "b", "c", "d"], "answer": 0}
                 for i, t in enumerate(q_times)]  # fmt: skip
    due = {steps[-1]: qs for steps, qs in chunk_plan(times, questions, len(times), GAP) if qs}
    return times, questions, due


def _run(times, due, window, leak=None, isolation=0, reads=None):
    def read_step(t):
        if reads is not None:
            reads.append(t)
        return ("px", t)

    return run_fresh_stream(times, due, window, GAP, read_step, lambda: _Session(leak), isolation)


def test_window_steps_end_at_the_answering_step_and_shrink_at_the_start():
    assert window_steps(20, 16) == list(range(5, 21))
    assert window_steps(2, 16) == [0, 1, 2] and window_steps(0, 4) == [0]
    assert window_steps(9, 4) == [6, 7, 8, 9]


def test_fresh_rows_use_the_last_w_steps_of_the_global_grid_without_future_frames():
    times, questions, due = _stream()
    reads = []
    rows = {r["id"]: r for r in _run(times, due, 16, reads=reads)}
    assert set(rows) == {q["id"] for q in questions}
    for q in questions:
        r = rows[q["id"]]
        k = max(i for i, t in enumerate(times) if t + GAP / 2 <= q["time"] + 1e-6)  # the §14.3 answering step
        assert r["t_step"] == times[k] and r["step_window_end"] <= q["time"] + 1e-6
        assert r["window_steps"] == min(16, k + 1) == r["session_steps"]
        assert r["first_step_t"] == times[max(0, k - 15)]
        assert r["evicted_tokens"] == 0 and r["kv_appended_tokens"] == r["kv_tokens"]
    assert rows["q0"]["window_steps"] == 3 and rows["q2"]["window_steps"] == 16  # short prefix vs full window
    assert reads == times[: max(due) + 1]  # every step decoded once, in order, never past the last query
    assert {r["window_steps"] for r in _run(times, due, 4)} == {3, 4}


def test_each_question_gets_an_independent_session_order_and_neighbours_do_not_matter():
    times, questions, due = _stream()
    together = {r["id"]: (r["choice"], r["scores"]) for r in _run(times, due, 16)}
    for q in questions:  # every question alone on the stream == the question among the others
        alone = {k: [x for x in qs if x["id"] == q["id"]] for k, qs in due.items()}
        alone = {k: qs for k, qs in alone.items() if qs}
        row = _run(times, alone, 16)[0]
        assert (row["choice"], row["scores"]) == together[q["id"]]
        k = next(iter(alone))  # and == an independently built fresh session on the same steps
        steps = [(times[i], ("px", times[i])) for i in window_steps(k, 16)]
        independent = answer_fresh(lambda: _Session(), steps, q)
        assert (independent["choice"], independent["scores"]) == together[q["id"]]
    # two questions at the same step, both orders
    same = [{**questions[2], "id": "a"}, {**questions[2], "id": "b", "options": ["x", "y"]}]
    k = 29
    ab = {r["id"]: r["scores"] for r in _run(times, {k: same}, 16)}
    ba = {r["id"]: r["scores"] for r in _run(times, {k: same[::-1]}, 16)}
    assert ab == ba


def test_aba_isolation_check_is_clean_without_shared_state_and_catches_a_leak():
    times, _, due = _stream()
    clean = _run(times, due, 16, isolation=2)
    checked = [r for r in clean if "isolation_max_abs_diff" in r]
    assert len(checked) == 2 and all(r["isolation_max_abs_diff"] == 0 and r["isolation_same_choice"] for r in checked)
    leaky = [r for r in _run(times, due, 16, leak=[], isolation=2) if "isolation_max_abs_diff" in r]
    assert any(r["isolation_max_abs_diff"] > 0 for r in leaky)  # state shared across sessions changes A on its repeat
    # session_steps == window_steps holds even with the leak: that field alone would not have shown it
    assert all(r["session_steps"] == r["window_steps"] for r in _run(times, due, 16, leak=[]))


def _questions(n_videos=30):
    tasks = BACKWARD + ("OCR", "FPD", "STU")
    return [{"id": f"q{v}-{k}", "stream": f"s{v}", "video": f"v{v}", "time": 20.0 + k, "task": tasks[k]}
            for v in range(n_videos) for k in range(6)]  # fmt: skip


def _row(q, correct, arm):
    control = arm.split("_", 1)[1]
    row = {"id": q["id"], "stream": q["stream"], "t": q["time"], "correct": int(correct), "t_step": q["time"] - 0.25,
           "step_window_end": q["time"], "kv_tokens": {"fifo": 1700, "stream_window": 700, "fresh16": 720, "fresh4": 180}[control]}  # fmt: skip
    if control.startswith("fresh"):
        w = int(control[5:])
        row.update({"window_steps": w, "session_steps": w, "first_step_t": q["time"] - 0.25 - (w - 1),
                    "evicted_tokens": 0, "kv_appended_tokens": row["kv_tokens"]})  # fmt: skip
    return row


def _write(tmp, questions, rule):
    for arm in tuple(f"{c}_fifo" for c in CHECKPOINTS) + NEW_ARMS:
        root = tmp / ("fifo" if arm.endswith("_fifo") else "new")
        write_jsonl(root / arm / "predictions.jsonl", [_row(q, rule(arm, q), arm) for q in questions])
    write_jsonl(tmp / "q.jsonl", questions)


def _video(q):
    return int(q["video"][1:])


def test_report_contrasts_and_interaction_on_the_same_bootstrap_samples(tmp_path):
    qs = _questions()

    def rule(arm, q):
        if q["task"] not in BACKWARD:
            return True
        if arm == "s2_fifo":
            return _video(q) % 2 == 0  # S2 loses half the backward questions only under fifo
        return True

    _write(tmp_path, qs, rule)
    r = recent_window_report(tmp_path / "new", tmp_path / "fifo", tmp_path / "q.jsonl")
    assert r["status"] == "complete" and not r["coverage_problems"]
    c = r["contrasts"]
    assert c["s2 - s1 under fifo"]["backward"]["delta_pp"] == -50.0
    assert c["s2 - s1 under fresh16"]["backward"]["delta_pp"] == 0.0
    inter = c["interaction: (s2 - s1)_fifo - (s2 - s1)_fresh16"]["backward"]
    assert inter["delta_pp"] == -50.0 and inter["ci95_pp"][1] < 0
    assert "interaction CI excludes 0" in r["reading"]["differential_effect"]
    assert c["s2: fifo - fresh16 (reliance on persistent state; primary control)"]["backward"]["delta_pp"] == -50.0
    assert c["s1: fifo - stream_window (diagnostic)"]["all"]["delta_pp"] == 0.0
    assert set(r["accuracy_pct"]) == {f"{ck}_{a}" for ck in CHECKPOINTS for a in ("fifo",) + CONTROLS}
    assert "not a Mage-VL" in r["fresh4_note"] and "no contrast isolates a single mechanism" in r["scope"]


def test_zero_under_fresh16_is_not_read_as_a_streaming_state_effect_without_the_interaction(tmp_path):
    qs = _questions(n_videos=4)

    def rule(arm, q):  # one question flips under fifo: wide interval, the interaction CI contains 0
        return not (arm == "s2_fifo" and q["id"] == "q0-0")

    _write(tmp_path, qs, rule)
    r = recent_window_report(tmp_path / "new", tmp_path / "fifo", tmp_path / "q.jsonl")
    assert r["contrasts"]["s2 - s1 under fresh16"]["backward"]["ci95_pp"] == [0.0, 0.0]
    assert "NOT evidence that the deficit is tied to streaming state" in r["reading"]["differential_effect"]


def test_interaction_equals_the_difference_of_the_two_gaps():
    qs = _questions(n_videos=12)
    meta = {q["id"]: q for q in qs}
    correct = {arm: {q["id"]: int((_video(q) * 7 + i * 3 + len(q["task"])) % (i + 2) == 0) for q in qs}
               for i, arm in enumerate(("s1_fifo", "s2_fifo", "s1_fresh16", "s2_fresh16"))}  # fmt: skip
    a = contrast(correct, meta, {"s2_fifo": 1, "s1_fifo": -1})["backward"]["delta_pp"]
    b = contrast(correct, meta, {"s2_fresh16": 1, "s1_fresh16": -1})["backward"]["delta_pp"]
    i = contrast(correct, meta, {"s2_fifo": 1, "s1_fifo": -1, "s2_fresh16": -1, "s1_fresh16": 1})["backward"]["delta_pp"]
    assert i == pytest.approx(a - b, abs=2e-3)


def test_protocol_violations_invalidate_the_report(tmp_path):
    qs = _questions(n_videos=3)
    _write(tmp_path, qs, lambda arm, q: True)
    rows = [_row(q, True, "s2_fresh16") for q in qs]
    rows[0]["evicted_tokens"], rows[1]["session_steps"] = 5, 99
    write_jsonl(tmp_path / "new" / "s2_fresh16" / "predictions.jsonl", rows)
    r = recent_window_report(tmp_path / "new", tmp_path / "fifo", tmp_path / "q.jsonl")
    assert r["status"] == "invalid_coverage" and any("evicted tokens in a fresh session" in p for p in r["coverage_problems"])
    preds = {arm: [_row(q, True, arm) for q in qs] for arm in ("s1_fifo", "s1_fresh16", "s2_fresh16")}
    preds["s2_fresh16"][2]["kv_tokens"] += 1
    preds["s1_fresh16"] = [{**row, "t_step": row["t_step"] - 1} for row in preds["s1_fresh16"]]
    problems = check_arms(preds, qs, ("s1_fresh16", "s2_fresh16"))
    assert any("differ between S1 and S2" in p for p in problems)
    assert any("answering step differs" in p for p in problems)
    assert any("no predictions" in p for p in check_arms({}, qs, ("s1_fresh4",)))


def test_flow_check_requires_a_clean_aba_isolation_result(tmp_path):
    streams = [{"id": "s0", "video": "v0", "duration": 30.0, "split": "dev",
                "questions": [{"id": "q0", "time": 20.0, "task": "EPM"}, {"id": "q1", "time": 25.0, "task": "OCR"}]}]  # fmt: skip
    write_jsonl(tmp_path / "streams.jsonl", streams)
    qs = [{**q, "stream": "s0", "video": "v0"} for q in streams[0]["questions"]]

    def write(diff):
        for arm in ("s1_fresh16", "s2_fresh16"):
            rows = [_row(q, True, arm) for q in qs]
            rows[0].update({"isolation_max_abs_diff": diff, "isolation_same_choice": True, "stream_seconds_elapsed": 4.0})
            write_jsonl(tmp_path / "flow" / arm / "predictions.jsonl", rows)

    write(0.0)
    ok = flow_check(tmp_path / "flow", tmp_path / "streams.jsonl")
    assert ok["passed"] and ok["isolation"]["s1_fresh16"]["checked"] == 1
    write(0.5)
    bad = flow_check(tmp_path / "flow", tmp_path / "streams.jsonl")
    assert not bad["passed"] and any("A-B-A isolation failed" in p for p in bad["problems"])


def test_exact_windows_from_the_manifest_grid_are_enforced(tmp_path):
    """Tester follow-up on ebb9724: any 1..W length with matching S1/S2 values used to pass."""
    from trace_vl.v3.recent_window import expected_windows

    times, questions, due = _stream()
    record = {"id": "s0", "video": "v0", "duration": 40.0, "split": "dev", "questions": questions}
    write_jsonl(tmp_path / "streams.jsonl", [record])
    grid = expected_windows(tmp_path / "streams.jsonl", 1.0, GAP, 600.0)
    assert grid["q0"] == {"k": 2, "t_step": 2.25, "windows": {16: {"window_steps": 3, "first_step_t": 0.25},
                                                               4: {"window_steps": 3, "first_step_t": 0.25}}}  # fmt: skip
    assert grid["q2"]["windows"][16] == {"window_steps": 16, "first_step_t": 14.25}
    assert grid["q2"]["windows"][4] == {"window_steps": 4, "first_step_t": 26.25}
    qs = [{**q, "stream": "s0", "video": "v0", "task": "EPM"} for q in questions]

    def rows(window, shorten=0):
        out = []
        for r in _run(times, due, window):
            r = {**r, "stream": "s0"}
            if shorten and r["id"] == "q2":  # the same wrong window in both checkpoints
                r.update({"window_steps": r["window_steps"] - shorten, "session_steps": r["session_steps"] - shorten,
                          "first_step_t": r["first_step_t"] + shorten})  # fmt: skip
            out.append(r)
        return out

    good = {"s1_fresh16": rows(16), "s2_fresh16": rows(16), "s1_fresh4": rows(4), "s2_fresh4": rows(4)}
    assert check_arms(good, qs, tuple(good), grid) == []
    short = {"s1_fresh16": rows(16, shorten=5), "s2_fresh16": rows(16, shorten=5)}
    assert check_arms(short, qs, tuple(short)) == []  # S1/S2 agree and 1 <= 11 <= 16: invisible without the grid
    problems = check_arms(short, qs, tuple(short), grid)
    assert any("expected 16 from 14.25" in p for p in problems)
    shifted = {"s1_fresh16": [{**r, "t_step": r["t_step"] - 1.0} for r in rows(16)]}
    assert any("answering step" in p for p in check_arms(shifted, qs, ("s1_fresh16",), grid))
