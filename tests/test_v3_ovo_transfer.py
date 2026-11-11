"""OVO-Bench Stage-2 transfer (experiments.md §14.3): adapter, coverage, no-future-frame, branch isolation, report."""

from __future__ import annotations

import json

import pytest
import torch

from trace_vl.v3.data import read_jsonl, write_jsonl
from trace_vl.v3.ovo_transfer import (
    ARMS,
    BACKWARD,
    build_transfer,
    check_coverage,
    stream_id,
    transfer_report,
)
from trace_vl.v3.temporal import video_half
from trace_vl.v3.train_stage2 import chunk_plan, stream_times

STEP, GAP, MAX = 1.0, 0.5, 600.0


def _videos(half: str, n: int) -> list[str]:
    out, k = [], 0
    while len(out) < n:
        v = f"src_videos/v{k}.mp4"
        if video_half(v) == half:
            out.append(v)
        k += 1
    return out


def _mc(qid, video, task, q, answer=0, options=("a", "b", "c", "d")):
    return {"id": qid, "source": "ovo-bench", "task": f"x/{task}", "video": video, "query_time": q,
            "question": "?", "options": list(options), "answer": answer, "region": None, "split": "test"}  # fmt: skip


@pytest.fixture
def ovo(tmp_path):
    diag, final = _videos("diagnostic", 3), _videos("final", 1)
    write_jsonl(tmp_path / "diagnostic.jsonl", [{"id": f"d{i}", "video": v, "task": "x/ASI"} for i, v in enumerate(diag)])
    write_jsonl(tmp_path / "final.jsonl", [{"id": "f0", "video": final[0], "task": "x/ASI"}])
    rows = [
        _mc("ovo-1", diag[0], "EPM", 120.0), _mc("ovo-2", diag[0], "ASI", 30.0), _mc("ovo-3", diag[1], "HLD", 400.0),
        _mc("ovo-4", diag[1], "OCR", 50.0), _mc("ovo-5", diag[2], "FPD", 10.0),
        _mc("ovo-6", diag[0], "EPM", 700.0),  # above the 600 s ceiling
        _mc("ovo-7", diag[1], "ATR", 900.0),  # above the ceiling (and after the video end)
        _mc("ovo-8", diag[2], "STU", 0.2),  # no complete step before the query
        _mc("ovo-9", diag[2], "OJR", 95.0),  # after the 90 s video end
        _mc("ovo-10", diag[0], "ACR", 20.0, answer=7),  # answer out of range
        _mc("ovo-11", final[0], "EPM", 30.0),  # frozen final video: out of scope
        {"id": "ovo-12", "task": "forward/REC", "video": diag[0], "query_time": 5.0, "kind": "count"},  # not MC
    ]  # fmt: skip
    write_jsonl(tmp_path / "mc.jsonl", rows)
    durations = {diag[0]: 300.0, diag[1]: 500.0, diag[2]: 90.0}
    inv = build_transfer(str(tmp_path / "mc.jsonl"), str(tmp_path / "diagnostic.jsonl"), str(tmp_path / "final.jsonl"),
                         str(tmp_path), str(tmp_path / "out"), STEP, GAP, MAX, durations=durations)  # fmt: skip
    return tmp_path, inv, diag


def test_inventory_counts_population_and_every_exclusion(ovo):
    tmp, inv, diag = ovo
    assert inv["diagnostic_videos"] == 3 and inv["overlap_with_frozen_final"] == 0
    assert inv["in_scope_questions"] == 10  # the final-half and forward rows are outside the population
    assert inv["excluded"] == {"invalid_options_or_answer": 1, "no_complete_step_before_query": 1,
                               "query_after_video_end": 1, "query_time_above_600s": 2}  # fmt: skip
    assert inv["eligible_questions"] == 5 and inv["streams"] == 3
    assert inv["backward"]["questions"] == 3 and inv["backward"]["videos"] == 2
    assert inv["backward"]["per_task"]["EPM"]["questions"] == 1 and inv["realtime"]["questions"] == 2
    assert inv["adequacy"]["adequate"] is False  # 3 backward questions: reported before any GPU work
    streams = {s["video"]: s for s in read_jsonl(tmp / "out" / "streams.jsonl")}
    assert streams[diag[0]]["duration"] == 120.0 and streams[diag[1]]["duration"] == 400.0  # cut at the last query
    assert all(s["split"] == "dev" and s["id"] == stream_id(s["video"]) for s in streams.values())


def test_rejects_diagnostic_videos_from_the_frozen_final_half(tmp_path):
    final = _videos("final", 1)
    write_jsonl(tmp_path / "d.jsonl", [{"id": "d", "video": final[0]}])
    write_jsonl(tmp_path / "f.jsonl", [{"id": "f", "video": final[0]}])
    write_jsonl(tmp_path / "mc.jsonl", [])
    with pytest.raises(AssertionError, match="frozen final"):
        build_transfer(str(tmp_path / "mc.jsonl"), str(tmp_path / "d.jsonl"), str(tmp_path / "f.jsonl"), str(tmp_path),
                       str(tmp_path / "out"), STEP, GAP, MAX, durations={final[0]: 100.0})  # fmt: skip


def test_every_manifest_question_is_scheduled_without_future_frames(ovo):
    tmp, _, _ = ovo
    for s in read_jsonl(tmp / "out" / "streams.jsonl"):
        times = stream_times(s, STEP, GAP, MAX)
        plan = chunk_plan(times, s["questions"], 8, GAP)
        asked = {q["id"]: steps[-1] for steps, qs in plan for q in qs}
        assert set(asked) == {q["id"] for q in s["questions"]}  # complete coverage
        for q in s["questions"]:
            assert times[asked[q["id"]]] + GAP / 2 <= q["time"] + 1e-6  # last seen window ends before the query
            later = [t for t in times if t > times[asked[q["id"]]]]
            assert not later or later[0] + GAP / 2 > q["time"] - 1e-6  # and it is the latest such step


def _fake_preds(tmp, questions, rule):
    for arm in ARMS:
        rows = [{"id": q["id"], "stream": q["stream"], "t": q["time"], "correct": int(rule(arm, q)),
                 "t_step": q["time"] - 0.25, "step_window_end": q["time"], "stream_steps": 10, "kv_tokens": 100}
                for q in questions]  # fmt: skip
        write_jsonl(tmp / "eval" / arm / "predictions.jsonl", rows)


def _questions(n_videos=40, per_video=6):
    qs = []
    for v in range(n_videos):
        for k in range(per_video):
            task = (BACKWARD + ("OCR", "FPD", "STU"))[k % 6]
            qs.append({"id": f"q{v}-{k}", "stream": f"s{v}", "video": f"v{v}", "time": 10.0 + k, "task": task})
    return qs


def test_report_decides_continue_and_reports_task_macro(tmp_path):
    qs = _questions()
    write_jsonl(tmp_path / "q.jsonl", qs)
    # S2 FIFO right on every backward question, S1 FIFO right on half of them; realtime identical
    _fake_preds(tmp_path, qs, lambda arm, q: q["task"] not in BACKWARD or arm.startswith("s2") or int(q["id"][1:].split("-")[0]) % 2)
    r = transfer_report(tmp_path / "eval", tmp_path / "q.jsonl")
    p = r["paired"]["s2_fifo_minus_s1_fifo"]
    assert r["decision"]["status"] == "continue" and p["backward"]["delta_pp"] == 50.0
    assert p["realtime"]["delta_pp"] == 0.0 and r["decision"]["realtime_no_harm"] is True
    assert set(p["per_task"]) == set(BACKWARD + ("OCR", "FPD", "STU"))
    assert p["backward_task_macro"]["delta_pp"] == 50.0
    assert r["accuracy_pct"]["s2_fifo"]["backward_task_macro"] == 100.0


def test_report_separates_inconclusive_from_threshold_failure(tmp_path):
    qs = _questions()
    write_jsonl(tmp_path / "q.jsonl", qs)
    _fake_preds(tmp_path, qs, lambda arm, q: True)  # identical arms: delta 0 with CI [0, 0] excludes +2
    assert transfer_report(tmp_path / "eval", tmp_path / "q.jsonl")["decision"]["status"] == "threshold_not_met"
    few = _questions(n_videos=4)  # one-video swings: the CI still includes +2 pp
    write_jsonl(tmp_path / "q2.jsonl", few)
    _fake_preds(tmp_path / "b", few, lambda arm, q: not (arm == "s1_fifo" and q["video"] == "v0" and q["task"] == "EPM"))
    d = transfer_report(tmp_path / "b" / "eval", tmp_path / "q2.jsonl")["decision"]
    assert d["status"] == "threshold_not_met_inconclusive" and d["continue_stage2_scaling"] is False


def test_coverage_problems_make_the_result_invalid(tmp_path):
    qs = _questions(n_videos=3)
    write_jsonl(tmp_path / "q.jsonl", qs)
    _fake_preds(tmp_path, qs, lambda arm, q: True)
    rows = read_jsonl(tmp_path / "eval" / "s1_full" / "predictions.jsonl")
    write_jsonl(tmp_path / "eval" / "s1_full" / "predictions.jsonl", rows[1:])
    r = transfer_report(tmp_path / "eval", tmp_path / "q.jsonl")
    assert r["decision"]["status"] == "invalid_coverage" and "s1_full: 1 missing" in r["coverage_problems"][0]
    late = [{**row, "step_window_end": row["t"] + 1.0} for row in rows]
    preds = {a: late if a == "s2_fifo" else rows for a in ARMS}
    assert any("after the query time" in p for p in check_coverage(preds, qs))
    shifted = [{**row, "t_step": row["t_step"] - 1.0} for row in rows]
    assert any("schedule differs" in p for p in check_coverage({a: shifted if a == "s2_full" else rows for a in ARMS}, qs))


class _FakeBackbone:
    """Minimal backbone: the cache is a list of token ids; lm appends to whichever cache it is given."""

    vision_end, tail_text, device, dtype = 9, "", "cpu", torch.float32

    def ids(self, text):
        return [1, 2, 3]

    def embed_ids(self, ids):
        return torch.zeros(1, len(ids), 4)

    def lm(self, embeds, positions, cache):
        cache["tokens"] += [0] * embeds.shape[1]
        cache["state"] += 1.0
        return torch.zeros(1, embeds.shape[1], 4)

    def fork(self, cache):
        return {"tokens": list(cache["tokens"]), "state": cache["state"].clone()}

    def letter_logprobs(self, hidden, ids):
        return torch.tensor([0.0, -1.0, -2.0, -3.0][: len(ids)])

    def recurrent_state(self, cache):
        return cache["state"]

    def kv_bytes(self, cache):
        return 2 * len(cache["tokens"])


def test_questions_run_on_a_fork_and_leave_the_live_stream_unchanged():
    from trace_vl.v3.positions import PositionCounter
    from trace_vl.v3.session import StreamSession
    from trace_vl.v3.train_stage2 import live_state

    session = StreamSession.__new__(StreamSession)
    session.bb, session.cache = _FakeBackbone(), {"tokens": [0] * 50, "state": torch.zeros(3)}
    session.pos, session.memory, session.step_index = PositionCounter(), [None] * 50, 7
    session.letter_ids, session.trace_ids = [11, 12, 13, 14], None
    before = live_state(session)
    choice, _ = session.ask("q?", ["a", "b", "c", "d"])
    assert choice == 0 and live_state(session) == before and len(session.cache["tokens"]) == 50
    session.bb.lm(torch.zeros(1, 2, 4), None, session.cache)  # a live-state change is detected
    assert live_state(session) != before


def test_inventory_is_json_serialisable(ovo):
    _, inv, _ = ovo
    assert json.loads(json.dumps(inv))["protocol"] == "ovo-transfer-dev-v2"


def test_v2_uses_the_whole_diagnostic_hash_half_and_v1_only_the_sampled_videos(tmp_path):
    """A100 inventory of e90d586: the 200 sampled §13 videos carried no EPM/HLD (filtered before sampling)."""
    from trace_vl.v3.ovo_transfer import half_counts

    diag, final = _videos("diagnostic", 3), _videos("final", 2)
    write_jsonl(tmp_path / "diagnostic.jsonl", [{"id": "d0", "video": diag[0]}])  # the §13 sample: one video
    write_jsonl(tmp_path / "final.jsonl", [{"id": "f0", "video": final[0]}])  # final[1] hashes final but is not listed
    rows = [_mc("ovo-1", diag[0], "ASI", 30.0), _mc("ovo-2", diag[1], "EPM", 40.0), _mc("ovo-3", diag[2], "HLD", 50.0),
            _mc("ovo-4", final[0], "EPM", 30.0), _mc("ovo-5", final[1], "HLD", 30.0)]  # fmt: skip
    write_jsonl(tmp_path / "mc.jsonl", rows)
    durations = {v: 100.0 for v in diag + final}
    args = (str(tmp_path / "mc.jsonl"), str(tmp_path / "diagnostic.jsonl"), str(tmp_path / "final.jsonl"), str(tmp_path))
    v2 = build_transfer(*args, str(tmp_path / "v2"), STEP, GAP, MAX, durations=durations)
    v1 = build_transfer(*args, str(tmp_path / "v1"), STEP, GAP, MAX, durations=durations, population="sampled-200")
    assert v2["protocol"] == "ovo-transfer-dev-v2" and v1["protocol"] == "ovo-transfer-dev-v1"
    assert v2["diagnostic_videos"] == 3 and v2["sampled_200_videos_in_population"] == 1
    assert {t: c["questions"] for t, c in v2["backward"]["per_task"].items()} == {"EPM": 1, "ASI": 1, "HLD": 1}
    assert {t: c["questions"] for t, c in v1["backward"]["per_task"].items()} == {"EPM": 0, "ASI": 1, "HLD": 0}
    videos = {q["video"] for q in read_jsonl(tmp_path / "v2" / "questions.jsonl")}
    assert videos == set(diag) and not videos & set(final)  # final[1] excluded although absent from final_frozen
    assert all(q["protocol"] == "ovo-transfer-dev-v2" for q in read_jsonl(tmp_path / "v2" / "questions.jsonl"))
    halves = half_counts(str(tmp_path / "mc.jsonl"))
    assert halves["all"]["EPM"] == {"questions": 2, "videos": 2} and halves["final"]["HLD"] == {"questions": 1, "videos": 1}
    assert halves["diagnostic"]["ASI"] == {"questions": 1, "videos": 1}
    with pytest.raises(ValueError, match="population"):
        build_transfer(*args, str(tmp_path / "x"), STEP, GAP, MAX, durations=durations, population="final")
