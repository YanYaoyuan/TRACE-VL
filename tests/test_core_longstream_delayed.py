import copy

import pytest
from test_core_longstream import step

from trace_vl.core_vlm.longstream import BoundedBank
from trace_vl.core_vlm.longstream_delayed import (
    CausalEvents,
    delayed_queries,
    events_for,
    memory_audit,
    read_steps,
    validate_predictions,
)


def original(identity="a", target=10.0):
    return {"id": identity, "query_time": target, "question": "What just happened?",
            "options": ["first", "second"], "answer": 1}


def test_only_media_availability_selects_delays_and_prompt_never_exposes_delay():
    row = original()
    before = copy.deepcopy(row)
    queries = delayed_queries([row], 190.0)
    assert [q["delay_seconds"] for q in queries] == [0, 15, 60, 180]
    assert len({q["question"] for q in queries}) == 1
    assert "10.000 seconds" in queries[0]["question"]
    assert row == before
    assert all(q["options"] == row["options"] and q["answer"] == row["answer"] for q in queries)
    assert [q["delay_seconds"] for q in delayed_queries([row], 309.99)] == [0, 15, 60, 180]
    assert len(delayed_queries([row], 310.0)) == 5
    with pytest.raises(ValueError, match="outside media"):
        delayed_queries([row], 9.9)


def test_events_respect_frame_end_capture_before_question_and_shared_targets():
    queries = delayed_queries([original("at_end", .5), original("between", .49)], 20)
    seen, count = [], 0

    def dispatch(event):
        seen.append((event, count))

    events = CausalEvents(events_for(queries), dispatch)
    for end in (.5, 1.5, 15.5, 16.5):
        events.due(end, inclusive=False)
        count += 1
        events.due(end, inclusive=True)
    events.due(float("inf"), inclusive=True)
    first = seen[:4]
    assert [(event[1], encoded) for event, encoded in first] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert first[0][0][2] == "between" and first[2][0][2] == "at_end"
    assert len([e for e, _ in seen if e[1] == 0]) == 2  # not one capture per delay
    assert events.cursor == len(events.events)
    corrupt = copy.deepcopy(queries)
    corrupt[-1]["answer"] = 0
    with pytest.raises(ValueError, match="target changed"):
        events_for(corrupt)


def test_retention_and_target_reads_do_not_receive_extra_recent8_slots():
    bank = BoundedBank(32, 17, keep_prefix=True)
    protected = None
    for i in range(100):
        bank.append(step(i))
        if i == 60:
            protected = list(bank.recent)
        if i == 75:
            assert [s.indices for s in read_steps(bank, "fifo_time8", 60.5)] == [s.indices for s in protected]
    assert read_steps(bank, "fifo_time8", 60.5) == []
    assert [s.times[0] for s in read_steps(bank, "prefix_time8", 60.5)] == [0, 1, 2, 3]
    bank.reservoir = [step(i) for i in range(32)]
    before = memory_audit(bank, {"q": protected})
    selected = read_steps(bank, "reservoir_time8", 95.5)
    assert max(s.end for s in selected) == 31.5  # recent8 older4 are excluded from this policy
    assert memory_audit(bank, {"q": protected}) == before
    assert before["per_policy_unique_retained_steps"]["reservoir"] == 36
    assert before["per_policy_unique_retained_steps"]["protected_snapshot8"] <= 24
    assert [s.times[0] for s in protected] == list(range(53, 61))
    with pytest.raises(ValueError, match="capacity exceeded"):
        memory_audit(bank, {"a": protected, "b": protected, "c": protected})


def test_prediction_gate_distinguishes_target_causality_from_late_current_control():
    from trace_vl.core_vlm.longstream_delayed import ARMS

    q = {"query_time": 20.0, "target_time": 10.0, "options": ["a", "b"]}
    p = {"query_time_encoder_calls": 0, "visual_tokens": 16, "read_times": [[9.0, 9.5]],
         "option_logits": [0.0, 1.0], "choice": 1, "option_logits_fp32": [0.1, 1.1], "choice_fp32": 1}
    result = {"predictions": {a: copy.deepcopy(p) for a in ARMS}}
    result["predictions"]["recent8_now"]["read_times"] = [[19., 19.5]]
    validate_predictions(result, q)
    result["predictions"]["fifo_time8"]["read_times"] = [[10., 10.5]]
    with pytest.raises(ValueError, match="causality"):
        validate_predictions(result, q)
    result["predictions"]["fifo_time8"]["read_times"] = [[9., 9.5]]
    result["predictions"]["fifo_time8"]["option_logits_fp32"] = [float("nan"), 1.]
    with pytest.raises(ValueError, match="logit scoring"):
        validate_predictions(result, q)


def test_real_native_lm_snapshot_logits_ignore_later_arrival_and_never_reencode(monkeypatch):
    from dataclasses import replace

    import torch
    from test_core_vlm_native import tiny_example

    from trace_vl.core_vlm.longstream import answer
    from trace_vl.core_vlm.longstream_delayed import anchored_question
    from trace_vl.core_vlm.native import NativeVideoQALoss

    bb, _, _ = tiny_example()

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][-1]["text"]

        def __call__(self, *, text, videos, **kwargs):
            assert not videos[0].any()
            count = len(videos[0]) // 2
            ids = torch.tensor([[5] + ([1] + [3] * 16 + [2, 8]) * count + [6, 7]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                    "mm_token_type_ids": torch.where(ids == 3, 2, 0),
                    "video_grid_thw": torch.tensor([[count, 8, 8]])}

    bb.processor = Processor()

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot query re-encoded pixels")

    monkeypatch.setattr(bb, "encode", forbidden)
    monkeypatch.setattr(bb.visual, "forward", forbidden)
    selected = [replace(step(i, (4, 4)), tokens=torch.randn(16, 32)) for i in range(8)]
    q = {"question": anchored_question("What just happened?", 10.),
         "options": ["first", "second"], "answer": 1, "query_time": 10.}
    metadata = {"streams": [{"avg_frame_rate": "2/1"}], "frames": [{}] * 622}
    qa = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    now = answer(bb, qa, selected, q, metadata, (128, 128), "video")
    late = answer(bb, qa, selected, {**q, "query_time": 310., "answer": 0}, metadata, (128, 128), "video")
    assert now == late and now["visual_tokens"] == 128 and now["query_time_encoder_calls"] == 0
