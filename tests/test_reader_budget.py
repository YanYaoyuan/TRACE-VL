import copy
import math

import pytest
import torch

from trace_vl.reader_budget import analysis, policies, run

TIMES = [[2.0 * i, 2.0 * i + .5] for i in range(36)]


def test_nested_reads_keep_recent_steps_and_have_exact_budgets():
    reads = policies.build(TIMES, 80., "source")
    assert reads == policies.build(TIMES, 80., "source")
    assert len(reads) == 13 and reads["all36"] == list(range(36))
    for family in ("spread", *(f"random:{seed}" for seed in policies.SEEDS)):
        name, _, seed = family.partition(":")
        suffix = f":{seed}" if seed else ""
        low, high = reads[f"{name}8{suffix}"], reads[f"{name}16{suffix}"]
        assert len(low) == 8 and len(high) == 16
        assert set(low) < set(high) < set(reads["all36"])
        for read in (low, high):
            assert read == sorted(set(read)) and {32, 33, 34, 35} <= set(read)
    assert policies.spread_order(TIMES)[:4] == [0, 31, 15, 23]
    other = policies.build(TIMES, 80., "other-source")
    assert reads["spread16"] == other["spread16"]
    assert any(reads[f"random8:{s}"] != other[f"random8:{s}"] for s in policies.SEEDS)


@pytest.mark.parametrize("kind", ["future", "nan", "unordered", "negative", "short", "bad_pair"])
def test_invalid_bank_is_rejected_before_selection(kind):
    times = copy.deepcopy(TIMES)
    if kind == "future":
        times[-1][-1] = 81.
    elif kind == "nan":
        times[1][0] = math.nan
    elif kind == "unordered":
        times[1], times[2] = times[2], times[1]
    elif kind == "negative":
        times[0][0] = -1.
    elif kind == "short":
        times.pop()
    else:
        times[0].append(.6)
    with pytest.raises(ValueError):
        policies.build(times, 80., "source")


def fixture_records(*, content=True):
    questions, records = [], []
    for i in range(4):
        reads = policies.build(TIMES, 80., str(i))
        q = {"id": str(i), "source_group": str(i), "gold": 0,
             "task": "before" if i % 2 else "after", "reads": reads}
        logits = {"text_only": [0., 0.]}
        for name in reads:
            value = 2. if policies.expected_tokens(name) > 128 else 0.
            logits[name] = [value, 0.]
            logits["blank:" + name] = [0. if content else value, 0.]
        records.append({**{k: q[k] for k in ("id", "source_group", "gold")}, "logits": logits,
                        "visual_tokens": {n: policies.expected_tokens(n) for n in logits}})
        questions.append(q)
    return questions, records


def test_content_endpoint_accepts_content_but_not_prompt_only_improvement():
    questions, records = fixture_records(content=True)
    value = analysis.analyze(records, questions)
    assert value["both_primary_intervals_positive"]
    assert value["tasks"]["before"]["sources"] == value["tasks"]["after"]["sources"] == 2
    questions, records = fixture_records(content=False)
    placebo = analysis.analyze(records, questions)
    assert not placebo["both_primary_intervals_positive"]
    assert all(placebo["contrasts"][n]["nll"]["mean"] == 0 for n in analysis.PRIMARY)


def test_seed_metrics_are_averaged_after_scoring_not_logits():
    record = {"gold": 0, "logits": {"random8:17": [2., 0.], "random8:29": [-6., 0.]}}
    result = analysis.arm_values(record)["random8"]
    assert result["acc"] == .5
    expected = (math.log1p(math.exp(-2.)) + math.log1p(math.exp(6.))) / 2
    assert result["nll"] == pytest.approx(expected)
    assert result["nll"] != pytest.approx(math.log1p(math.exp(2.)))


def test_analysis_rejects_incomplete_duplicated_or_wrong_budget_records():
    questions, records = fixture_records()
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze(records[:-1], questions)
    with pytest.raises(ValueError, match="duplicate"):
        analysis.analyze([records[0]] * 4, questions)
    records[0]["visual_tokens"]["all36"] = 128
    with pytest.raises(ValueError, match="budget"):
        analysis.analyze(records, questions)


def test_native_packing_handles_sixteen_and_thirtysix_with_matched_blanks(monkeypatch):
    from test_core_vlm_native import tiny_example

    from trace_vl.core_vlm.native import NativeVideoQALoss

    bb, _, _ = tiny_example()

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][-1]["text"]

        def __call__(self, *, videos, text, **kwargs):
            n = len(videos[0]) // 2
            ids = torch.tensor([[5] + ([1] + [3] * 16 + [2, 8]) * n + [6, 7]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                    "mm_token_type_ids": torch.where(ids == 3, 2, 0),
                    "video_grid_thw": torch.tensor([[n, 8, 8]])}

    bb.processor = Processor()
    monkeypatch.setattr(bb, "ids", lambda text: [5, 6] if len(text) > 1 else [9 + ord(text) - 65])
    entry = {"features": torch.randn(36, 16, bb.hidden), "times": TIMES, "hw": [128, 128],
             "grid": [4, 4], "fps": 2., "frame_count": 162,
             "indices": [[4 * i, 4 * i + 1] for i in range(36)]}
    row = {"id": "test", "question": "Which came first?", "options": ["a", "b"],
           "answer": 0, "query_time": 80.}
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    reads = {"spread8": list(range(28, 36)), "spread16": list(range(20, 36)), "all36": list(range(36))}
    logits, tokens, times, text_seconds = run.score_question(
        bb, net, entry, row, reads, torch.zeros(16, bb.hidden)
    )
    assert set(logits) == {"text_only", *reads, *("blank:" + n for n in reads)}
    for name, indices in reads.items():
        assert tokens[name] == tokens["blank:" + name] == 16 * len(indices)
        assert times[name]["input_tokens"] == times["blank:" + name]["input_tokens"]
        assert times[name]["lm_seconds"] >= 0 and times[name]["packing_seconds"] >= 0
    assert text_seconds >= 0
