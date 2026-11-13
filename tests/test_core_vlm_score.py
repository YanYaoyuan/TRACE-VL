import copy

import pytest

from trace_vl.core_vlm.score_ovo import PROTOCOL_FIELDS, TASKS, compare_protocols, score


def population():
    qs = [{"id": t, "video": f"video-{t}", "task": t, "options": ["a", "b"], "answer": 0}
          for t in TASKS]
    ps = [{"id": t, "choice": 0 if t == "OCR" else 1} for t in TASKS]
    return qs, ps


def test_task_macro_is_not_micro_or_full_ovo():
    qs, ps = population()
    qs.append({**qs[0], "id": "extra-ocr"})
    ps.append({"id": "extra-ocr", "choice": 0})
    result = score(qs, ps, scope="development")
    assert result["question_weighted_percent"] == 20
    assert result["realtime_task_macro_percent"] == pytest.approx(100/6)
    assert result["two_category_macro_percent"] == pytest.approx(100/12)
    assert result["official_overall"] is None


def test_missing_task_cannot_silently_raise_macro():
    qs, ps = population()
    result = score(qs[:-1], ps[:-1], scope="development")
    assert result["missing_tasks"] == ["HLD"]
    assert result["two_category_macro_percent"] is None


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_exact_prediction_coverage(change):
    qs, ps = population()
    if change == "missing":
        ps.pop()
    elif change == "extra":
        ps.append({"id": "unknown", "choice": 0})
    else:
        ps.append(ps[0])
    with pytest.raises(ValueError):
        score(qs, ps, scope="development")


def test_recompute_correctness_and_keep_unparseable():
    qs, ps = population()
    ps[0]["correct"] = 0
    with pytest.raises(ValueError, match="correctness"):
        score(qs, ps, scope="development")
    ps[0] = {"id": "OCR", "choice": None}
    result = score(qs, ps, scope="development")
    assert result["unparseable_answers"] == 1
    assert result["questions"] == 9
    assert result["question_weighted_percent"] == 0


@pytest.mark.parametrize("times", [[1, 4], [2, 1], [float("nan")], [-1], []])
def test_invalid_causal_observations(times):
    qs, ps = population()
    qs[0]["query_time"] = 3
    ps[0]["frame_times"] = times
    with pytest.raises(ValueError):
        score(qs, ps, scope="development")


def test_real_frame_count_is_part_of_protocol():
    a = {key: "declared-value" for key in PROTOCOL_FIELDS}
    b = copy.deepcopy(a)
    assert compare_protocols(a, b)["same_declared_protocol"]
    b["frame_manifest_sha256"] = "eight-frames-not-four"
    assert compare_protocols(a, b)["mismatches"] == ["frame_manifest_sha256"]
    assert not compare_protocols({}, {})["same_declared_protocol"]


@pytest.mark.parametrize("choice", [True, -1, 2, "A", float("nan")])
def test_bad_choices_are_not_silently_coerced(choice):
    qs, ps = population()
    ps[0]["choice"] = choice
    with pytest.raises(ValueError):
        score(qs, ps, scope="development")


def test_forward_task_requires_separate_evaluator():
    qs, ps = population()
    qs[0]["task"] = "REC"
    with pytest.raises(ValueError, match="FAR"):
        score(qs, ps, scope="development")
