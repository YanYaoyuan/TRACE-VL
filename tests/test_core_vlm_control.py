import pytest

from trace_vl.core_vlm.control import score_answer, select_records


def test_mc_does_not_mine_answer_letters_from_explanations():
    row = {"options": ["a", "b", "c"], "answer": 1}
    assert score_answer("B.", row)["correct"]
    for text in ("Because A is wrong, B", "Z", "", "The answer is B", "AB"):
        assert not score_answer(text, row)["correct"]


def test_exact_match_is_labelled_and_not_semantic_judging():
    row = {"answer_text": "a child"}
    assert score_answer(" A CHILD. ", row)["correct"]
    assert not score_answer("a kid", row)["correct"]
    assert score_answer("a child", row)["metric"] == "normalized_text_exact_match"


def test_selection_is_input_order_independent_and_video_balanced():
    rows = [{"id": str(i), "video": v} for i, v in enumerate(["a", "a", "b", "c"])]
    assert [r["video"] for r in select_records(rows, 3)] == ["a", "b", "c"]
    assert select_records(rows, 0) == select_records(rows[::-1], 0)
    with pytest.raises(ValueError):
        select_records(rows, -1)
