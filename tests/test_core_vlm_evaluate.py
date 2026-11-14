import pytest

from trace_vl.core_vlm.evaluate import summarize


def rows():
    return [{"id": "a", "video": "v1", "question": "Q", "options": ["one", "two"], "answer": 0},
            {"id": "b", "video": "v2", "question": "Q", "answer_text": "two"}]


def test_report_recomputes_scores_instead_of_trusting_saved_flags():
    results = [{"id": "a", "run": "r", "answer": "B", "correct": True},
               {"id": "b", "run": "r", "answer": "Two.", "correct": False}]
    report = summarize(rows(), results, "r")
    assert report["scores"]["strict_mc_accuracy"]["score"] == 0
    assert report["scores"]["normalized_text_exact_match"]["score"] == 1
    assert report["videos"] == 2


@pytest.mark.parametrize("results", [[], [{"id": "a"}], [{"id": "a"}, {"id": "a"}],
                                    [{"id": "a"}, {"id": "extra"}]])
def test_partial_duplicate_or_extra_predictions_never_form_complete_report(results):
    with pytest.raises(ValueError):
        summarize(rows(), results, "r")


def test_report_refuses_predictions_from_other_model_run():
    results = [{"id": "a", "run": "wrong", "answer": "A"}, {"id": "b", "run": "r", "answer": "two"}]
    with pytest.raises(ValueError, match="identity"):
        summarize(rows(), results, "r")
