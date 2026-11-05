from copy import deepcopy

import pytest

from trace_vl.data.schema import read_jsonl
from trace_vl.evaluation.metrics import (
    auroc,
    average_precision,
    choice_letter,
    event_counts,
    score_predictions,
)


def test_ties_auc_and_ap():
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert average_precision([0, 1], [0.5, 0.5]) == 0.5
    assert auroc([0, 1], [0.1, 0.9]) == 1.0
    assert average_precision([0, 1], [0.1, 0.9]) == 1.0
    assert auroc([1, 1], [0.1, 0.9]) is None
    assert average_precision([0, 0], [0.1, 0.9]) is None


def test_choice_parser_does_not_match_words():
    assert choice_letter("apple") is None
    assert choice_letter("The answer is B.") == "B"
    assert choice_letter("E", 4) is None


def test_one_to_one_early_duplicate_and_censored_alerts():
    event = {"known_at": 2.0, "deadline": 4.0, "category": "x"}
    predictions = [{"time": t, "category": "x"} for t in [1.0, 2.5, 3.0]]
    metrics = event_counts(predictions, [event], 5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 2, 0)
    assert metrics["delays"] == [0.5]
    censored = event_counts([{"time": 2.5, "category": "x"}], [event], 3)
    assert (censored["tp"], censored["fp"], censored["fn"], censored["censored_events"]) == (0, 0, 0, 1)
    with pytest.raises(ValueError, match="outside"):
        event_counts([{"time": 6}], [], 5)


def test_scoring_uses_gt_labels_not_prediction_labels(fixture_manifest):
    row = read_jsonl(fixture_manifest)[0]
    prediction = {
        "id": row["id"],
        "answer": row["answer"],
        "observed_until": 3.5,
        "alerts": [{"time": 2.0, "category": "object_left"}],
        "frame_scores": [{"time": 0.0, "score": 0.1, "label": 1}, {"time": 2.0, "score": 0.9, "label": 0}],
    }
    score = score_predictions([row], [prediction])
    assert score["sampled_frame_auroc"] == 1.0
    assert score["qa_exact_match"] == 1.0
    assert score["events"]["tp"] == 1
    with pytest.raises(ValueError, match="mismatch"):
        score_predictions([row], [])
    row2, p2 = deepcopy(row), deepcopy(prediction)
    row2["id"], p2["id"] = "duplicate_episode", "duplicate_episode"
    with pytest.raises(ValueError, match="duplicate monitoring episode"):
        score_predictions([row, row2], [prediction, p2])


def test_nonfinite_frame_score_rejected(fixture_manifest):
    row = read_jsonl(fixture_manifest)[0]
    p = {
        "id": row["id"],
        "answer": "yes",
        "observed_until": 3.5,
        "frame_scores": [{"time": 0.0, "score": float("nan")}],
    }
    with pytest.raises(ValueError, match="invalid frame score"):
        score_predictions([row], [p])
