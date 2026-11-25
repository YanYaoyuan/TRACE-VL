import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trace_vl.longstream_training.prepare import prepare, validate
from trace_vl.longstream_training.selection import QuestionSelector, retained_indices


def test_question_blind_reservoir_reproducible_bounded_and_retains_recent():
    times = [float(i) for i in range(400)]
    a = retained_indices(times, seed=17)
    assert a == retained_indices(times, seed=17) and a != retained_indices(times, seed=29)
    assert len(a) == len(set(a)) == 36 and a[-4:] == [396, 397, 398, 399]
    assert len([i for i in a if i < 396]) == 32
    assert retained_indices(times[:7], seed=17) == list(range(7))
    with pytest.raises(ValueError, match="order"):
        retained_indices([1.0, 1.0], seed=17)


def test_hard_read_budget_and_answer_gradient_without_hidden_temporal_address():
    torch.manual_seed(17)
    model = QuestionSelector(32, 12)
    features = torch.randn(36, 16, 32)
    question = torch.randn(32)
    selected, indices, _ = model(features, list(range(36)), question, 40.0)
    assert selected.shape == (8, 16, 32) and torch.equal(selected, features[indices])
    assert indices.tolist() == sorted(indices.tolist())
    (selected * torch.randn_like(selected)).sum().backward()
    assert model.query.weight.grad.norm() > 0 and model.key.weight.grad.norm() > 0
    model.eval()
    replay, ii, _ = model(features, list(range(36)), question, 40.0)
    assert torch.equal(replay, selected) and torch.equal(ii, indices)
    with pytest.raises(ValueError, match="future"):
        model(features, list(range(36)), question, 10.0)
    with pytest.raises(ValueError, match="fixed-budget"):
        model(features, list(range(36)), question, 40.0, count=32)


def test_real_staging_resume_preserves_questions_splits_and_bytes(tmp_path):
    ref = tmp_path / "reference"
    ref.mkdir()
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        json.dumps({"video": "/training/other.mp4", "source_aliases": ["other"], "media_sha256": "unrelated"})
        + "\n"
    )
    data = []
    for i, split in enumerate(("train", "dev")):
        video = tmp_path / f"raw{i}.mp4"
        video.write_bytes(f"actual unique source {i}".encode())
        row = {
            "id": f"q{i}",
            "source_group": f"source{i}",
            "split": split,
            "task": "before",
            "query_time": 80.0,
            "media_duration": 90.0,
            "evidence": [3.0, 20.0],
            "protocol": "causal-prefix",
            "question": "What happened before the described event?",
            "options": ["a", "b"],
            "answer": 0,
            "video": str(video),
        }
        (ref / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        data.append(row)
    args = SimpleNamespace(out=tmp_path / "prepared", reference=ref, prior=prior, workers=2)
    prepare(args)
    first = (args.out / "data-audit.json").read_bytes()
    prepare(args)
    assert (args.out / "data-audit.json").read_bytes() == first
    for row in data:
        new = json.loads((args.out / f"{row['split']}.jsonl").read_text())
        assert new["question"] == row["question"] and new["answer"] == row["answer"]
        assert Path(new["video"]).read_bytes() == Path(row["video"]).read_bytes()
    with pytest.raises(ValueError, match="temporal address"):
        validate({**data[0], "target_time": 20.0})
    (ref / "train.jsonl").write_text(json.dumps({**data[0], "answer": 1}) + "\n")
    with pytest.raises(ValueError):
        prepare(args)
