import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trace_vl.longstream_training.prepare import prepare, validate
from trace_vl.longstream_training.selection import QuestionSelector, retained_indices


@pytest.fixture(autouse=True)
def fixture_disk_reserve(monkeypatch):
    # Tiny staging fixtures run on /tmp, which may have less than the production
    # 30GiB reserve. Mock capacity only in tests; keep the actual /data guard.
    from trace_vl.longstream_training import prepare as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=1 << 40))


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


def test_native_answer_loss_reaches_both_lora_and_question_selector(monkeypatch):
    from test_core_vlm_native import tiny_example

    from trace_vl.core_vlm.native import NativeVideoQALoss
    from trace_vl.longstream_training.train import sample

    bb, _, _ = tiny_example()
    bb.add_lora(4, 8, dropout=0.0)

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][-1]["text"]

        def __call__(self, *, videos, text, **kwargs):
            n = len(videos[0]) // 2
            assert not videos[0].any()
            ids = torch.tensor([[5] + ([1] + [3] * 16 + [2, 8]) * n + [6, 7]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.where(ids == 3, 2, 0),
                "video_grid_thw": torch.tensor([[n, 8, 8]]),
            }

    bb.processor = Processor()
    for p in bb.visual.parameters():
        assert not p.requires_grad
    entry = {
        "features": torch.randn(36, 16, 32),
        "times": [[2.0 * i, 2.0 * i + 0.5] for i in range(36)],
        "hw": [128, 128],
        "grid": [4, 4],
        "fps": 2.0,
        "frame_count": 162,
        "indices": [[4 * i, 4 * i + 1] for i in range(36)],
    }
    row = {
        "id": "question",
        "question": "Which event happened before the person left?",
        "options": ["event a", "event b"],
        "answer": 0,
        "query_time": 80.0,
        "evidence": [1.0, 3.0],
    }
    selector = QuestionSelector(32, 12)
    steps, record, selected = sample(bb, selector, entry, row)
    steps2, changed, selected2 = sample(bb, selector, entry, {**row, "answer": 1, "evidence": [70.0, 71.0]})
    assert selected == selected2
    assert all(torch.equal(a.tokens, b.tokens) for a, b in zip(steps, steps2))
    qa = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    loss, info = qa(steps, record)
    assert info["visual_tokens_total"] == 128
    loss.backward()
    assert selector.query.weight.grad.norm() > 0 and selector.key.weight.grad.norm() > 0
    assert (
        sum(p.grad.norm() for n, p in bb.model.named_parameters() if "lora_" in n and p.grad is not None) > 0
    )
    assert all(p.grad is None for p in bb.visual.parameters())
    assert changed["answer"] == 1 and record["answer"] == 0


def test_staging_accepts_access_time_updates_but_verifies_content(tmp_path, monkeypatch):
    import os

    from trace_vl.longstream_training import prepare as module

    ref = tmp_path / "reference"
    ref.mkdir()
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        json.dumps({"video": "/other.mp4", "source_aliases": [], "media_sha256": "other"}) + "\n"
    )
    originals = set()
    for i, split in enumerate(("train", "dev")):
        video = tmp_path / f"{split}.mp4"
        video.write_bytes(bytes([i + 1]) * 10)
        originals.add(video)
        row = {
            "id": split,
            "source_group": split,
            "split": split,
            "question": "Which event happened first?",
            "options": ["a", "b"],
            "answer": 0,
            "protocol": "causal-prefix",
            "query_time": 80.0,
            "media_duration": 90.0,
            "evidence": [1.0, 10.0],
            "video": str(video),
        }
        (ref / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    actual = module.sha256

    def read_with_atime(path):
        result = actual(path)
        if Path(path) in originals:
            st = Path(path).stat()
            os.utime(path, ns=(st.st_atime_ns + 1_000_000_000, st.st_mtime_ns))
        return result

    monkeypatch.setattr(module, "sha256", read_with_atime)
    module.prepare(SimpleNamespace(out=tmp_path / "staged", reference=ref, prior=prior, workers=2))
    assert json.loads((tmp_path / "staged/data-audit.json").read_text())["completed"]
