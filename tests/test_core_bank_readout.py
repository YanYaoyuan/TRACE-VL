import copy
import hashlib
import io
import json
import tarfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_core_longstream import step
from test_core_vlm_native import tiny_example

from trace_vl.core_vlm import bank_readout as module
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.longstream import answer
from trace_vl.core_vlm.native import NativeVideoQALoss
from trace_vl.core_vlm.prepare import digest, write_json


def slot(i):
    return {
        "indices": [2 * i, 2 * i + 1],
        "times": [float(i), i + 0.5],
        "rgb_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
    }


def test_each_reader_uses_own_logged_bank_and_never_protected_union_or_future():
    bank = [slot(i) for i in range(12)] + [slot(100)]
    before = copy.deepcopy(bank)
    latest = module.select_slots(bank, 10.5, "latest8")
    uniform = module.select_slots(bank, 10.5, "uniform8")
    full = module.select_slots(bank, 10.5, "all_bank")
    assert [s["times"][0] for s in latest] == list(range(3, 11))
    assert len(uniform) == 8 and uniform[0] == slot(0) and uniform[-1] == slot(10)
    assert len(full) == 11 and bank == before
    record = {
        "query": {"target_time": 50.5},
        "snapshot": [slot(50)],
        "banks": {"original": bank, "29": [slot(20)], "43": [slot(30)]},
    }
    assert slot(50) in module.union_slots(record)
    assert slot(20) not in module.select_slots(bank, 50.5, "all_bank")
    assert slot(50) not in module.select_slots(bank, 50.5, "all_bank")
    assert all(s["times"][-1] <= 10.5 for s in full)
    for policy in module.POLICIES:
        assert module.select_slots([], 10, policy) == []
    with pytest.raises(ValueError, match="population"):
        module.select_slots([slot(0), slot(0)], 10, "all_bank")
    with pytest.raises(ValueError, match="32"):
        module.select_slots([slot(i) for i in range(33)], 100, "all_bank")
    with pytest.raises(ValueError, match="conflicting"):
        module.union_slots({**record, "snapshot": [{**slot(0), "rgb_sha256": "f" * 64}]})


def test_real_native_lm_explicit512_token_diagnostic_preserves_default_and_no_vision(monkeypatch):
    bb, _, _ = tiny_example()

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][-1]["text"]

        def __call__(self, *, text, videos, **kwargs):
            assert not videos[0].any()
            count = len(videos[0]) // 2
            ids = torch.tensor([[5] + ([1] + [3] * 16 + [2, 8]) * count + [6, 7]])
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.where(ids == 3, 2, 0),
                "video_grid_thw": torch.tensor([[count, 8, 8]]),
            }

    bb.processor = Processor()

    def forbidden(*args, **kwargs):
        raise AssertionError("answer re-encoded reconstructed pixels")

    monkeypatch.setattr(bb, "encode", forbidden)
    monkeypatch.setattr(bb.visual, "forward", forbidden)
    selected = [replace(step(i, (4, 4)), tokens=torch.randn(16, 32)) for i in range(32)]
    query = {"question": "What happened before?", "options": ["a", "b"], "answer": 1, "query_time": 40.0}
    metadata = {"streams": [{"avg_frame_rate": "2/1"}], "frames": [{}] * 82}
    qa = NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4))
    old = answer(bb, qa, selected[:8], query, metadata, (128, 128), "same")
    assert answer(bb, qa, selected[:8], query, metadata, (128, 128), "same", max_read_steps=32) == old
    with pytest.raises(ValueError, match="bounded"):
        answer(bb, qa, selected, query, metadata, (128, 128), "same")
    full = answer(bb, qa, selected, query, metadata, (128, 128), "same", max_read_steps=32)
    changed = answer(
        bb, qa, selected, {**query, "answer": 0}, metadata, (128, 128), "same", max_read_steps=32
    )
    assert full == changed and full["visual_tokens"] == 512 and full["query_time_encoder_calls"] == 0
    with pytest.raises(ValueError, match="diagnostic budgets"):
        answer(bb, qa, selected, query, metadata, (128, 128), "same", max_read_steps=100)


def test_prepare_all77_stored_banks_archive_and_stable_resume(tmp_path, monkeypatch):
    ref = tmp_path / "reference"
    (ref / "questions").mkdir(parents=True)
    plain = tmp_path / "checkpoint"
    plain.mkdir()
    video = tmp_path / "video.bytes"
    video.write_bytes(b"fixed-video")
    metadata = tmp_path / "pts.json"
    write_json(metadata, {"frames": []})
    sources = []
    archive_members = {}
    for i in range(77):
        q = {
            "id": f"q{i}/delay300",
            "target_id": f"q{i}",
            "delay_seconds": 300,
            "target_time": 60.0 + i,
            "query_time": 360.0 + i,
            "question": "What was held?",
            "options": ["a", "b"],
            "answer": 1,
        }
        bank = [slot(j) for j in range(3 if i == 0 else 32)]
        snapshot = [slot(j) for j in range(40, 48)]
        view = module.reconstruction_view(module.select_slots(bank, q["target_time"], "latest8"))
        predictions = {
            name: copy.deepcopy(view)
            for name in (
                "text_only",
                "protected_snapshot8",
                "reservoir_time8",
                "reservoir_seed29_time8",
                "reservoir_seed43_time8",
            )
        }
        record = {
            "predictions": predictions,
            "memory": {
                "retained": {"reservoir": bank, "recent8_control": [], f"snapshot:q{i}": snapshot},
                "writer_controls": {
                    "banks": {
                        name: {"retained": {"history": bank, "recent": []}}
                        for name in ("reservoir_seed29_time8", "reservoir_seed43_time8")
                    }
                },
            },
        }
        name = "questions/" + digest(q["id"]) + ".json"
        write_json(ref / name, record)
        archive_members[name] = (ref / name).read_bytes()
        sources.append(
            {
                "source_group": f"used{i}",
                "video": str(video),
                "video_sha256": sha256(video),
                "metadata": str(metadata),
                "metadata_sha256": sha256(metadata),
                "queries": [q],
            }
        )
    base = {
        "sources": sources,
        "model": "/fake/model",
        "model_identity": {"fixed": True},
        "plain": str(plain),
        "checkpoint": {"run": "fixed"},
        "pixels": 65536,
        "primary_score": "fp32_letter_head_over_bf16_hidden",
        "secondary_score": "bf16_letter_head",
    }
    base["id"] = digest(base)
    write_json(ref / "inputs.json", base)
    write_json(ref / "report.json", {"completed": True, "questions": 385, "source_videos": 77})
    with tarfile.open(ref / "reports.tar.gz", "w:gz") as archive:
        for name, raw in archive_members.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    monkeypatch.setattr(module, "REFERENCE_ID", base["id"])
    monkeypatch.setattr(module, "REPORT_SHA256", sha256(ref / "report.json"))
    monkeypatch.setattr(module, "ARCHIVE_SHA256", sha256(ref / "reports.tar.gz"))
    monkeypatch.setattr(module, "checked_checkpoint", lambda *args: base["checkpoint"])
    monkeypatch.setattr(module, "checkpoint_identity", lambda *args: base["model_identity"])
    args = SimpleNamespace(out=tmp_path / "new", reference=ref)
    module.prepare(args)
    sealed = json.loads((args.out / "inputs.json").read_text())
    assert sealed["id"] == digest({k: v for k, v in sealed.items() if k != "id"})
    assert len(sealed["records"]) == 77 and sealed["predictions"] == 847
    assert len(sealed["smoke_ids"]) == 2
    assert sealed["records"][0]["banks"]["original"] == [slot(i) for i in range(3)]
    module.prepare(args)
    assert json.loads((args.out / "inputs.json").read_text()) == sealed
    (ref / "questions" / f"{digest('q0/delay300')}.json").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        module.prepare(args)
