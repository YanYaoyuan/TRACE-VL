import copy
import json
from types import SimpleNamespace

import pytest

from trace_vl.core_vlm.longstream_delayed_candidates import choose_smoke, validate_rows


def cohort():
    return [{"id": f"q{i}", "source_group": f"used{i % 28}", "query_time": 60.0 + i,
             "task": "Event Understanding" if i < 26 else "Prospective Reasoning",
             "temporal_clue_type": "Prior", "frames_required": "multiple"} for i in range(34)]


def test_candidate_population_and_semantic_strata_cannot_silently_drop_or_relabel_targets():
    rows = cohort()
    allowed = {r["source_group"] for r in rows}
    validate_rows(rows, allowed)
    with pytest.raises(ValueError, match="all34"):
        validate_rows(rows[:-1], allowed)
    with pytest.raises(ValueError, match="all34"):
        validate_rows(rows, allowed - {"used0"})
    wrong = copy.deepcopy(rows)
    wrong[-1]["task"] = "Event Understanding"
    with pytest.raises(ValueError, match="semantic strata"):
        validate_rows(wrong, allowed)
    for changes in ({"query_time": 59.9}, {"query_time": float("nan")}, {"frames_required": "single"},
                    {"temporal_clue_type": "Concurrent"}, {"evidence": [0, 1]}):
        wrong = copy.deepcopy(rows)
        wrong[0].update(changes)
        with pytest.raises(ValueError, match="candidate metadata"):
            validate_rows(wrong, allowed)


def test_smoke_deterministically_covers_longest_and_distinct_multiple_capture_source():
    sources = [
        {"source_group": "long", "planned_stream_encoder_calls": 1040, "queries": [{"target_id": "a"}] * 5},
        {"source_group": "multi", "planned_stream_encoder_calls": 692,
         "queries": [{"target_id": "b"}] * 5 + [{"target_id": "c"}] * 5},
        {"source_group": "short", "planned_stream_encoder_calls": 432, "queries": [{"target_id": "d"}] * 5},
    ]
    assert choose_smoke(sources) == choose_smoke(sources[::-1]) == ["long", "multi"]
    with pytest.raises(ValueError, match="multi-target"):
        choose_smoke([sources[0], sources[2]])


def test_prepare_seals_all170_queries_from_actual_manifests_without_inherited_id(tmp_path, monkeypatch):
    from trace_vl.core_vlm import longstream_delayed_candidates as module
    from trace_vl.core_vlm.data import sha256
    from trace_vl.core_vlm.prepare import digest, write_json

    # Hardware/model identities and /data video locations are external fixture
    # boundaries. Real JSON IO, hashes, PTS scheduling, cohort construction and
    # ensure_run_manifest are exercised end to end without a GPU or NAS mount.
    data, reference = tmp_path / "data", tmp_path / "reference"
    data.mkdir()
    reference.mkdir()
    plain = tmp_path / "train" / "checkpoint-002500"
    plain.mkdir(parents=True)
    write_json(plain.parent / "run.json", {"videos": {"other": "not-a-candidate-hash"}})
    base = {"model": "/data/fake-frozen-model", "model_identity": {"config": "fixed"}, "plain": str(plain),
            "checkpoint": {"run": "fixed-plain-run"}, "pixels": 65536, "capacity": 32, "seed": 17,
            "primary_score": "fp32_letter_head_over_bf16_hidden", "secondary_score": "bf16_letter_head"}
    base["id"] = digest(base)
    write_json(reference / "inputs.json", base)
    monkeypatch.setattr(module.core, "REFERENCE_ID", base["id"])
    monkeypatch.setattr(module, "checked_checkpoint", lambda *a: base["checkpoint"])
    monkeypatch.setattr(module, "checkpoint_identity", lambda *a: base["model_identity"])
    metadata = tmp_path / "pts.json"
    write_json(metadata, {"frames": [{"best_effort_timestamp_time": str(i / 2)} for i in range(1001)]})
    rows = cohort()
    groups = sorted({q["source_group"] for q in rows})
    video_files = {}
    sources = []
    for group in groups:
        video = "/data/test-candidate-" + group + ".mp4"
        backing = tmp_path / (group + ".bytes")
        backing.write_bytes(group.encode())
        video_files[video] = backing
        sources.append({"source_group": group, "video": video, "video_sha256": sha256(backing),
                        "metadata": str(metadata), "metadata_sha256": sha256(metadata), "last_valid_source_time": 500.0})
    for q in rows:
        q.update(question="What happened?", options=["a", "b"], answer=0,
                 video="/data/test-candidate-" + q["source_group"] + ".mp4")
    questions = data / "questions.jsonl"
    questions.write_text(''.join(json.dumps(q) + '\n' for q in rows))
    old = tmp_path / "old-development.jsonl"
    old.write_text(''.join(json.dumps({"source_group": g}) + '\n' for g in groups))
    spec = {"question_ids": sorted(q["id"] for q in rows), "old_development_manifest_sha256": sha256(old)}
    spec["id"] = digest(spec)
    write_json(data / "candidate-spec.json", spec)
    ready = {"completed": True, "candidate_spec_id": spec["id"], "questions_sha256": sha256(questions), "sources": sources}
    ready["id"] = digest(ready)
    write_json(data / "ready.json", ready)
    monkeypatch.setattr(module, "OLD_DEVELOPMENT_MANIFEST", old)
    monkeypatch.setattr(module, "READY_SHA256", sha256(data / "ready.json"))
    monkeypatch.setattr(module, "DATA_ID", ready["id"])
    monkeypatch.setattr(module, "sha256", lambda p: sha256(video_files.get(str(p), p)))
    out = tmp_path / "sealed"
    args = SimpleNamespace(out=out, reference=reference, data=data, overlap=tmp_path / "prior-overlap")
    module.prepare(args)
    result = json.loads((out / "inputs.json").read_text())
    assert result["id"] == digest({k: v for k, v in result.items() if k != "id"})
    assert result["id"] != base["id"]
    assert len(result["sources"]) == 28 and sum(len(s["queries"]) for s in result["sources"]) == 170
    assert all(len(s["queries"]) in (5, 10) for s in result["sources"])
    assert result["primary_targets"] == 26 and result["secondary_prediction_targets"] == 8
    assert result["cohorts"] == {str(d): {"questions": 34, "sources": 28} for d in (0, 15, 60, 180, 300)}
    # Resumption must verify the same canonical identity instead of inheriting a
    # reference ID or publishing a new accidental run identity.
    module.prepare(args)
    assert json.loads((out / "inputs.json").read_text()) == result
