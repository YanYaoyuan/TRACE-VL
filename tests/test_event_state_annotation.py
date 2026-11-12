"""P1 step 1 annotation feasibility tools (docs/event_state/causal_label_protocol.md)."""

from __future__ import annotations

import json

import pytest

from trace_vl.event_state.annotation import (
    RECORD_VERSION,
    blank_record,
    feasibility_report,
    load_uca,
    official_test_videos,
    prefix_plan,
    rule_draft,
    sample_feasibility,
    stratum_guess,
    validate_record,
)


def _uca(tmp_path, n=60):
    data = {"Train": {}, "Val": {}, "Test": {}}
    for i in range(n):
        split = ("Train", "Train", "Val", "Test")[i % 4]
        name = f"{('Abuse', 'Arson', 'Normal_Videos_')[i % 3]}{i:03d}_x264"
        sentences = ["A man opens the gate", "The car is parked near the wall",
                     "A man walks in. Then he seemed to take the bag and runs away"]
        data[split][name] = {"duration": 120.0, "timestamps": [[20.0 + k * 20, 30.0 + k * 20] for k in range(3)],
                             "sentences": sentences}  # fmt: skip
    for split, entries in data.items():
        (tmp_path / f"UCFCrime_{split}.json").write_text(json.dumps(entries))
    (tmp_path / "Anomaly_Test.txt").write_text("Abuse/Abuse000_x264.mp4\nArson/Arson001_x264.mp4  \n\n")
    return load_uca(tmp_path), official_test_videos(tmp_path / "Anomaly_Test.txt")


def test_uca_loader_and_heuristics(tmp_path):
    rows, test_videos = _uca(tmp_path)
    assert len(rows) == 180 and test_videos == {"Abuse000_x264", "Arson001_x264"}
    assert stratum_guess("A man opens the gate") == "simple_action"
    assert stratum_guess("The car is parked near the wall") == "state_like"
    assert stratum_guess("A man walks in. Then he runs") == "compound"
    assert rule_draft("A man walks in. Then he runs away.") == "A man walks in"
    (tmp_path / "UCFCrime_Val.json").write_text(json.dumps({"x": {"duration": 1, "timestamps": [[0, 1]], "sentences": []}}))
    with pytest.raises(ValueError, match="unexpected UCA entry"):
        load_uca(tmp_path)


def test_sample_excludes_official_test_videos_and_is_deterministic(tmp_path):
    rows, test_videos = _uca(tmp_path)
    strata = {"simple_action": 6, "state_like": 4, "compound": 5, "negative_not_yet": 3, "negative_mismatch": 3}
    items, summary = sample_feasibility(rows, test_videos, seed=7, strata=strata, double=6)
    again, _ = sample_feasibility(rows, test_videos, seed=7, strata=strata, double=6)
    assert items == again and len(items) == 21 and summary["per_stratum"] == strata and not summary["unfilled"]
    videos = [i["video"] for i in items] + [i["review_video"] for i in items if i["review_video"] != i["video"]]
    assert len(videos) == len(set(videos)) and not set(videos) & test_videos  # one item per video, no test video
    assert summary["excluded_before_sampling"]["official_test"] == 6
    assert sum(i["double_annotated"] for i in items) == 6
    not_yet = [i for i in items if i["stratum"] == "negative_not_yet"]
    assert all(i["clip_end"] == i["raw_start"] - 2.0 and i["negative_candidate"] == "not_yet" for i in not_yet)
    mismatch = [i for i in items if i["stratum"] == "negative_mismatch"]
    assert all(i["review_video"] != i["video"] and i["review_video"][:5] == i["video"][:5] for i in mismatch)
    assert any(i["interpretive_wording"] for i in items if i["stratum"] == "compound")


def _item(**kw):
    base = {"record_version": RECORD_VERSION, "id": "p1f-000", "stratum": "simple_action", "video": "v", "review_video": "v",
            "duration": 100.0, "raw_start": 20.0, "raw_end": 30.0, "source_sentence": "A man opens the gate",
            "rule_draft": "A man opens the gate", "rule_source": "uca_rewrite", "negative_candidate": None,
            "clip_end": None, "double_annotated": False}  # fmt: skip
    return {**base, **kw}


def _accepted(annotator="a1", **kw):
    rec = blank_record(_item(**{k: kw.pop(k) for k in list(kw) if k in ("id", "stratum", "clip_end", "raw_start")}), annotator)
    rec.update({"rule": "Alert when a person opens the gate", "decision": "accepted", "event_start": 20.0,
                "candidate_ready": 23.0, "evidence_ready": 23.0, "valid_end": 40.0, "evidence_note": "gate visibly open",
                "prefix_checked": True, "earlier_checked": True, "seconds_spent": 90})  # fmt: skip
    rec.update(kw)
    return rec


def test_record_validation_rules():
    assert validate_record(_accepted()) == []
    assert validate_record(_accepted(evidence_ready=[22.0, 24.0], valid_end=None)) == []
    assert any("prefix clip" in p for p in validate_record(_accepted(prefix_checked=False)))
    assert any("historical-fact" in p for p in validate_record(_accepted(rule_semantics="historical")))
    assert validate_record(_accepted(rule_semantics="historical", valid_end=None)) == []
    assert any("after evidence_ready" in p for p in validate_record(_accepted(valid_end=22.0)))
    assert any("outside the video" in p for p in validate_record(_accepted(evidence_ready=500.0)))
    assert any("rule text missing" in p for p in validate_record(_accepted(rule=" ")))
    assert any("is empty" in p for p in validate_record(_accepted(uncertain_intervals=[[5.0, 5.0]])))
    negative = _accepted(negative_type="not_yet", evidence_ready=None, valid_end=None, clip_end=18.0)
    assert validate_record(negative) == []
    assert any("negative record" in p for p in validate_record({**negative, "evidence_ready": 10.0}))
    assert any("after the end of the reviewed clip" in p for p in validate_record(_accepted(clip_end=18.0)))
    undecided = {**_accepted(), "decision": "cannot_determine"}
    assert validate_record(undecided) == ["exclusion_reason is required unless accepted"]
    assert validate_record({**undecided, "exclusion_reason": "intent cannot be seen"}) == []
    assert validate_record({**_accepted(), "decision": ""})[0].startswith("decision must be")


def test_report_gives_size_and_direction_of_time_differences_and_agreement():
    records = [_accepted(id=f"p1f-{i:03d}", evidence_ready=20.0 + d) for i, d in enumerate([0.0, 1.0, 5.0, 6.0, -3.0])]
    records.append(_accepted(id="p1f-002", annotator="a2", evidence_ready=26.5))  # second annotator, 1.5 s apart
    records.append(_accepted(id="p1f-001", annotator="a2", decision="cannot_determine", exclusion_reason="occluded"))
    records.append({**_accepted(id="p1f-010"), "decision": "excluded", "exclusion_reason": "intent"})
    records.append(_accepted(id="p1f-011", negative_type="never", evidence_ready=None, valid_end=None))
    records.append({**_accepted(id="p1f-012"), "seconds_spent": None})  # invalid: dropped and listed
    r = feasibility_report(records)
    assert list(r["invalid_records"]) == ["p1f-012/a1"] and r["items"] == 7
    assert r["decisions"] == {"accepted": 6, "excluded": 1} and r["negatives"] == {"never": 1}
    d = r["time_difference"]
    assert d["n"] == 5 and d["evidence_minus_raw_start_s"]["median"] == 1.0
    assert d["abs_within_near_share"] == 0.4 and d["raw_earlier_than_evidence_share"] == 0.4
    assert d["raw_later_than_evidence_share"] == 0.2
    assert r["largest_differences"][0]["difference_s"] == 6.0
    a = r["agreement"]
    assert a["double_annotated_items"] == 2 and a["same_decision"] == 1 and a["both_positive"] == 1
    assert a["evidence_ready_gap_s"]["median"] == 1.5 and a["gap_within_near_share"] == 1.0
    assert r["seconds_spent"]["total_hours"] == round(9 * 90 / 3600, 2)


def test_prefix_plan_never_passes_the_clip_limit_and_hides_times_in_names():
    plan = prefix_plan(_accepted(candidate_ready=23.0))
    assert [p["end_seconds"] for p in plan] == [23.0, 21.0, 19.0]
    assert all(str(p["end_seconds"]) not in p["name"] and p["name"].endswith(".mp4") for p in plan)
    assert len({p["name"] for p in plan}) == 3
    limited = prefix_plan({**_accepted(candidate_ready=None), "clip_end": 18.0})  # falls back to raw_start 20
    assert [p["end_seconds"] for p in limited] == [18.0, 16.0]
    assert prefix_plan({**_accepted(candidate_ready=[1.0, 2.0])})[0]["end_seconds"] == 1.5


def test_exported_prefix_contains_no_frame_after_the_end_time(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from trace_vl.event_state.annotation import export_prefix, export_prefixes

    source = tmp_path / "videos" / "cat" / "v.mp4"
    source.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(100):  # 10 s at 10 fps; brightness encodes the frame index
        writer.write(np.full((48, 64, 3), i * 2, dtype=np.uint8))
    writer.release()
    info = export_prefix(source, 3.0, tmp_path / "clip.mp4")
    assert info["frames"] == 31 and info["last_frame_time"] == 3.0
    cap = cv2.VideoCapture(str(tmp_path / "clip.mp4"))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(int(frame.mean()))
    cap.release()
    assert len(frames) == 31 and max(frames) <= 30 * 2 + 6  # nothing brighter than frame 30 (codec tolerance)
    with pytest.raises(ValueError, match="no frames"):
        export_prefix(source, 50.0, tmp_path / "none.mp4", start_seconds=40.0)
    result = export_prefixes([_accepted(candidate_ready=6.0), {**_accepted(id="p1f-001"), "review_video": "missing"}],
                             tmp_path / "videos", tmp_path / "clips")  # fmt: skip
    assert result["clips"] == 3 and result["problems"][0]["problem"] == "missing"
    key = json.loads((tmp_path / "clips-key.json").read_text())
    assert sorted(k["end_seconds"] for k in key) == [2.0, 4.0, 6.0]
    assert not list((tmp_path / "clips").glob("*key*"))  # the key is not handed out with the clips
