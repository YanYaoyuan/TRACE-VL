"""CPU-only tests for the inspection pipeline; no VLM weights or videos needed."""

import json

import numpy as np
import pytest
from PIL import Image

from trace_vl.inspection import meva
from trace_vl.inspection.alerts import predicted_alerts
from trace_vl.inspection.change import ChangeDetector, DetectorConfig, Transition
from trace_vl.inspection.evidence import METHODS, PackConfig, Packer, fit, tokens_of
from trace_vl.inspection.memory import TransitionStore


def scene(t, present):
    frame = np.full((120, 160, 3), 100, np.uint8)
    if present:
        frame[40:60, 70:90] = 250
    return frame


def run_detector(put=10.0, take=40.0, end=60.0):
    cfg = DetectorConfig(work_width=160, confirm_seconds=3.0, vanish_seconds=2.0, short_seconds=2.0)
    detector, events = ChangeDetector(cfg), []
    for t in np.arange(0, end, 0.5):
        for event, transition in detector.step(scene(t, put <= t < take), float(t)):
            events.append((event, float(t), transition))
    return events


def test_detector_reports_causal_appear_and_resolve_with_onset_times():
    events = run_detector()
    kinds = [e[0] for e in events]
    assert kinds == ["appear", "resolve"]
    (_, t_appear, transition), (_, t_resolve, _) = events
    assert transition.t_known == t_appear and 12.0 <= t_appear <= 20.0
    assert abs(transition.t_start - 10.0) <= 0.5  # onset, not the confirmation lag
    assert abs(transition.t_end - 40.0) <= 0.5 and transition.t_end_known == t_resolve
    assert t_resolve - transition.t_end >= 2.0
    x1, y1, x2, y2 = transition.box
    assert x1 < 90 / 160 and x2 > 70 / 160 and y1 < 60 / 120 and y2 > 40 / 120
    assert transition.before.mean() < 110 and transition.after.mean() > transition.before.mean() + 30


def test_view_hides_future_resolution():
    _, (_, t_resolve, transition) = run_detector()
    assert transition.view_at(transition.t_known - 1) is None
    early = transition.view_at(t_resolve - 0.5)
    assert early.status == "active" and early.t_end is None and early.after_end is None
    assert transition.view_at(t_resolve).status == "resolved"


def test_global_change_resets_instead_of_flooding_memory():
    detector = ChangeDetector(DetectorConfig(work_width=160))
    detector.step(np.full((120, 160, 3), 100, np.uint8), 0.0)
    events = detector.step(np.full((120, 160, 3), 200, np.uint8), 0.5)
    assert [e[0] for e in events] == ["global"]


def _transition(i, status="active", t=0.0):
    crop = np.zeros((8, 8, 3), np.uint8)
    item = Transition(i, (0.1, 0.1, 0.2, 0.2), t, t, crop, crop, status=status)
    if status == "resolved":
        item.t_end, item.t_end_known = t + 1, t + 2
    return item


def test_store_evicts_resolved_before_active_and_snapshots_causally():
    store = TransitionStore(capacity=2)
    store.update([("appear", _transition(1, "active", 0)), ("appear", _transition(2, "resolved", 1))])
    store.update([("appear", _transition(3, "active", 5))])
    assert sorted(store.items) == [1, 3] and store.evicted == 1
    assert [t.id for t in store.snapshot(2.0)] == [1]
    store.update([("resolve", _transition(2, "resolved", 1))])  # evicted items are not resurrected
    assert 2 not in store.items


def test_fit_respects_token_budget_and_unit():
    image = fit(Image.new("RGB", (1920, 1080)), 100, 32)
    assert image.size[0] % 32 == 0 and image.size[1] % 32 == 0
    assert 80 <= tokens_of(image, 32) <= 100


@pytest.fixture
def prepared(tmp_path):
    def save(name, color):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), color).save(path)
        return name

    transitions = [
        {
            "id": i,
            "kind": "region",
            "box": [0.4, 0.4, 0.5, 0.5],
            "t_start": 10.0 * i,
            "t_known": 10.0 * i + 5,
            "t_end": None,
            "t_end_known": None,
            "status": "active",
            "crops": {"before": save(f"t/{i}_b.jpg", "gray"), "after": save(f"t/{i}_a.jpg", "white")},
        }
        for i in range(1, 20)
    ]
    snapshot = {
        "id": "q",
        "stream": "s",
        "query_time": 300.0,
        "frame_time": 300.0,
        "window": 300,
        "region": [0.3, 0.3, 0.6, 0.6],
        "current": save("cur.jpg", "blue"),
        "uniform": [{"t": 20.0 * i, "image": save(f"u{i}.jpg", "red")} for i in range(16)],
        "recent": [{"t": 290.0 + i * 0.5, "image": save(f"r{i}.jpg", "green")} for i in range(16)],
        "transitions": transitions,
        "transitions_known": len(transitions),
    }
    (tmp_path / "streams" / "s").mkdir(parents=True)
    (tmp_path / "streams" / "s" / "captions.jsonl").write_text(
        "".join(json.dumps({"t": t, "caption": f"scene at {t}"}) + "\n" for t in (100.0, 200.0, 400.0))
    )
    question = {"question": "Was it moved?", "options": ["Yes", "No"], "answer": 0}
    return tmp_path, snapshot, question


@pytest.mark.parametrize("method", METHODS)
def test_every_method_stays_within_budget(prepared, method):
    root, snapshot, question = prepared
    cfg = PackConfig(frame_tokens=64, budget_tokens=256, crop_tokens=32, frames=4)
    segments, info = Packer(root, cfg).pack(snapshot, question, method)
    images = [s for s in segments if s["type"] == "image"]
    assert info["planned_visual_tokens"] <= cfg.frame_tokens + cfg.budget_tokens
    assert segments[-1]["text"].startswith("Answer with the option letter")
    if method == "none":
        assert not images
    if method == "transition":
        assert info["transitions_shown"] == 256 // (2 * 32)
    if method == "textlog":
        text = "\n".join(s["text"] for s in segments if s["type"] == "text")
        assert "scene at 200.0" in text and "scene at 400.0" not in text  # no captions from the future


def _clip(index, events):
    start = 10 * 3600 + index * 300
    name = f"2018-03-07.10-{index * 5:02d}-00.10-{index * 5 + 5:02d}-00.admin.G329"
    return {**meva.parse_clip(name), "start": start, "end": start + 300, "events": events}


def _event(kind, frame, box):
    return {
        "kind": kind,
        "frames": [frame, frame + 60],
        "object_id": frame,
        "object_type": "bag",
        "object_frames": [frame, frame + 60],
        "box_first": box,
        "box_last": box,
    }


def test_meva_streams_link_and_questions_are_consistent():
    box = [900, 500, 960, 560]
    records = [_clip(i, []) for i in range(6)]
    records[2]["events"] = [_event("put", 3000, box)]  # t = 700 s
    records[3]["events"] = [_event("pick", 3000, box)]  # t = 1000 s
    streams = meva.build_streams(records)
    assert len(streams) == 1
    stream_id, stream = next(iter(streams.items()))
    assert stream["duration"] == 1800 and [c["offset"] for c in stream["clips"]] == [0, 300, 600, 900, 1200, 1500]
    events, rests = meva.link_events(stream)
    assert len(rests) == 1 and not rests[0]["censored"]
    assert rests[0]["start"] == pytest.approx(702) and rests[0]["end"] == pytest.approx(1000)
    import random

    questions = meva.questions_for(stream_id, stream, random.Random(0))
    assert {q["type"] for q in questions} >= {"placed_and_removed", "duration", "still_there", "count"}
    for q in questions:
        tq = q["query_time"]
        assert q["window"] <= tq
        if q["type"] == "placed_and_removed":
            assert q["answer"] == (0 if tq > 1002 and tq - q["window"] <= 700 else 1)
        if q["type"] == "still_there":
            assert q["answer"] == (0 if 702 <= tq < 1000 else 1)
        if q["type"] == "duration":
            seconds = tq - 702
            assert q["answer"] == sum(seconds >= edge for edge in (60, 300, 900))
        if q["type"] == "count":
            assert q["answer"] == (1 if tq - q["window"] <= 700 <= tq else 0)


def test_balance_keeps_hard_negatives():
    import random

    questions = [{"type": "t", "answer": 0, "gap": 60, "stream": "s", "query_time": i} for i in range(5)]
    questions += [{"type": "t", "answer": 1, "gap": None, "stream": "s", "query_time": i} for i in range(3)]
    kept = meva.balance(questions, random.Random(0))
    assert sorted(q["answer"] for q in kept) == [0, 0, 0, 1, 1, 1]


def test_predicted_alerts_are_causal_and_respect_duration():
    appear = {"event": "appear", "id": 1, "kind": "region", "box": [0.1, 0.1, 0.2, 0.2], "t_start": 100.0,
              "t_known": 106.0, "t_end": None, "t_end_known": None, "status": "active", "crops": {}}  # fmt: skip
    short = {**appear, "id": 2, "t_start": 500.0, "t_known": 506.0}
    resolve_short = {**short, "event": "resolve", "t_end": 560.0, "t_end_known": 562.0, "status": "resolved"}
    alerts = predicted_alerts([appear, short, resolve_short], min_duration=120)
    assert [(a["id"], a["alert_time"]) for a in alerts] == [(1, 220.0)]
