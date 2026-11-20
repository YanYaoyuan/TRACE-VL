"""Temporal diagnostic: fixed categories, video-disjoint diagnostic/frozen halves, paired report and decision."""

import json

from trace_vl.v3.temporal import build_diagnostic, category, temporal_report, video_half


def ovo_records(n_videos=40):
    rows = []
    for v in range(n_videos):
        for task in ("backward/ASI", "realtime/ACR", "realtime/OJR", "realtime/OCR", "backward/EPM", "forward/REC"):
            rows.append({"id": f"ovo-{v}-{task}", "task": task, "video": f"v{v}.mp4",
                         "query_time": 30.0 + v * 5, "answer": 0, "kind": "mcq", "options": ["a", "b"]})
    return rows


def test_categories_are_fixed_by_task():
    assert category({"task": "backward/ASI"}) == "temporal" and category({"task": "forward/REC"}) == "temporal"
    assert category({"task": "realtime/OJR"}) == "static" and category({"task": "backward/EPM"}) is None


def test_split_is_video_disjoint_capped_and_bounded(tmp_path):
    src = tmp_path / "ovo.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in ovo_records()))
    m = build_diagnostic(str(src), str(tmp_path / "out"), per_task=5, max_query_seconds=150.0)
    diag = [json.loads(s) for s in (tmp_path / "out" / "diagnostic.jsonl").read_text().splitlines()]
    final = [json.loads(s) for s in (tmp_path / "out" / "final_frozen.jsonl").read_text().splitlines()]
    assert not {r["video"] for r in diag} & {r["video"] for r in final}
    assert all(float(r["query_time"]) <= 150 for r in diag + final) and all(r["category"] for r in diag)
    assert set(m["diagnostic"]["per_task"]) == {"ASI", "ACR", "OJR", "OCR", "REC"}  # EPM excluded
    assert all(n <= 5 for n in m["diagnostic"]["per_task"].values())
    assert all(video_half(r["video"]) == "diagnostic" for r in diag)
    again = build_diagnostic(str(src), str(tmp_path / "out2"), per_task=5, max_query_seconds=150.0)
    assert again["diagnostic_sha256"] == m["diagnostic_sha256"]  # deterministic


def write_preds(root, arm, records, correct):
    (root / arm).mkdir(parents=True)
    rows = [{"id": r["id"], "answer": 0, "prediction": 0 if correct(r) else 1, "video_key": r["video"],
             "lm_tokens_per_step": 46.9} for r in records]
    (root / arm / "predictions.0of1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


def test_report_decides_continue_only_with_a_temporal_specific_gain(tmp_path):
    records = []
    for v in range(30):
        for task, cat in (("backward/ASI", "temporal"), ("realtime/OJR", "static")):
            for q in range(3):
                records.append({"id": f"{v}-{task}-{q}", "task": task, "video": f"v{v}", "category": cat})
    rec = tmp_path / "diag.jsonl"
    rec.write_text("".join(json.dumps(r) + "\n" for r in records))
    # b4 / native / after right on 2 of 3 per video and category; transition right on all temporal, same static.
    base = lambda r: not r["id"].endswith("-2")  # noqa: E731
    root = tmp_path / "eval"
    for arm in ("native", "after", "b4"):
        write_preds(root, arm, records, base)
    write_preds(root, "transition", records, lambda r: True if r["category"] == "temporal" else base(r))
    report = temporal_report(str(root), str(rec))
    t = report["paired"]["transition_minus_b4"]
    assert t["temporal"]["delta_pp"] == 33.333 and t["static"]["delta_pp"] == 0.0
    assert t["interaction"]["ci95_pp"][0] > 0 and report["decision"]["continue_transition_track"] is True
    # Same gain on static as on temporal: no interaction -> stop.
    root2 = tmp_path / "eval2"
    for arm in ("native", "after", "b4"):
        write_preds(root2, arm, records, base)
    write_preds(root2, "transition", records, lambda r: True)
    assert temporal_report(str(root2), str(rec))["decision"]["continue_transition_track"] is False
