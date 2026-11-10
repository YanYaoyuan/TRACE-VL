"""Stage-2 memory pilot: frozen source-video splits, policy report and predeclared decision (CPU)."""

import json

import pytest

from trace_vl.v3.config import V3Config, load_v3_config
from trace_vl.v3.memory_pilot import make_splits, memory_report


def test_splits_exclude_stage1_videos_and_are_video_disjoint(tmp_path):
    rows = [{"id": f"q{v}-{k}", "video": f"v{v}.mp4", "question": "?", "options": ["a", "b"], "answer": 0}
            for v in range(200) for k in range(2)]
    (tmp_path / "mc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "s1.jsonl").write_text("".join(json.dumps({"video": f"v{v}.mp4"}) + "\n" for v in range(20)))
    m = make_splits([str(tmp_path / "mc.jsonl")], [str(tmp_path / "s1.jsonl")], str(tmp_path / "out"),
                    max_videos={"train": 50, "dev": 10, "test": 10})
    split = {s: [json.loads(x) for x in (tmp_path / "out" / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "dev", "test")}
    videos = {s: {r["video"] for r in rows} for s, rows in split.items()}
    assert not (videos["train"] & videos["dev"]) and not (videos["train"] & videos["test"]) and not (videos["dev"] & videos["test"])
    assert all(int(v[1:-4]) >= 20 for vs in videos.values() for v in vs)  # Stage-1 videos excluded
    assert len(videos["train"]) == 50 and len(videos["dev"]) <= 10 and m["excluded_videos"] == 20
    assert all(r["split"] == s for s, rows in split.items() for r in rows)
    again = make_splits([str(tmp_path / "mc.jsonl")], [str(tmp_path / "s1.jsonl")], str(tmp_path / "out2"),
                        max_videos={"train": 50, "dev": 10, "test": 10})
    assert again["splits"] == m["splits"]  # deterministic, hashes included


def write_run(root, name, rows):
    (root / name).mkdir(parents=True)
    (root / name / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def rows_for(correct, kept, kv=1700, evicted=11000):
    out = []
    for s in range(30):
        for q in range(5):
            age = 5.0 if q == 0 else 60.0 + q  # q0 within the window, others beyond it
            out.append({"stream": f"s{s}", "id": f"s{s}-q{q}", "delay": age, "evidence_age": age,
                        "correct": int(correct(s, q)), "evidence_kept": 1.0 if q == 0 else kept,
                        "kv_tokens": kv, "kv_mb": 3.0, "evicted_tokens": evicted, "streamed_tokens": 13000,
                        "stream_seconds_elapsed": 20.0})
    return out


def test_report_supports_learned_retention_only_with_pressure_equal_budget_and_gain(tmp_path):
    root = tmp_path / "eval"
    write_run(root, "s2_tkm", rows_for(lambda s, q: q != 4, 0.7))          # beyond-window 3/4
    write_run(root, "s2_mass", rows_for(lambda s, q: q in (0, 1), 0.5))    # beyond-window 1/4
    write_run(root, "s2_fifo", rows_for(lambda s, q: q in (0, 1, 2), 0.3))  # beyond-window 2/4
    write_run(root, "s2_full", rows_for(lambda s, q: True, 1.0, kv=13000, evicted=0))
    r = memory_report(str(root), 16.0)
    assert r["runs"]["s2_tkm"]["beyond_window"]["n"] == 120 and r["runs"]["s2_tkm"]["within_window"]["n"] == 30
    assert r["paired"]["s2_tkm_minus_s2_mass"]["beyond_window"]["delta_pp"] == 50.0
    assert r["decision"]["memory_pressure"] and r["decision"]["equal_budget"]
    assert r["decision"]["learned_retention_supported"] is True
    # No memory pressure (FIFO keeps the evidence): not supported even with a gain.
    root2 = tmp_path / "eval2"
    write_run(root2, "s2_tkm", rows_for(lambda s, q: q != 4, 0.95))
    write_run(root2, "s2_mass", rows_for(lambda s, q: q in (0, 1), 0.95))
    write_run(root2, "s2_fifo", rows_for(lambda s, q: q in (0, 1, 2), 0.95))
    assert memory_report(str(root2), 16.0)["decision"]["learned_retention_supported"] is False
    # Unequal retained KV: not supported.
    root3 = tmp_path / "eval3"
    write_run(root3, "s2_tkm", rows_for(lambda s, q: q != 4, 0.7, kv=2500))
    write_run(root3, "s2_mass", rows_for(lambda s, q: q in (0, 1), 0.5))
    write_run(root3, "s2_fifo", rows_for(lambda s, q: q in (0, 1, 2), 0.3))
    assert memory_report(str(root3), 16.0)["decision"]["equal_budget"] is False


def test_memory_pilot_config_loads_and_evict_every_is_validated():
    cfg, _ = load_v3_config("configs/v3/memory_pilot_b4.yaml")
    assert cfg.transition.p_step_mode == "pooled" and cfg.memory.policy == "tkm" and cfg.memory.budget_tokens == 1024
    assert cfg.memory.window_seconds == 16.0 and cfg.stage2.steps == 300
    with pytest.raises(ValueError, match="evict_every"):
        V3Config.from_dict({"memory": {"evict_every": 0}})
