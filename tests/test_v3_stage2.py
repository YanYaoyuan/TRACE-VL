"""CPU tests for Stage 2 (long-stream training through the streaming cache). No Qwen3.5 weights needed."""

import json

import numpy as np
import pytest
import torch

from trace_vl.v3.backbone import branch_cache, detach_cache, functional_linear_cache
from trace_vl.v3.config import MemoryConfig, V3Config, load_v3_config
from trace_vl.v3.kv import KVMemory
from trace_vl.v3.train_stage2 import chunk_plan, evidence_kept, evidence_label, stream_times, summarize


def test_stage2_configs_load_and_validate():
    cfg, _ = load_v3_config("configs/v3/stage2.yaml")
    assert cfg.memory.policy == "tkm" and cfg.stage2.chunk_steps == 8
    b4, _ = load_v3_config("configs/v3/stage2_b4.yaml")
    assert b4.transition.p_step_mode == "pooled" and b4.memory.policy == "tkm"
    a100, _ = load_v3_config("configs/v3/stage2_a100.yaml")
    assert a100.stage2.max_seconds == 900.0
    with pytest.raises(ValueError, match="stage2"):
        V3Config.from_dict({"stage2": {"chunk_steps": 0}})
    with pytest.raises(ValueError, match="unknown"):
        V3Config.from_dict({"stage2": {"chunks": 3}})


def test_chunk_plan_cuts_at_questions_and_caps_length():
    times = stream_times({"duration": 20.0}, 1.0, 0.5, 600.0)
    assert times[0] == 0.25 and times[-1] + 0.25 <= 20.0
    questions = [{"id": "a", "time": 5.6}, {"id": "b", "time": 5.6}, {"id": "c", "time": 0.1},
                 {"id": "d", "time": 19.9}, {"id": "e", "time": 99.0}]  # fmt: skip
    plan = chunk_plan(times, questions, 4, 0.5)
    flat = [k for steps, _ in plan for k in steps]
    assert flat == list(range(len(times)))  # every step exactly once, in order
    assert all(len(steps) <= 4 for steps, _ in plan)
    asked = {q["id"]: steps[-1] for steps, due in plan for q in due}
    # "a"/"b" at 5.6 s see steps whose window ends <= 5.6 (t=5.25 ends 5.5); "c" precedes the first step;
    # "e" is long after the stream and is dropped.
    assert asked == {"a": times.index(5.25), "b": times.index(5.25), "d": len(times) - 1}
    assert stream_times({"duration": 0.2}, 1.0, 0.5, 600.0) == [] and chunk_plan([], questions, 4, 0.5) == []


def test_stream_times_respect_max_seconds():
    times = stream_times({"duration": 1000.0}, 1.0, 0.5, 30.0)
    assert times[-1] + 0.25 <= 30.0 and len(times) == 30


def test_evidence_label_only_for_later_questions():
    questions = [{"time": 50.0, "evidence": [[10.0, 20.0]]}, {"time": 15.0, "evidence": [[30.0, 40.0]]}]
    assert evidence_label(12.0, questions) == 1.0
    assert evidence_label(35.0, questions) == 0.0  # its question (t=15) was asked before the evidence
    assert evidence_label(25.0, questions) == 0.0
    assert evidence_label(12.0, [{"time": 50.0, "evidence": None}]) == 0.0


def test_evidence_kept_counts_visual_tokens_in_cache():
    memory = KVMemory(MemoryConfig(policy="window", window_seconds=5.0))
    memory.append(3, "prompt", -1.0, ("prompt",), protected=True)
    appended = []
    for k, t in enumerate([1.0, 2.0, 10.0]):
        memory.append(2, "text", t, ("step", k))
        memory.append(4, "gist", t, ("step", k))
        appended.append((t, 4))
    assert evidence_kept(memory, appended, [[0.5, 2.5]]) == 1.0
    mask = memory.keep_mask(10.0)  # window keeps only t >= 5 (and the prompt)
    memory.apply(mask)
    assert evidence_kept(memory, appended, [[0.5, 2.5]]) == 0.0
    assert evidence_kept(memory, appended, [[9.0, 11.0]]) == 1.0
    assert evidence_kept(memory, appended, None) is None and evidence_kept(memory, appended, [[50, 60]]) is None


def test_summarize_handles_empty_and_mixed_streams():
    out = summarize([{"steps": 0}, {"steps": 4, "ce": [1.0, 3.0], "correct": [1, 0], "active": [0, 2],
                                    "tokens": 40, "evidence_kept": [0.5]}])  # fmt: skip
    assert out["questions"] == 2 and out["q_ce"] == 2.0 and out["q_acc"] == 0.5
    assert out["p_steps_with_active"] == 0.5 and out["tokens_per_step"] == 10.0 and out["evidence_kept"] == 0.5


# ---------------------------------------------------------------------------- cache plumbing (tiny real model)
def tiny_qwen35():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_conv_kernel_dim=4,
                               linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                               linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"])  # fmt: skip
    torch.manual_seed(0)
    return Qwen3_5TextModel(config).eval(), config


def new_cache(config):
    from transformers import DynamicCache

    return DynamicCache(config=config)


def test_functional_cache_matches_in_place_updates():
    model, config = tiny_qwen35()
    a, b = torch.randn(1, 5, 32), torch.randn(1, 3, 32)
    with torch.no_grad():
        reference = new_cache(config)
        model(inputs_embeds=a, past_key_values=reference, use_cache=True)
        expected = model(inputs_embeds=b, past_key_values=reference, use_cache=True).last_hidden_state
        with functional_linear_cache():
            cache = new_cache(config)
            model(inputs_embeds=a, past_key_values=cache, use_cache=True)
            got = model(inputs_embeds=b, past_key_values=cache, use_cache=True).last_hidden_state
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(cache.layers[0].recurrent_states[0], reference.layers[0].recurrent_states[0])
    torch.testing.assert_close(cache.layers[0].conv_states[0], reference.layers[0].conv_states[0])
    from transformers.cache_utils import LinearAttentionLayer

    assert LinearAttentionLayer.update_recurrent_state.__name__ == "update_recurrent_state"
    assert "copy_" in LinearAttentionLayer.update_recurrent_state.__code__.co_names  # restored after the context


def test_gradient_flows_through_cache_into_a_question_fork():
    model, config = tiny_qwen35()
    stream = torch.randn(1, 6, 32, requires_grad=True)
    question = torch.randn(1, 3, 32)
    with functional_linear_cache():
        cache = new_cache(config)
        model(inputs_embeds=stream, past_key_values=cache, use_cache=True)
        state_before = cache.layers[0].recurrent_states[0].detach().clone()
        length_before = cache.get_seq_length()
        fork = branch_cache(cache)
        out = model(inputs_embeds=question, past_key_values=fork, use_cache=True).last_hidden_state
        out.square().mean().backward()
    assert stream.grad is not None and float(stream.grad.abs().sum()) > 0  # through KV and GDN states
    # The fork never leaks into the live stream.
    assert cache.get_seq_length() == length_before and fork.get_seq_length() == length_before + 3
    torch.testing.assert_close(cache.layers[0].recurrent_states[0].detach(), state_before)
    detach_cache(cache)
    assert not cache.layers[0].recurrent_states[0].requires_grad and not cache.layers[1].keys.requires_grad
    with functional_linear_cache(), torch.no_grad():  # the stream continues after detaching
        model(inputs_embeds=torch.randn(1, 2, 32), past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == length_before + 2


# ---------------------------------------------------------------------------- data builder
def write_video(path, seconds: float, fps: int = 4, size=(64, 48)):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(int(seconds * fps) + 1):
        writer.write(np.full((size[1], size[0], 3), (i * 7) % 255, dtype=np.uint8))
    writer.release()


def test_build_stage2_streams(tmp_path):
    from trace_vl.v3.data import build_stage2_streams, read_jsonl

    rows = []
    for v in range(6):
        write_video(tmp_path / f"v{v}.avi", 10.0)
        for q in range(3):
            rows.append({"id": f"q{v}-{q}", "video": f"v{v}.avi", "query_time": None, "question": f"What {q}?",
                         "options": ["a", "b"], "answer": q % 2, "split": "train"})  # fmt: skip
    write_video(tmp_path / "long.avi", 40.0)
    rows.append({"id": "timed", "video": "long.avi", "query_time": 30.0, "question": "Where?", "options": ["x", "y"],
                 "answer": 1, "split": "train", "evidence": [[5.0, 12.0]]})  # fmt: skip
    rows.append({"id": "missing", "video": "gone.avi", "query_time": None, "question": "?", "options": ["a", "b"],
                 "answer": 0, "split": "train"})  # fmt: skip
    rows.append({"id": "val", "video": "v0.avi", "query_time": None, "question": "?", "options": ["a", "b"],
                 "answer": 0, "split": "val"})  # fmt: skip
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = build_stage2_streams(str(tmp_path / "records.jsonl"), str(tmp_path / "streams.jsonl"), str(tmp_path),
                                  stream_seconds=25.0, questions_per_clip=2, delay_min=5.0, delay_max=8.0,
                                  immediate_fraction=0.5)  # fmt: skip
    streams = read_jsonl(tmp_path / "streams.jsonl")
    assert report["streams"] == len(streams) and report["skipped_records"] == 1
    timed = [s for s in streams if s["source"] == "timed"]
    assert len(timed) == 1 and timed[0]["questions"][0]["evidence"] == [[5.0, 12.0]]
    concat = [s for s in streams if s["source"] == "concat"]
    assert sum(len(s["clips"]) for s in concat) == 6  # every readable offline video used once
    for s in concat:
        assert s["questions"] == sorted(s["questions"], key=lambda q: q["time"])
        for clip, nxt in zip(s["clips"], s["clips"][1:]):
            assert nxt["offset"] == pytest.approx(clip["end"], abs=1e-3)
        for q in s["questions"]:
            (a, b), = q["evidence"]
            assert q["time"] >= b - 1e-6 and q["time"] <= s["duration"] + 1e-6  # asked from memory
            assert q["question"].startswith(f"Consider the part of the video from {a:.0f} to {b:.0f} seconds.")
            assert q["delay"] == pytest.approx(q["time"] - b, abs=1e-3)
    assert all(q["id"] != "val" for s in streams for q in s["questions"])
    with pytest.raises(ValueError, match="split"):
        build_stage2_streams(str(tmp_path / "records.jsonl"), str(tmp_path / "x.jsonl"), str(tmp_path), split="test")
