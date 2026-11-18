import copy

import pytest
import torch
from test_core_vlm_native import tiny_example

from trace_vl.core_vlm.native import NativeVideoQALoss, encode_native
from trace_vl.core_vlm.preservation import original_answer_logits
from trace_vl.core_vlm.temporal import attach_temporal
from trace_vl.core_vlm.train import sample_indices


def batch_example(temporal=True):
    bb, layout, row = tiny_example()
    bb.add_lora(2, 4, 0.)
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=(1, 1), answer_end_token_id=2)
    if temporal:
        attach_temporal(net, "causal")
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if "lora_B" in name or name == "temporal.up.weight":
                parameter.normal_(0, .02)
    steps = encode_native(bb, layout)
    other = copy.deepcopy(row)
    inputs = other["native_layout"].inputs
    # Different prompt and answer lengths exercise padding and per-question
    # weighting rather than a token-weighted average.
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[11, 12, 13]])], 1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 3, dtype=torch.long)], 1)
    other["answer_text"] = "r"
    return bb, net, [steps, steps], [row, other]


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
def test_batched_hybrid_model_loss_and_gradients_match_accumulation(checkpointed, temporal):
    bb, net, steps, records = batch_example(temporal)
    if checkpointed:
        bb.enable_checkpointing()
    net.train()
    teachers = [original_answer_logits(net, steps[0], records[0]), None]
    expected_loss = 0
    for i in range(2):
        loss, _ = net(steps[i], records[i], teacher_logits=teachers[i],
                      preservation_weight=.5 if i == 0 else 0.)
        (loss / 2).backward()
        expected_loss += loss.detach() / 2
    gradients = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)
    batched_teacher = original_answer_logits(net, [steps[0]], [records[0]])[0]
    torch.testing.assert_close(batched_teacher, teachers[0][0], atol=1e-6, rtol=1e-5)
    actual, metadata = net(steps, records, teacher_logits=[batched_teacher, None], preservation_weight=.5)
    torch.testing.assert_close(actual, expected_loss, atol=1e-6, rtol=1e-5)
    actual.backward()
    assert metadata["answer_tokens"] == 6  # run + EOS, r + EOS
    assert metadata["teacher_questions"] == 1
    assert metadata["padded_sequence_tokens"] > metadata["sequence_tokens"]
    for name, parameter in net.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-6, rtol=3e-4)
    assert all(p.grad is None for p in net.parameters() if not p.requires_grad)


def test_samples_are_independent_with_mixed_lengths_and_changed_neighbour():
    _, net, steps, records = batch_example()
    net.eval()
    first, _, _ = net.answer_logits_batch(steps, records)
    other = copy.deepcopy(records)
    other[1]["answer_text"] = "different answer"
    modified_steps = [steps[0], [copy.copy(s) for s in steps[1]]]
    for step in modified_steps[1]:
        step.tokens = step.tokens + 100
    second, _, _ = net.answer_logits_batch(modified_steps, other)
    torch.testing.assert_close(first[0], second[0], atol=1e-6, rtol=1e-5)


def test_regrouping_preserves_each_rank_question_order_across_epochs():
    for update in range(8):
        for rank in range(4):
            expected = sample_indices(19, update, 4, 4, rank, 17)
            for microbatch, accumulation in ((1, 4), (2, 2), (4, 1)):
                selected = sample_indices(19, update, 4, microbatch * accumulation, rank, 17)
                assert selected == expected


def test_empty_or_mismatched_batches_fail():
    _, net, steps, records = batch_example()
    with pytest.raises(ValueError, match="matching"):
        net.answer_logits_batch([], [])
    with pytest.raises(ValueError, match="matching"):
        net.answer_logits_batch(steps[:1], records)
    with pytest.raises(ValueError, match="teacher batch"):
        net(steps, records, teacher_logits=[None])


def test_benchmark_selects_common_safe_configuration():
    from trace_vl.core_vlm.batch_benchmark import recommend
    trials = {f"{arm}-mb{size}": {"passed": True, "question_ids_by_update": [[["a", "b"]]],
              "initial_lora": "same", "first_loss": 1., "peak_reserved_gib_max": 12. * size,
              "median_seconds_per_update": 12. / size}
              for arm in ("plain", "temporal") for size in (1, 2, 4)}
    # Batch 4 is faster but exceeds the memory reserve.
    assert recommend(trials)["microbatch_size"] == 2
    trials["temporal-mb2"]["question_ids_by_update"] = [[["different"]]]
    assert recommend(trials)["microbatch_size"] == 1
    trials["plain-mb1"]["passed"] = False
    assert recommend(trials)["microbatch_size"] is None


def test_benchmark_entry_reuses_prepared_training_data_and_runs_six_trials(tmp_path, monkeypatch):
    import json

    from trace_vl.core_vlm import batch_benchmark
    from trace_vl.core_vlm.data import sha256
    prepared = tmp_path / "prepared"
    (prepared / "mixture").mkdir(parents=True)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"verified media")
    manifest = prepared / "mixture/manifest.jsonl"
    rows = [dict(id=str(i), video=str(video), question="q", options=["x", "y"], answer=0, split="train")
            for i in range(64)]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    saved = manifest.read_bytes()
    (prepared / "mixture/complete.json").write_text(json.dumps({"files": {"manifest.jsonl": sha256(manifest)},
                                                               "videos": {str(video): sha256(video)}}))
    cfg = dict(gpus=[0, 1, 2, 3], seed=17, model="base", steps=8, pixels=65536, pool_grid=[4, 5],
               timestamp_policy="native-index-fps", updates=2500, flow_updates=2, accumulation=4,
               rank=8, lr=2e-5, save_every=100, cache_mb=4096)
    (prepared / "config.json").write_text(json.dumps(cfg))
    commands = []
    monkeypatch.setenv("TRACE_WANDB", "0")
    monkeypatch.setattr(batch_benchmark, "run_jobs", lambda jobs, *a: commands.extend(jobs))
    monkeypatch.setattr(batch_benchmark, "trial_summary", lambda *a: {
        "passed": True, "question_ids_by_update": [["fixed"]], "initial_lora": "same", "first_loss": 1.,
        "median_seconds_per_update": 1., "samples_per_second": 16., "peak_reserved_gib_max": 12.})
    batch_benchmark.run(prepared, tmp_path / "out", steps=3, warmup=1)
    assert len(commands) == 6
    for tag, command, devices in commands:
        size = int(tag[-1])
        assert command[command.index("--microbatch-size") + 1] == str(size)
        assert command[command.index("--accumulation") + 1] == str(4 // size)
        assert command[command.index("--updates") + 1] == "2500"
        assert command[command.index("--stop-after") + 1] == "3"
        assert devices == [0, 1, 2, 3]
    assert manifest.read_bytes() == saved
    report = json.loads((tmp_path / "out/batch-report.json").read_text())
    assert report["completed"] and report["benchmark_questions"] == 64
