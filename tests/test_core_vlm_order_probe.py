import hashlib
import json
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_core_vlm_native import tiny_example

from trace_vl.core_vlm import order_probe
from trace_vl.core_vlm.native import NativeVideoQALoss, encode_native
from trace_vl.core_vlm.temporal import attach_temporal
from trace_vl.core_vlm.train import tensor_digest


def test_real_decode_reorders_exactly_same_frames():
    import tempfile
    from pathlib import Path

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    with tempfile.TemporaryDirectory() as directory:
        video = Path(directory) / "source.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=s=320x224:r=16:d=4", "-threads", "1", "-c:v", "ffv1", str(video)], check=True)
        pair = {"id": "p", "video": str(video), "video_sha256": order_probe.sha256(video),
                "actions": ["first scene", "second scene"], "ask": "last", "spans": [[.05, 1.4], [2., 3.7]]}
        halves, receipt = order_probe.sample_pair(pair)
        left, lt, lr, la = order_probe.packed_view(pair, halves, False)
        right, rt, rr, ra = order_probe.packed_view(pair, halves, True)
        assert len(receipt["sampling"][0]["indices"]) == len(receipt["sampling"][1]["indices"]) == 8
        assert left.shape[:2] == right.shape[:2] == (8, 2)
        assert np.array_equal(left[:4], right[4:]) and np.array_equal(left[4:], right[:4])
        assert np.array_equal(lt, rt) and la["frame_indices"] == ra["frame_indices"]
        assert lr["options"] == rr["options"] and lr["question"] == rr["question"]
        assert lr["answer"] != rr["answer"]
        assert la["decoded_rgb_sha256"] == hashlib.sha256(left.tobytes()).hexdigest()
        single, _, _, _ = order_probe.packed_view(pair, halves, False, single=True)
        assert np.all(single == single[0, 0])


def test_exact_video_copy_cannot_cross_splits(tmp_path):
    first, second = tmp_path / "1.mp4", tmp_path / "2.mp4"
    first.write_bytes(b"same-media")
    second.write_bytes(b"same-media")
    rows = [{"id": str(i), "video": str(p), "source_group": f"g{i}", "split": split,
             "actions": ["up", "down"], "ask": "first", "spans": [[0, 1], [2, 3]]}
            for i, (p, split) in enumerate(zip((first, second), ("train", "val"), strict=True))]
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="same group and split"):
        order_probe.load_pairs(path)


def test_adapter_updates_through_truly_frozen_qwen_decoder():
    bb, layout, row = tiny_example()
    steps = encode_native(bb, layout)
    steps_reversed = [replace(step, tokens=other.tokens) for step, other in zip(steps, reversed(steps), strict=True)]
    net = NativeVideoQALoss(bb, mode="grid", pool_grid=(1, 1))
    before = tensor_digest(dict(bb.model.named_parameters()))
    attach_temporal(net, "causal")
    initial = net.temporal.up.weight.detach().clone()
    metrics = list(order_probe.fit(net, [[(steps, row), (steps_reversed, {**row, "answer_text": "sit"})]],
                                   updates=3, seed=17, lr=1e-3))
    assert len(metrics) == 3 and all(x["adapter_grad_norm"] > 0 for x in metrics)
    assert not torch.equal(net.temporal.up.weight, initial)
    assert tensor_digest(dict(bb.model.named_parameters())) == before
    assert all(x.grad is None and not x.requires_grad for x in bb.model.parameters())


def test_pair_metric_is_not_question_accuracy_or_flips():
    rows = [{"source_group": "a", "gold": [0, 1], "plain": [0, 0],
             "single_frame": [1, 0], "text_only": [1, 1], "adapter": [0, 1]},
            {"source_group": "b", "gold": [1, 0], "plain": [0, 1],
             "single_frame": [0, 0], "text_only": [1, 1], "adapter": [1, 0]}]
    result = order_probe.summarize(rows)
    assert result["arms"]["plain"] == {"question_accuracy": .25, "pair_accuracy": 0., "prediction_flip_rate": .5}
    assert result["arms"]["adapter"]["pair_accuracy"] == 1.
    assert result["paired"]["adapter_minus_plain"]["ci95_pp"] == [100., 100.]
