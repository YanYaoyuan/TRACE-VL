import hashlib
import subprocess
from dataclasses import replace

import numpy as np
import torch
from test_core_longstream import step
from test_core_vlm_native import tiny_example

from trace_vl.core_vlm.longstream import answer
from trace_vl.core_vlm.native import NativeVideoQALoss
from trace_vl.core_vlm.qa import VisualStep, grid_pool
from trace_vl.core_vlm.snapshot_readout import dense_answer, verified_pixels


def test_real_native_readout_keeps_spatial_information_lost_by_grid_without_vision_or_gold(monkeypatch):
    bb, _, _ = tiny_example()

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][-1]["text"]

        def __call__(self, *, text, videos, **kwargs):
            assert not videos[0].any()
            count = len(videos[0]) // 2
            ids = torch.tensor([[5] + ([1] + [3] * 36 + [2, 8]) * count + [6, 7]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                    "mm_token_type_ids": torch.where(ids == 3, 2, 0),
                    "video_grid_thw": torch.tensor([[count, 12, 12]])}

    bb.processor = Processor()

    def forbidden(*args, **kwargs):
        raise AssertionError("readout must consume supplied features without re-encoding")

    monkeypatch.setattr(bb, "encode", forbidden)
    monkeypatch.setattr(bb.visual, "forward", forbidden)
    tokens = torch.randn(36, 32)
    tokens[[7, 8, 13, 14]] = 0  # one2x2 spatial bin, exactly cancelling perturbation below
    changed = tokens.clone()
    delta = torch.randn(32) * 2
    changed[7] = delta
    changed[8] = -delta
    pooled = grid_pool(tokens, (6, 6), (4, 4))[0]
    assert torch.equal(pooled, grid_pool(changed, (6, 6), (4, 4))[0])
    stored = [replace(step(0, (6, 6)), tokens=pooled)]
    query = {"question": "Which action?", "options": ["a", "b"], "answer": 1, "query_time": 1.0}
    metadata = {"streams": [{"avg_frame_rate": "2/1"}], "frames": [{}] * 4}
    before = dense_answer(bb, [VisualStep(tokens, (6, 6), .5, (0., .5))], stored, query, metadata, (192, 192), "video")
    after = dense_answer(bb, [VisualStep(changed, (6, 6), .5, (0., .5))], stored, query, metadata, (192, 192), "video")
    gold_changed = dense_answer(bb, [VisualStep(tokens, (6, 6), .5, (0., .5))], stored,
                                {**query, "answer": 0}, metadata, (192, 192), "video")
    assert before == gold_changed
    assert before["visual_tokens"] == after["visual_tokens"] == 36
    assert before["option_logits_fp32"] != after["option_logits_fp32"]
    compressed = answer(bb, NativeVideoQALoss(bb, mode="grid", pool_grid=(4, 4)), stored,
                        query, metadata, (192, 192), "video")
    assert compressed["visual_tokens"] == 16 and before["query_time_encoder_calls"] == 0


def test_reference_seek_mismatch_falls_back_to_exact_real_decoder(tmp_path, monkeypatch):
    from trace_vl.core_vlm import snapshot_readout

    frames = np.stack([np.full((32, 32, 3), 20 * i, np.uint8) for i in range(4)])
    path = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32",
                    "-r", "2", "-i", "pipe:0", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)],
                   input=frames.tobytes(), capture_output=True, check=True)
    prediction = {"read_frame_indices": list(range(4)), "read_times": [[0., .5], [1., 1.5]],
                  "real_feature_rgb_hashes": [hashlib.sha256(pair.tobytes()).hexdigest()
                                              for pair in frames.reshape(2, 2, 32, 32, 3)]}
    monkeypatch.setattr(snapshot_readout, "fast_frame", lambda *a: np.zeros((32, 32, 3), np.uint8))
    actual, fast = verified_pixels(path, prediction, (32, 32))
    assert not fast
    np.testing.assert_array_equal(actual.reshape(-1, 32, 32, 3), frames)
