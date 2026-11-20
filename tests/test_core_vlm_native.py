from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from trace_vl.core_vlm.native import NativeLayout, NativeVideoQALoss, encode_native, prepare_native
from trace_vl.v3.backbone import Backbone


def tiny_example():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                         linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
                         linear_num_key_heads=2, linear_num_value_heads=2,
                         layer_types=["linear_attention", "full_attention"]),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=32, num_position_embeddings=64),
        image_token_id=4, video_token_id=3, vision_start_token_id=1, vision_end_token_id=2)
    torch.manual_seed(3)
    bb = Backbone.__new__(Backbone)
    bb.model = Qwen3_5ForConditionalGeneration(config).eval()
    bb.device, bb.dtype, bb.hidden = torch.device("cpu"), torch.float32, 32
    bb.video_token = 3
    bb.ids = lambda text: [9 + ord(c) % 50 for c in text]
    ids = torch.tensor([[5, 1, 3, 3, 3, 3, 2, 8, 1, 3, 3, 3, 3, 2, 6, 7]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  mm_token_type_ids=torch.where(ids == 3, 2, 0),
                  video_grid_thw=torch.tensor([[2, 4, 4]]),
                  pixel_values_videos=torch.randn(32, 3 * 2 * 16 * 16))
    layout = NativeLayout(inputs, [[0., .5], [1., 1.5]], "Action?", {})
    row = {"question": "Action?", "answer_text": "run", "query_time": 1.5, "native_layout": layout}
    return bb, layout, row


def test_full_packing_logits_match_native_multimodal_forward():
    bb, layout, row = tiny_example()
    steps = encode_native(bb, layout)
    net = NativeVideoQALoss(bb)
    embeds, positions, _, _, _ = net.prefix(steps, row)
    native = bb.model(**layout.inputs, use_cache=False).logits
    packed = bb.logits(bb.lm(embeds, positions, cache=None, use_cache=False))
    torch.testing.assert_close(native, packed, atol=1e-6, rtol=1e-5)


def test_answer_ce_matches_native_teacher_forcing_and_trains_visual_features():
    bb, layout, row = tiny_example()
    steps = encode_native(bb, layout)
    for step in steps:
        step.tokens = step.tokens.detach().requires_grad_()
    net = NativeVideoQALoss(bb, answer_end_token_id=2)
    actual, _ = net(steps, row)
    answer = bb.ids("run") + [2]
    extended = dict(layout.inputs)
    extended["input_ids"] = torch.cat([extended["input_ids"], torch.tensor([answer[:-1]])], 1)
    extended["attention_mask"] = torch.ones_like(extended["input_ids"])
    extended["mm_token_type_ids"] = torch.cat([extended["mm_token_type_ids"], torch.zeros(1, 3, dtype=torch.long)], 1)
    native = bb.model(**extended, use_cache=False).logits[:, -len(answer):]
    expected = F.cross_entropy(native.reshape(-1, native.shape[-1]), torch.tensor(answer))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(s.tokens.grad.abs().sum() > 0 for s in steps)


def test_grid_preserves_all_native_text_positions_and_compresses_only_vision():
    bb, layout, row = tiny_example()
    steps = encode_native(bb, layout)
    full = NativeVideoQALoss(bb)
    grid = NativeVideoQALoss(bb, mode="grid", pool_grid=(1, 1))
    _, fp, _, _, _ = full.prefix(steps, row)
    ge, gp, _, _, meta = grid.prefix(steps, row)
    assert meta["visual_tokens_total"] == 2
    assert ge.shape[1] == 10
    # Centre of each 2x2 grid is the top-left integer centre used by grid_pool.
    kept = [0, 1, 2, 6, 7, 8, 9, 13, 14, 15]
    torch.testing.assert_close(gp, fp[:, kept])
    with pytest.raises(ValueError, match="question/observations"):
        full.prefix(steps, {**row, "question": "Wrong question"})


def test_native_processor_never_silently_guesses_fps_or_vfr_timestamps():
    images = np.zeros((2, 2, 32, 32, 3), np.uint8)
    import hashlib
    audit = {"source_fps": 0, "frame_indices": [0, 1, 2, 3],
             "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    row = {"query_time": 4.}
    with pytest.raises(ValueError, match="positive FPS"):
        prepare_native(SimpleNamespace(), images, np.array([[0., 1.], [2., 3.]]), row, audit)
    audit["source_fps"] = 2
    with pytest.raises(ValueError, match="differ from decoded PTS"):
        prepare_native(SimpleNamespace(), images, np.array([[0., 1.], [2., 3.]]), row, audit)


def test_explicit_native_timestamp_approximation_keeps_measured_frames():
    import hashlib

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            meta = kwargs["video_metadata"][0]
            assert meta.fps == 2 and meta.frames_indices == [0, 1, 2, 3]
            assert kwargs["do_sample_frames"] is False
            return {"input_ids": torch.tensor([[1, 3]]), "video_grid_thw": torch.tensor([[2, 2, 2]])}

    images = np.zeros((2, 2, 32, 32, 3), np.uint8)
    times = np.array([[0., .5], [1., 1.8]])
    audit = {"source_fps": 2., "frame_indices": [0, 1, 2, 3], "source_frame_count": 4,
             "decoded_rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    row = {"query_time": 1.8, "question": "Action?", "answer_text": "run"}
    layout = prepare_native(Processor(), images, times, row, audit, allow_native_timestamp_approximation=True)
    assert layout.frame_times == times.tolist()
    assert layout.audit["timestamp_approximation_over_50ms"] is True
    assert layout.audit["timestamp_max_error_seconds"] == pytest.approx(.3)
    assert layout.audit["timestamp_text_seconds"] == [.25, 1.25]
