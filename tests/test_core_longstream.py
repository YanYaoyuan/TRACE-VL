import hashlib
import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest
import torch

from trace_vl.core_vlm.longstream import (
    ARMS,
    BoundedBank,
    StoredStep,
    answer,
    choose_sources,
    expand_pooled,
    schedule,
    stream_pixels,
)
from trace_vl.core_vlm.native import NativeVideoQALoss
from trace_vl.core_vlm.qa import grid_pool


def step(i, shape=(7, 9)):
    return StoredStep(torch.full((16, 8), float(i), dtype=torch.bfloat16), shape,
                      (float(i), i + .5), (2 * i, 2 * i + 1), hashlib.sha256(str(i).encode()).hexdigest())


def test_bounded_storage_is_independent_of_stream_length_and_questions():
    queried, untouched = BoundedBank(8, 17), BoundedBank(8, 17)
    for i in range(1000):
        s = step(i)
        queried.append(s)
        untouched.append(s)
        before = queried.audit()
        for arm in ARMS:
            selected = queried.read(arm)
            assert len(selected) <= 8
            assert all(a.end < b.end for a, b in zip(selected, selected[1:]))
        queried.read('stored_oracle', (0, 1))
        assert queried.audit() == before
        assert queried.audit() == untouched.audit()
        assert [s.indices for s in queried.reservoir] == [s.indices for s in untouched.reservoir]
    assert len(queried.reservoir) == len(queried.fifo) == 8
    assert len(queried.recent) == 8
    assert queried.old_seen == 996
    limit = queried.audit()['per_policy_feature_and_fixed_metadata_bytes']
    for i in range(1000, 1100):
        queried.append(step(i))
    assert queried.audit()['per_policy_feature_and_fixed_metadata_bytes'] == limit
    assert min(s.times[0] for s in queried.fifo) >= 1088
    assert min(s.times[0] for s in queried.reservoir) < 1000
    with pytest.raises(ValueError, match='increasing'):
        queried.append(step(1099))


@pytest.mark.parametrize('shape', [(4, 4), (6, 10), (7, 9)])
def test_compact_feature_expansion_preserves_native_pool_values_exactly(shape):
    torch.manual_seed(3)
    stored = replace(step(0, shape), tokens=torch.randn(16, 8).to(torch.bfloat16))
    expanded = expand_pooled(stored)
    assert torch.equal(grid_pool(expanded.tokens, shape, (4, 4))[0], stored.tokens)
    assert expanded.frame_times == stored.times


def test_sampler_uses_fixed_causal_clock_and_preserves_decoder_indices():
    metadata = {'frames': [{'best_effort_timestamp_time': str(x)} for x in [5, 5.5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]]}
    indices, times, dropped = schedule(metadata, 4.3)
    assert dropped == 1 and 2 not in indices
    assert len(indices) % 2 == 0 and times.max() <= 4.3
    longer, _, _ = schedule(metadata, 5)
    assert longer[:len(indices)].tolist() == indices.tolist()
    with pytest.raises(ValueError, match='outside'):
        schedule(metadata, 6)


def test_delay_cohort_keeps_questions_labels_sources_and_three_arrivals_fixed():
    activity = [{'id': str(i), 'source_group': str(i), 'video': str(i), 'query_time': 20.,
                 'media_duration': 100., 'evidence': [2, 8], 'answer': i % 2, 'question': 'before anchor?'}
                for i in range(10)]
    ovo = [{'id': 'o' + str(i), 'source_group': 'o' + str(i), 'video': str(i), 'query_time': 310.,
            'task': 'EPM', 'answer': 0, 'question': 'past event?'} for i in range(10)]
    before = json.dumps([activity, ovo])
    selected = choose_sources(activity, ovo, 3)
    assert selected == choose_sources(activity[::-1], ovo[::-1], 3)
    assert len(selected) == len({s['source_group'] for s in selected}) == 6
    assert sum(len(s['queries']) for s in selected) == 12
    for s in selected[:3]:
        assert [q['query_time'] for q in s['queries']] == [20, 35, 80]
        assert len({(q['question'], q['answer']) for q in s['queries']}) == 1
    assert json.dumps([activity, ovo]) == before
    with pytest.raises(ValueError, match='insufficient'):
        choose_sources([{**r, 'media_duration': 50} for r in activity], ovo, 3)


def test_real_ffmpeg_decoder_yields_each_selected_pixel_once(tmp_path):
    frames = np.stack([np.full((32, 32, 3), i * 20, dtype=np.uint8) for i in range(8)])
    video = tmp_path / 'real.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '32x32',
                    '-r', '2', '-i', 'pipe:0', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(video)],
                   input=frames.tobytes(), check=True, capture_output=True)
    result = list(stream_pixels(video, [0, 2, 4, 6], 32, 32))
    assert len(result) == 2
    np.testing.assert_array_equal(np.concatenate(result), frames[[0, 2, 4, 6]])


def test_real_tiny_native_decoder_reads_stored_features_without_vision_or_gold(monkeypatch):
    from test_core_vlm_native import tiny_example

    bb, _, _ = tiny_example()

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]['content'][1]['text']

        def __call__(self, *, text, videos, **kwargs):
            assert not videos[0].any()  # layout placeholders only
            count = len(videos[0]) // 2
            token_ids = [5]
            for _ in range(count):
                token_ids += [1] + [3] * 16 + [2, 8]
            token_ids += [6, 7]
            ids = torch.tensor([token_ids])
            return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                    'mm_token_type_ids': torch.where(ids == 3, 2, 0),
                    'video_grid_thw': torch.tensor([[count, 8, 8]])}

    bb.processor = Processor()
    def forbidden(*args, **kwargs):
        raise AssertionError('vision encoder called at query time')
    monkeypatch.setattr(bb, 'encode', forbidden)
    monkeypatch.setattr(bb.visual, 'forward', forbidden)
    selected = [replace(step(i, (4, 4)), tokens=torch.randn(16, 32)) for i in range(2)]
    qa = NativeVideoQALoss(bb, mode='grid', pool_grid=(4, 4))
    query = {'question': 'Which action?', 'options': ['first', 'second'], 'answer': 1, 'query_time': 4.}
    metadata = {'streams': [{'avg_frame_rate': '2/1'}], 'frames': [{}] * 8}
    a = answer(bb, qa, selected, query, metadata, (128, 128), 'video')
    b = answer(bb, qa, selected, {**query, 'answer': 0}, metadata, (128, 128), 'video')
    assert a == b and a['query_time_encoder_calls'] == 0 and a['visual_tokens'] == 32
    with pytest.raises(ValueError, match='causal'):
        answer(bb, qa, selected, {**query, 'query_time': .5}, metadata, (128, 128), 'video')
