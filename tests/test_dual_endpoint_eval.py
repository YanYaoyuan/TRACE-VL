import json

import numpy as np
import pytest
import torch
from test_endpoint_reader import bb as bb
from test_frozen_ingest_reader import frozen_copy

from trace_vl.complementary_memory.state_stream import Block, block_inputs
from trace_vl.core_vlm import dual_endpoint_eval as dual
from trace_vl.core_vlm.native import NativeLayout, encode_native
from trace_vl.core_vlm.prepare import digest
from trace_vl.core_vlm.qa import grid_pool, question_and_answer
from trace_vl.v3.positions import PositionCounter


def guarded_inputs(bb):
    prefix = frozen_copy(bb)
    bb.model.eval().requires_grad_(False)
    source = {'source_group': 'unit-source'}
    inputs = {'reference': {'sources': [source]}, 'dual_code': dual.code_identity()}
    inputs['id'] = digest(inputs)
    return prefix, source, inputs


@pytest.mark.parametrize('change', ['code', 'manifest', 'cohort', 'plain'])
def test_dual_evaluation_refuses_wrong_identity_cohort_or_plain_label(bb, change):
    prefix, source, inputs = guarded_inputs(bb)
    variant = 'frozen_ingest'
    if change == 'code':
        inputs['dual_code'] = {**inputs['dual_code'], 'dual_endpoint_eval': 'wrong'}
        inputs['id'] = digest({k: v for k, v in inputs.items() if k != 'id'})
    elif change == 'manifest':
        inputs['unrecorded_change'] = True
    elif change == 'cohort':
        source = {'source_group': 'outside-source'}
    else:
        variant = 'plain'
    with pytest.raises(ValueError):
        dual._validate(bb, prefix, inputs, source, variant)


@pytest.mark.parametrize('change', ['prefix_trainable', 'query_trainable', 'mode', 'signature', 'same', 'storage'])
def test_dual_evaluation_requires_frozen_compatible_independent_models(bb, change):
    prefix, source, inputs = guarded_inputs(bb)
    if change == 'prefix_trainable':
        next(prefix.model.parameters()).requires_grad_(True)
    elif change == 'query_trainable':
        next(bb.model.parameters()).requires_grad_(True)
    elif change == 'mode':
        prefix.model.train()
    elif change == 'signature':
        prefix.hidden *= 2
    elif change == 'same':
        prefix = bb
    else:
        next(prefix.model.parameters()).data = next(bb.model.parameters()).data
    with pytest.raises(ValueError):
        dual._validate(bb, prefix, inputs, source, 'frozen_ingest')


@pytest.mark.parametrize('adapt_query', [False, True])
def test_real_eight_step_dual_flow_routes_ingestion_and_queries_separately(bb, monkeypatch, tmp_path,
                                                                         adapt_query):
    n = 8
    query = {'id': 'query', 'target_id': 'target', 'question': 'Action?',
             'options': ['first', 'second'], 'answer': 1, 'target_time': 7.5,
             'query_time': 7.5, 'delay_seconds': 0, 'planned_encoder_calls': n}
    ids = torch.tensor([[5, *sum(([8, 1, *([3] * 16), 2] for _ in range(n)), []), 6, 7]])
    generator = torch.Generator().manual_seed(29)
    frame_times = np.arange(2 * n, dtype=float).reshape(n, 2) / 2
    layout = NativeLayout({'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                           'mm_token_type_ids': torch.where(ids == 3, 2, 0),
                           'video_grid_thw': torch.tensor([[n, 8, 8]]),
                           'pixel_values_videos': torch.randn(n * 64, 3 * 2 * 16 * 16, generator=generator)},
                          frame_times.tolist(), question_and_answer(query)[0], {})
    with torch.no_grad():
        raw = encode_native(bb, layout)
    bb.processor = object()
    prefix = frozen_copy(bb)
    if adapt_query:
        with torch.no_grad():
            for name, parameter in bb.model.named_parameters():
                if 'lora_B' in name:
                    parameter.add_(.1)
    bb.model.eval().requires_grad_(False)
    before = [p.detach().clone() for p in prefix.model.parameters()]
    metadata = {'streams': [{'avg_frame_rate': '2/1', 'height': 128, 'width': 128}],
                'frames': [{} for _ in range(2 * n)]}
    video, metadata_path = tmp_path / 'video.mp4', tmp_path / 'metadata.json'
    video.write_bytes(b'fixed-video-identity')
    metadata_path.write_text(json.dumps(metadata))
    source = {'source_group': 'unit-source', 'video': str(video), 'video_sha256': dual.sha256(video),
              'metadata': str(metadata_path), 'metadata_sha256': dual.sha256(metadata_path),
              'sampled_frame_indices': list(range(2 * n)), 'sampled_frame_times': frame_times.tolist(),
              'queries': [query]}
    inputs = {'reference': {'sources': [source], 'pixels': 128 * 128}, 'dual_code': dual.code_identity()}
    inputs['id'] = digest(inputs)
    frames = [np.full((2, 128, 128, 3), i, dtype=np.uint8) for i in range(n)]
    grid = torch.tensor([[1, 8, 8]])
    monkeypatch.setattr(bb, 'preprocess', lambda frame: (int(frame[0, 0, 0, 0]), grid))
    monkeypatch.setattr(bb, 'encode', lambda index, shape: raw[index].tokens)
    monkeypatch.setattr(dual.longstream, 'schedule', lambda meta, end: (np.arange(2 * n), frame_times, None))
    monkeypatch.setattr(dual.longstream, 'stream_pixels', lambda *args: iter_frames(frames))
    monkeypatch.setattr(dual.native, 'native_prompt', lambda *args: ([5], [6, 7]))
    monkeypatch.setattr(dual.native, 'stamp_ids', lambda *args: [8])
    seen_gold = []

    def prepare(processor, images, times, record, audit, **kwargs):
        seen_gold.append(record['answer'])
        return layout

    monkeypatch.setattr(dual.frozen_prefix_parity, 'prepare_native', prepare)
    counter = PositionCounter()
    counter.text(1)
    for i, step in enumerate(raw):
        tokens, rows, cols = grid_pool(step.tokens, step.grid, (4, 4))
        block_inputs(prefix, Block(i, step.end, (8,), tokens, rows, cols, step.grid), counter)
    old = {'snapshot_steps': list(range(n)),
           'streams': {'window_stream': {'next_position': counter.pos}}}
    monkeypatch.setattr(dual.endpoint_reader_eval, 'frozen_rows', lambda *args: (old, {}))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    calls = {'ingestion': [], 'replay': [], 'score': [], 'text': []}
    native_stream, replay, score = dual.NativeStream, dual.replay, dual.score

    def create_stream(model, *args, **kwargs):
        calls['ingestion'].append(model)
        return native_stream(model, *args, **kwargs)

    def replay_stream(model, *args, **kwargs):
        calls['replay'].append(model)
        return replay(model, *args, **kwargs)

    def answer(model, *args, **kwargs):
        calls['score'].append(model)
        return score(model, *args, **kwargs)

    def text_answer(model, *args):
        calls['text'].append(model)
        return {'choice_fp32': 0, 'option_logits_fp32': [1., -1.]}

    monkeypatch.setattr(dual, 'NativeStream', create_stream)
    monkeypatch.setattr(dual, 'replay', replay_stream)
    monkeypatch.setattr(dual, 'score', answer)
    monkeypatch.setattr(dual.longstream, 'text_answer', text_answer)
    out = tmp_path / 'evaluation'
    dual.evaluate_source(bb, inputs, source, out, 'frozen_ingest', prefix_bb=prefix)
    assert calls['ingestion'] == [prefix, prefix]
    assert calls['replay'] == [prefix]
    assert len(calls['score']) == 6 and all(model is bb for model in calls['score'])
    assert calls['text'] == [bb]
    row = json.loads((out / 'questions' / f"{digest(query['id'])}.json").read_text())
    receipt = json.loads((out / 'sources' / f"{digest(source['source_group'])}.json").read_text())
    assert row['completed'] and row['inputs'] == inputs['id'] and row['steps_seen'] == n
    assert set(row['predictions']) == set(dual.POLICIES)
    assert row['predictions']['reservoir17'] == row['predictions']['window40']
    assert row['predictions']['reservoir17'] == row['predictions']['recent_replay8']
    assert all(s['kv_steps'] == n for s in row['streams'].values())
    assert receipt['completed'] and receipt['parity']['mode'] == 'staged_native_prefix_query'
    assert receipt['parity']['exact_packing'] and receipt['parity']['max_letter_logit_gap'] <= 3e-5
    assert seen_gold == [0]
    assert all(torch.equal(a, b) for a, b in zip(before, prefix.model.parameters(), strict=True))
    assert all(p.grad is None for model in (bb.model, prefix.model) for p in model.parameters())


def iter_frames(frames):
    yield from frames
