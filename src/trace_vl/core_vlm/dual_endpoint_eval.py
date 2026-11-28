"""Fixed-ingestion / adapted-query development evaluator, separate from old runs.

The caller verifies both loaded checkpoint receipts and the shared frozen base /
vision identity before constructing backbones. This module binds its code hashes
to the input manifest and validates the live cache/role contract on each source.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from fractions import Fraction

import torch

from trace_vl.complementary_memory import native_state as native
from trace_vl.complementary_memory.state_stream import Block, NativeStream, replay, score
from trace_vl.v3.backbone import frame_size

from . import endpoint_reader, endpoint_reader_eval, frozen_prefix_parity, longstream
from .data import sha256
from .endpoint_reader import _cache_signature
from .longstream_delayed import CausalEvents, events_for
from .prepare import digest, write_json
from .qa import VisualStep, grid_pool

POLICIES = endpoint_reader_eval.POLICIES


def code_identity():
    """Caller stores this as inputs['dual_code'] before computing inputs['id']."""
    return {"dual_endpoint_eval": sha256(__file__),
            "frozen_prefix_parity": sha256(frozen_prefix_parity.__file__),
            "endpoint_reader": sha256(endpoint_reader.__file__),
            "reference_evaluator": sha256(endpoint_reader_eval.__file__),
            **native.code_identity()}


def _validate(bb, prefix_bb, inputs, source, variant):
    if (inputs.get('dual_code') != code_identity()
            or inputs['id'] != digest({k: v for k, v in inputs.items() if k != 'id'})):
        raise ValueError('dual evaluation code/input identity changed')
    if source not in inputs['reference']['sources']:
        raise ValueError('source is outside the frozen input cohort')
    if variant == 'plain':
        raise ValueError('dual evaluator cannot claim original Plain reproduction')
    if prefix_bb is bb or _cache_signature(bb) != _cache_signature(prefix_bb):
        raise ValueError('independent compatible prefix/query backbones required')
    if any(p.requires_grad for model in (bb.model, prefix_bb.model) for p in model.parameters()):
        raise ValueError('evaluation query and prefix backbones must be fully frozen')
    if any(m.training for model in (bb.model, prefix_bb.model) for m in model.modules()):
        raise ValueError('dual evaluation requires eval mode')
    storage = {p.untyped_storage().data_ptr() for p in prefix_bb.model.parameters() if p.numel()}
    if any(p.numel() and p.untyped_storage().data_ptr() in storage for p in bb.model.parameters()):
        raise ValueError('prefix and query parameter storage must be independent')


@torch.no_grad()
def evaluate_source(bb, inputs, source, out, variant, *, prefix_bb):
    """All visual prefix/replay paths ingest with prefix_bb; answers use bb.

    Checkpoint/base identity is a loader responsibility, not inferred from
    architectural compatibility. No old cache is transplanted between weights.
    """
    _validate(bb, prefix_bb, inputs, source, variant)
    started = time.monotonic()
    models = (bb.model, prefix_bb.model)
    versions = tuple((id(p), p._version, p.requires_grad) for m in models for p in m.parameters())

    def unchanged():
        if versions != tuple((id(p), p._version, p.requires_grad) for m in models for p in m.parameters()):
            raise RuntimeError('evaluation model parameters changed within the source')
        if any(part.training for model in models for part in model.modules()):
            raise RuntimeError('evaluation model mode changed within the source')

    if sha256(source['video']) != source['video_sha256'] or sha256(source['metadata']) != source['metadata_sha256']:
        raise ValueError('source media changed')
    metadata = endpoint_reader_eval.read(source['metadata'])
    indices, times, _ = longstream.schedule(metadata, max(q['query_time'] for q in source['queries']))
    if indices.tolist() != source['sampled_frame_indices'] or times.tolist() != source['sampled_frame_times']:
        raise ValueError('frozen causal observation changed')
    fps = float(Fraction(metadata['streams'][0]['avg_frame_rate']))
    meta = metadata['streams'][0]
    hw = frame_size(meta['height'], meta['width'], inputs['reference']['pixels'], 32)
    queries = {q['id']: q for q in source['queries']}
    prompts = {k: native.native_prompt(bb, q, indices, times, metadata, hw) for k, q in queries.items()}
    if any(native.native_prompt(prefix_bb, q, indices, times, metadata, hw) != prompts[k]
           for k, q in queries.items()):
        raise ValueError('prefix and query native prompt tokens differ')
    head = next(iter(prompts.values()))[0]
    if any(h != head for h, _ in prompts.values()):
        raise ValueError('question changed live prompt head')
    live = {'reservoir17': NativeStream(prefix_bb, head, 'reservoir', recent=8, capacity=32,
                                       seed=native.memory_seed(source['source_group'], 17)),
            'window40': NativeStream(prefix_bb, head, 'window', recent=40)}
    recent, snapshots, completed = deque(maxlen=8), {}, []
    parity_parts, parity = ([], [], []), None
    torch.cuda.reset_peak_memory_stats()

    def ask(q):
        unchanged()
        if live['window40'].seen != q['planned_encoder_calls']:
            raise ValueError('query has wrong causal prefix')
        old, _ = endpoint_reader_eval.frozen_rows(inputs, q)
        if (snapshots[q['target_id']] != old['snapshot_steps']
                or live['window40'].counter.pos != old['streams']['window_stream']['next_position']):
            raise ValueError('snapshot/position differs from frozen reference')
        tail, options = prompts[q['id']][1], len(q['options'])
        before = {k: (s.snapshot(), s.rng.getstate()) for k, s in live.items()}
        predictions = {k: score(bb, s.fork(), s.counter.pos, tail, options) for k, s in live.items()}
        fresh = replay(prefix_bb, head, list(recent))
        fresh_before, fresh_rng = fresh.snapshot(), fresh.rng.getstate()
        predictions['recent_replay8'] = score(bb, fresh.fork(), fresh.counter.pos, tail, options)
        fresh.verify(fresh_before)
        if fresh.rng.getstate() != fresh_rng:
            raise ValueError('question changed recent replay RNG')
        text = longstream.text_answer(bb, q, source['video_sha256'])
        predictions['text_only'] = {'choice': text['choice_fp32'], 'option_logits_fp32': text['option_logits_fp32']}
        for key, stream in live.items():
            stream.verify(before[key][0])
            if stream.rng.getstate() != before[key][1]:
                raise ValueError('question changed retention RNG')
        if not completed:
            for key, stream in {**live, 'recent_replay8': fresh}.items():
                if score(bb, stream.fork(), stream.counter.pos, tail, options) != predictions[key]:
                    raise ValueError('repeat question is not deterministic')
            for key, stream in live.items():
                stream.verify(before[key][0])
                if stream.rng.getstate() != before[key][1]:
                    raise ValueError('repeat question changed retention RNG')
            fresh.verify(fresh_before)
            if fresh.rng.getstate() != fresh_rng:
                raise ValueError('repeat question changed recent replay RNG')
        for prediction in predictions.values():
            values = prediction['option_logits_fp32']
            if len(values) != options or not all(math.isfinite(v) for v in values):
                raise ValueError('nonfinite/wrong answer options')
        unchanged()
        write_json(out / 'questions' / f"{digest(q['id'])}.json", {
            'completed': True, 'inputs': inputs['id'], 'variant': variant, 'id': q['id'],
            'source_group': source['source_group'], 'video_sha256': source['video_sha256'],
            'gold': q['answer'], 'delay_seconds': q['delay_seconds'], 'query_time': q['query_time'],
            'target_time': q['target_time'], 'steps_seen': live['window40'].seen,
            'predictions': predictions, 'streams': {k: s.audit() for k, s in live.items()},
            'reference_max_gap': None, 'prefix_mode': 'frozen_ingestion_adapted_query'})
        completed.append(q['id'])

    def dispatch(event):
        when, kind, identity = event
        if kind == 1:
            return ask(queries[identity])
        if not recent or any(block.end > when for block in recent):
            raise ValueError('noncausal anchor snapshot')
        snapshots[identity] = [block.index for block in recent]

    events = CausalEvents(events_for(source['queries']), dispatch)
    pixels = longstream.stream_pixels(source['video'], indices, *hw)
    try:
        for i, frames in enumerate(pixels):
            end = float(times[i][-1])
            events.due(end, inclusive=False)
            # Both loaders must verify the same frozen base/vision identity.
            values, grid = bb.preprocess(frames)
            dense = bb.encode(values, grid)
            spatial = bb.merged_grid(grid)
            tokens, rows, cols = grid_pool(dense, spatial, (4, 4))
            block = Block(i, end, tuple(native.stamp_ids(prefix_bb, indices[2*i:2*i+2], fps)),
                          tokens.detach(), rows, cols, spatial)
            recent.append(block)
            for stream in live.values():
                stream.append(block)
                if stream.audit()['kv_steps'] > 40:
                    raise ValueError('retained step budget exceeded')
            if parity is None:
                for store, value in zip(parity_parts, (block, VisualStep(dense, spatial, end,
                                               tuple(map(float, times[i]))), frames), strict=True):
                    store.append(value)
                if len(parity_parts[0]) == native.PARITY_STEPS:
                    q = min(queries.values(), key=lambda row: row['query_time'])
                    parity = frozen_prefix_parity.staged_prefix_parity(
                        bb, prefix_bb, head, prompts[q['id']][1], *parity_parts, q, indices, times, metadata)
                    parity_parts = ([], [], [])
            events.due(end, inclusive=True)
            if i % 60 == 0:
                print(json.dumps({'variant': variant, 'source': source['source_group'],
                                  'step': i + 1, 'total_steps': len(times)}), flush=True)
        events.due(float('inf'), inclusive=True)
    finally:
        pixels.close()
    unchanged()
    if len(completed) != len(queries) or events.cursor != len(events.events) or parity is None:
        raise ValueError('incomplete source/questions/staged native parity')
    torch.cuda.synchronize()
    write_json(out / 'sources' / f"{digest(source['source_group'])}.json", {
        'completed': True, 'inputs': inputs['id'], 'source_group': source['source_group'],
        'steps': len(times), 'questions': len(completed), 'seconds': time.monotonic() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'parity': parity, 'reference_max_gap': None, 'prefix_mode': 'frozen_ingestion_adapted_query'})
