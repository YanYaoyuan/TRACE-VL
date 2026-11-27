"""Source-atomic, truncated LoRA adaptation on complete native endpoint streams.

This engineering pilot has no held-out scoring or learned retention policy.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trace_vl.complementary_memory.state_stream import Block, block_inputs, segment_layout
from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.positions import PositionCounter
from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .control import checkpoint_identity
from .data import presentation_frames, probe_video, sha256
from .endpoint_reader import EndpointReaderEpisode
from .longstream import stream_pixels
from .native import NativeVideoQALoss, prepare_native
from .qa import VisualStep, grid_pool
from .train import checked_checkpoint, latest_checkpoint, restore_checkpoint, save_checkpoint, tensor_digest


def training_sources(rows):
    """Validate source-atomic training groups before constructing any model."""
    groups, ids, owners = {}, set(), {}
    for row in rows:
        if (row.get('split') != 'candidate_train_pilot'
                or row.get('endpoint_rule') != 'actual_full_video_container_end'
                or row.get('evidence_intervals') is not None
                or row.get('memory_necessity_verified') is not False):
            raise ValueError('only declared weak native endpoint training QA is accepted')
        if (not isinstance(row.get('question'), str) or not row['question'].strip()
                or not isinstance(row.get('answer'), str) or not row['answer'].strip()
                or not math.isfinite(row['query_time']) or row['query_time'] <= 0):
            raise ValueError('nonempty native QA and finite positive endpoint required')
        key, group = row['source_id'], row['source_group_id']
        if not key or not group or row['id'] in ids:
            raise ValueError('unique QA IDs and nonempty source/group IDs required')
        ids.add(row['id'])
        if group in owners and owners[group] != key:
            raise ValueError('pilot requires one source per metadata group')
        owners[group] = key
        siblings = groups.setdefault(key, [])
        if siblings and any(row[k] != siblings[0][k] for k in
                            ('video', 'video_sha256', 'query_time', 'source_group_id')):
            raise ValueError('source metadata differs between questions')
        siblings.append(row)
    if len(groups) < 2:
        raise ValueError('at least two training sources required')
    # The first two cover cost extremes, selected solely by duration and ID.
    ordered = sorted(groups, key=lambda k: (groups[k][0]['query_time'], k))
    first, last = ordered[0], ordered[-1]
    remainder = sorted(set(ordered) - {first, last}, key=lambda k: hashlib.sha256(
        ('TRACEVL_ENDPOINT_READER_ORDER_V1_20261127:' + k).encode()).hexdigest())
    return [sorted(groups[k], key=lambda r: r['id']) for k in [first, last, *remainder]]


def causal_indices(pts, endpoint):
    """Floor actual PTS at 2 fps; retain complete, distinct two-frame steps."""
    pts = np.asarray(pts, dtype=float)
    if (pts.ndim != 1 or len(pts) < 2 or not np.isfinite(pts).all()
            or np.any(np.diff(pts) <= 0) or abs(pts[0]) > .001
            or not math.isfinite(endpoint) or endpoint <= 0):
        raise ValueError('strict finite zero-origin frame PTS and positive endpoint required')
    valid = np.flatnonzero(pts <= endpoint)
    if len(valid) < 2:
        raise ValueError('fewer than two causal frames')
    slots = np.arange(0., min(endpoint, float(pts[valid[-1]])) + 1e-8, .5)
    selected = valid[np.maximum(0, np.searchsorted(pts[valid], slots, side='right') - 1)]
    selected = np.unique(selected)
    selected = selected[:len(selected) // 2 * 2]
    if len(selected) < 16:
        raise ValueError('at least eight complete native steps required')
    return selected


def native_prompt(bb, record, frames, indices, times, fps, count, first_stamp, answer='unused'):
    rgb = np.asarray(frames, dtype=np.uint8)
    audit = {'source_fps': fps, 'source_frame_count': count,
             'frame_indices': list(map(int, indices)),
             'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
    row = {**record, 'answer_text': answer}
    layout = prepare_native(bb.processor, rgb, np.asarray(times), row, audit,
                            allow_native_timestamp_approximation=True)
    head, tail = segment_layout(layout.inputs['input_ids'][0].tolist(), bb.video_token,
                                bb.vision_start, bb.vision_end, first_stamp)
    return head, tail, layout


def source_observations(bb, records, *, parity):
    row = records[0]
    metadata = probe_video(row['video'], 180)
    pts, keep = presentation_frames(metadata['frames'], drop_bad_pts=False)
    duration = float(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
         'default=noprint_wrappers=1:nokey=1', row['video']], text=True, timeout=30))
    if not keep.all() or abs(duration - row['query_time']) > .001:
        raise ValueError('actual source duration/PTS differs; never alter the query endpoint')
    indices = causal_indices(pts, row['query_time'])
    times = pts[indices].reshape(-1, 2)
    fps = float(Fraction(metadata['streams'][0]['avg_frame_rate']))
    hw = frame_size(metadata['streams'][0]['height'], metadata['streams'][0]['width'], 256 * 256, 32)
    blocks, raw, rgb = [], [], []
    started = time.monotonic()
    decoder = stream_pixels(row['video'], indices, *hw)
    try:
        for i, frames in enumerate(decoder):
            with torch.no_grad():
                pixels, grid = bb.preprocess(frames)
                dense = bb.encode(pixels, grid)
                spatial = bb.merged_grid(grid)
                tokens, rows, cols = grid_pool(dense, spatial, (4, 4))
                stamp = tuple(bb.ids(f'<{float(np.mean(indices[2*i:2*i+2] / fps)):.1f} seconds>'))
                blocks.append(Block(i, float(times[i, -1]), stamp,
                                    tokens.detach(), rows, cols, spatial))
                if parity and i < 8:
                    raw.append(VisualStep(dense.detach(), spatial, float(times[i, -1]),
                                          tuple(map(float, times[i]))))
                    rgb.append(frames)
    finally:
        decoder.close()
    if len(blocks) != len(times) or len(blocks) < 40 or len(blocks) > 660:
        raise ValueError('complete pilot source must have 40..660 paired steps')
    torch.cuda.synchronize()
    vision_seconds = time.monotonic() - started
    dummy = np.zeros((1, 2, *hw, 3), np.uint8)
    prompts = [native_prompt(bb, r, dummy, indices[:2], times[:1], fps,
                             len(pts), blocks[0].stamp) for r in records]
    head = prompts[0][0]
    if any(p[0] != head for p in prompts):
        raise ValueError('question changed the live source head')
    alternate = native_prompt(bb, row, dummy, indices[:2], times[:1], fps,
                              len(pts), blocks[0].stamp, answer='different unused gold')
    if alternate[:2] != prompts[0][:2]:
        raise ValueError('gold changed native question input')
    eos = bb.tokenizer.eos_token_id
    if type(eos) is not int:
        raise ValueError('integer EOS required')
    answers = [bb.ids(r['answer']) + [eos] for r in records]
    if any(not 0 < len(a) <= 512 for a in answers):
        raise ValueError('answer exceeds declared budget; no silent truncation')
    parity_gap = None
    if parity:
        _, tail, layout = native_prompt(bb, row, np.stack(rgb), indices[:16], times[:8], fps,
                                        len(pts), blocks[0].stamp, answer=row['answer'])
        native = NativeVideoQALoss(bb, mode='grid', pool_grid=(4, 4), answer_end_token_id=eos)
        native_row = {**row, 'answer_text': row['answer'], 'native_layout': layout}
        with torch.no_grad():
            embed, pos, *_ = native.prefix(raw, native_row)
            counter = PositionCounter()
            parts = [(bb.embed_ids(head), counter.text(len(head), bb.device))]
            parts += [block_inputs(bb, block, counter) for block in blocks[:8]]
            parts.append((bb.embed_ids(tail), counter.text(len(tail), bb.device)))
            if (not torch.equal(torch.cat([p[0] for p in parts], 1), embed)
                    or not torch.equal(torch.cat([p[1] for p in parts], 1), pos)):
                raise ValueError('native embeddings/positions differ')
            logits, targets, _ = native.answer_logits(raw, native_row)
            expected = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets))
        short = EndpointReaderEpisode(bb, head, 'full')
        for block in blocks[:8]:
            short.append(block)
        short.seal(row['query_time'])
        loss, _ = short.answer_loss(tail, answers[0])
        parity_gap = abs(float(loss.detach()) - expected)
        if parity_gap > .05:
            raise ValueError('short cached/uncached CE exceeds predeclared 0.05 tolerance')
    audit = {'source_id': row['source_id'], 'video_sha256': row['video_sha256'],
             'query_time': row['query_time'], 'frame_indices': indices.tolist(),
             'frame_times': times.tolist(), 'prefix_steps': len(blocks),
             'short_native_CE_gap': parity_gap, 'decode_vision_seconds': vision_seconds}
    return head, blocks, [(p[1], a) for p, a in zip(prompts, answers, strict=True)], audit


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.training.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(args, out)


def _run(args, out):
    import transformers

    manifest = Path(args.manifest)
    if sha256(manifest) != args.manifest_sha256:
        raise ValueError('private training manifest hash differs')
    sources = training_sources([json.loads(line) for line in manifest.read_text().splitlines()])
    if len(sources) != 64 or sum(map(len, sources)) != 317:
        raise ValueError('this pilot requires the frozen 64-source/317-QA cohort')
    for records in sources:
        row = records[0]
        video = Path(row['video']).resolve()
        if (not video.is_relative_to(manifest.parent.resolve() / 'videos')
                or sha256(video) != row['video_sha256']):
            raise ValueError('training media path/hash differs')
    base = json.loads(Path(args.base_inputs).read_text())
    if checkpoint_identity(base['model']) != base['model_identity']:
        raise ValueError('local base model differs')
    if checked_checkpoint(Path(base['plain']), base['checkpoint']['run']) != base['checkpoint']:
        raise ValueError('initial Plain adapter differs')
    code = {name: sha256(Path(__file__).with_name(name)) for name in
            ('train_endpoint_reader.py', 'endpoint_reader.py', 'native.py', 'data.py', 'train.py')}
    stream_path = Path(__file__).parents[1] / 'complementary_memory/state_stream.py'
    code['state_stream.py'] = sha256(stream_path)
    identity = {'protocol': 'native-endpoint-source-reader-pilot-v1', 'arm': args.arm,
                'manifest_sha256': args.manifest_sha256, 'initialization': base,
                'source_order': [r[0]['source_id'] for r in sources],
                'QA_order': [[r['id'] for r in records] for records in sources],
                'seed': 17, 'lr': 1e-5, 'clip': 1., 'weight_decay': 0., 'updates': 64,
                'retained_steps': 40, 'reservoir_recent': 8, 'reservoir_history': 32,
                'pixels': 256 * 256, 'pool_grid': [4, 4], 'fps': 2,
                'timestamp_policy': 'native-index-fps', 'prefix_gradients': False,
                'torch': str(torch.__version__), 'transformers': transformers.__version__, 'code': code}
    identity['id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ensure_run_manifest(out / 'run.json', identity)
    previous = latest_checkpoint(out, identity['id'])
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    bb = Backbone(BackboneConfig(name=base['model'], dtype='bfloat16', local_files_only=True),
                  StreamConfig(pixels_per_frame=256 * 256), device='cuda:0')
    bb.add_lora(8, 16, dropout=0., adapter=str(Path(base['plain']) / 'adapter'))
    bb.model.eval()
    params = {n: p for n, p in bb.model.named_parameters() if p.requires_grad}
    if not params or any('lora_' not in n or 'language_model' not in n for n in params):
        raise ValueError('only LM LoRA may train')
    base_versions = [(id(p), p._version) for p in bb.model.parameters() if not p.requires_grad]
    initial_digest = tensor_digest(params)
    atomic_text(out / 'initialization.json', json.dumps({'adapter_sha256': initial_digest}) + '\n')
    optimizer = torch.optim.AdamW(list(params.values()), lr=1e-5, weight_decay=0.)
    start, metrics = (0, []) if previous is None else restore_checkpoint(
        previous, identity['id'], params, optimizer, 0, 1)
    if len(metrics) != start or [m['source_id'] for m in metrics] != identity['source_order'][:start]:
        raise ValueError('checkpoint source schedule differs')
    limit = args.stop_after or len(sources)
    if not start <= limit <= len(sources):
        raise ValueError('stop-after must preserve the declared full update schedule')
    torch.cuda.reset_peak_memory_stats()
    for update in range(start, limit):
        records = sources[update]
        tick = time.monotonic()
        atomic_text(out / 'progress.json', json.dumps({'completed_sources': update, 'total_sources': 64,
                    'stage': 'source_decode_vision', 'source_id': records[0]['source_id']}) + '\n')
        head, blocks, questions, audit = source_observations(bb, records, parity=update == start)
        policy = 'reservoir' if args.arm == 'reservoir' else 'window'
        episode = EndpointReaderEpisode(bb, head, policy, recent=8 if policy == 'reservoir' else 40,
                                        capacity=32, seed=17)
        prefix_tick = time.monotonic()
        for block in blocks:
            episode.append(block)
        episode.seal(records[0]['query_time'])
        torch.cuda.synchronize()
        prefix_seconds = time.monotonic() - prefix_tick
        stream = episode.stream.audit()
        if stream['kv_steps'] != 40 or stream['seen'] != len(blocks):
            raise ValueError('retention/stream accounting differs')
        update_tick = time.monotonic()
        result = episode.train_step(questions, optimizer)
        torch.cuda.synchronize()
        if (base_versions != [(id(p), p._version) for p in bb.model.parameters() if not p.requires_grad]
                or any(p.grad is not None for p in bb.model.parameters() if not p.requires_grad)):
            raise RuntimeError('frozen base changed or received gradients')
        audit.update(update=update + 1, **result, stream=stream, prefix_seconds=prefix_seconds,
                     update_seconds=time.monotonic() - update_tick,
                     source_seconds_before_checkpoint=time.monotonic() - tick,
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        metrics.append(audit)
        # Two arms share this parent; leave room for another full checkpoint.
        existing = sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())
        if existing > 63 * 2**30:
            raise RuntimeError('pilot output budget nearly exhausted; preserve checkpoints and stop')
        save_checkpoint(out, update + 1, identity['id'], bb, params, optimizer, 0, 1, metrics)
        receipt = checked_checkpoint(out / f'checkpoint-{update+1:06d}', identity['id'])
        if receipt['update'] != update + 1:
            raise RuntimeError('source checkpoint incomplete')
        atomic_text(out / 'progress.json', json.dumps({'stage': 'checkpointed', 'completed_sources': update + 1,
                    'total_sources': 64, 'last_source': audit, 'checkpoint': f'checkpoint-{update+1:06d}'}) + '\n')
        print(json.dumps({'source_id': audit['source_id'], 'update': update + 1,
                          'steps': len(blocks), 'source_seconds': time.monotonic() - tick}), flush=True)
        del episode, blocks, questions
    atomic_text(out / 'report.json', json.dumps({'completed_sources': limit, 'total_sources': 64,
                'completed': limit == 64, 'engineering_training_only': True,
                'run': identity['id'], 'arm': args.arm, 'initial_adapter_sha256': initial_digest,
                'last_checkpoint': f'checkpoint-{limit:06d}', 'held_out_evaluation': False}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--base-inputs', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', choices=('reservoir', 'window40'), required=True)
    parser.add_argument('--stop-after', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
