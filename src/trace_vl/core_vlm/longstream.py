"""Frozen-checkpoint, one-pass streaming-memory diagnostic on existing development media.

This is a baseline/capability experiment, not a new trained memory architecture.
Caption-derived labels have not been visually certified. Privileged evidence
oracles are reported separately from deployable, query-blind memory policies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from trace_vl.streaming_vision.media import VideoProbeCache
from trace_vl.v3.backbone import Backbone, frame_size
from trace_vl.v3.config import BackboneConfig, StreamConfig
from trace_vl.v3.run_io import ensure_run_manifest

from .control import checkpoint_identity
from .data import presentation_frames, sha256
from .experiment import run_jobs
from .native import NativeVideoQALoss, check_native_forward, prepare_native
from .prepare import digest, write_json
from .qa import VisualStep, grid_pool, question_and_answer
from .train import checked_checkpoint

ARMS = ('recent4', 'recent8', 'fifo_uniform', 'reservoir_uniform')
BASELINES = ('text_only',) + ARMS
ORACLES = ('evidence_oracle', 'stored_oracle')
SCOPE = ('Development capability diagnostic, frozen Plain LoRA. Sequential 2-fps pixels, '
         'one vision encode per two-frame step, query-blind bounded FIFO/reservoir writes. '
         'At most 128 visual read tokens. No historical pixel replay at query time. '
         'Caption labels require visibility review; additional wait is not the exact evidence gap. '
         'Evidence and stored-routing oracles use privileged annotations and are not deployable methods.')


@dataclass(frozen=True)
class StoredStep:
    tokens: torch.Tensor
    grid: tuple[int, int]
    times: tuple[float, float]
    indices: tuple[int, int]
    rgb_sha256: str

    @property
    def end(self):
        return self.times[-1]


def spread(items, count=4):
    items = sorted(items, key=lambda s: s.end)
    if len(items) <= count:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, count).round().astype(int)]


class BoundedBank:
    """Writer accepts only observations; questions/answers cannot affect storage."""

    def __init__(self, capacity=32, seed=17, *, keep_prefix=False):
        if type(capacity) is not int or capacity < 4:
            raise ValueError('at least four historical slots required')
        self.capacity = capacity
        self.recent = deque(maxlen=8)
        self.fifo = deque(maxlen=capacity)
        self.reservoir = []
        self.prefix = []
        self.keep_prefix = keep_prefix
        self.old_seen = 0
        self.seen = 0
        self.rng = random.Random(seed)
        self.lineage = hashlib.sha256()

    def append(self, step):
        if self.recent and step.times[0] <= self.recent[-1].end:
            raise ValueError('strictly increasing, nonoverlapping stream steps required')
        if tuple(step.tokens.shape[:1]) != (16,) or step.tokens.ndim != 2:
            raise ValueError('stored observations require exactly 16 spatial tokens')
        # A step becomes historical after it leaves the common four-step window.
        if len(self.recent) >= 4:
            old = self.recent[-4]
            self.old_seen += 1
            self.fifo.append(old)
            if len(self.reservoir) < self.capacity:
                self.reservoir.append(old)
            else:
                index = self.rng.randrange(self.old_seen)
                if index < self.capacity:
                    self.reservoir[index] = old
        self.recent.append(step)
        if self.keep_prefix and len(self.prefix) < 4:
            self.prefix.append(step)
        self.seen += 1
        self.lineage.update(bytes.fromhex(step.rgb_sha256))
        self.lineage.update(np.asarray(step.indices, dtype=np.int64).tobytes())

    def read(self, arm, evidence=None):
        recent = list(self.recent)[-4:]
        if arm == 'recent4':
            return recent
        if arm == 'recent8':
            return list(self.recent)
        if arm == 'prefix4' and self.keep_prefix:
            return sorted({s.indices: s for s in self.prefix + recent}.values(), key=lambda s: s.end)
        if arm == 'fifo_uniform':
            old = spread(self.fifo)
        elif arm == 'reservoir_uniform':
            old = spread(self.reservoir)
        elif arm in ('stored_oracle', 'fifo_stored_oracle') and evidence is not None:
            retained = self.reservoir if arm == 'stored_oracle' else self.fifo
            matching = [s for s in retained if any(evidence[0] <= t <= evidence[1] for t in s.times)]
            old = spread(matching)
            selected_indices = {s.indices for s in old}
            other = [s for s in retained if s.indices not in selected_indices]
            if len(old) < 4:
                old += spread(other, 4 - len(old))
        else:
            raise ValueError('unknown memory arm')
        return sorted(old + recent, key=lambda s: s.end)

    def audit(self):
        # Per-policy feature payload and fixed-width source metadata (96 B/slot).
        per_step = self.recent[-1].tokens.numel() * self.recent[-1].tokens.element_size() if self.recent else 0
        union = {s.indices: s for s in (*self.recent, *self.fifo, *self.reservoir, *self.prefix)}
        return {'seen': self.seen, 'lineage': self.lineage.hexdigest(), 'old_seen': self.old_seen,
                'recent_steps': len(self.recent), 'fifo_steps': len(self.fifo),
                'reservoir_steps': len(self.reservoir), 'capacity': self.capacity,
                'prefix_control_enabled': self.keep_prefix, 'prefix_control_steps': len(self.prefix),
                'per_policy_retained_step_limit': self.capacity + 4,
                'joint_experiment_unique_retained_steps': len(union),
                'joint_experiment_feature_and_fixed_metadata_bytes': len(union) * (per_step + 96),
                'per_policy_feature_and_fixed_metadata_bytes': {
                    'fifo': (len(self.fifo) + min(4, len(self.recent))) * (per_step + 96),
                    'reservoir': (len(self.reservoir) + min(4, len(self.recent))) * (per_step + 96),
                    **({'prefix4': (len(self.prefix) + min(4, len(self.recent))) * (per_step + 96)} if self.keep_prefix else {})},
                'accounting': 'Tensor payload plus 96-byte logical metadata/slot; Python/framework overhead and shared decoder buffers reported separately, not claimed as exact RSS.'}


def expand_pooled(step):
    """Expand constant bins only for native packing; this creates no new evidence."""
    h, w = step.grid
    values = step.tokens.reshape(4, 4, -1)
    dense = step.tokens.new_empty((h, w, step.tokens.shape[-1]))
    for y in range(4):
        for x in range(4):
            dense[y * h // 4:(y + 1) * h // 4, x * w // 4:(x + 1) * w // 4] = values[y, x]
    restored = dense.reshape(h * w, -1)
    if not torch.equal(grid_pool(restored, step.grid, (4, 4))[0], step.tokens):
        raise ValueError('pooled-feature expansion changed evidence')
    return VisualStep(restored, step.grid, step.end, step.times)


def schedule(metadata, end):
    pts, keep = presentation_frames(metadata['frames'], drop_bad_pts=True)
    times = pts - pts[0]
    valid = np.flatnonzero(keep)
    if end > times[valid[-1]] + 1e-5 or end < 4:
        raise ValueError('query outside available causal video')
    slots = np.arange(0., end + 1e-8, .5)
    indices = valid[np.maximum(0, np.searchsorted(times[valid], slots, side='right') - 1)]
    indices = np.unique(indices)
    indices = indices[:len(indices) // 2 * 2]
    return indices, times[indices].reshape(-1, 2), int((~keep).sum())


def stream_pixels(path, indices, height, width, *, frames_per_block=2):
    """Single FFmpeg process; only one requested RGB block is resident here."""
    if frames_per_block not in (1, 2) or len(indices) % frames_per_block:
        raise ValueError('one/two-frame decoder blocks must divide selected frame count')
    expr = '+'.join(f'eq(n\\,{int(i)})' for i in indices)
    command = ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-threads', '1', '-noautorotate',
               '-i', str(path), '-map', '0:v:0', '-vf', f'select={expr},scale={width}:{height}',
               '-vsync', '0', '-frames:v', str(len(indices)), '-filter_threads', '1',
               '-threads', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    size = frames_per_block * height * width * 3
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            for _ in range(len(indices) // frames_per_block):
                data = bytearray()
                while len(data) < size:
                    block = process.stdout.read(size - len(data))
                    if not block:
                        raise ValueError('stream decoder ended before the selected prefix')
                    data.extend(block)
                yield np.frombuffer(data, dtype=np.uint8).reshape(frames_per_block, height, width, 3).copy()
            if process.wait(timeout=30) != 0:
                errors.seek(0)
                raise ValueError('stream decoder failed: ' + errors.read(2000).decode(errors='replace'))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()


def choose_sources(activity, ovo, count):
    # Selection precedes any model outputs. Every ActivityNet source supports the
    # SAME three query arrivals; no different-video delay comparison.
    eligible = [r for r in activity if r['query_time'] + 60 <= r['media_duration'] and r.get('evidence')]
    selected, groups = [], set()
    for cohort, rows, waits in (
        ('activitynet_delayed', eligible, (0, 15, 60)),
        ('ovo_long', [r for r in ovo if r['task'] == 'EPM' and r['query_time'] >= 300], (0,)),
    ):
        added = 0
        for row in sorted(rows, key=lambda r: digest(['longstream-v1', r['id']])):
            if row['source_group'] in groups:
                continue
            groups.add(row['source_group'])
            queries = [{**row, 'id': row['id'] + f'-wait{delay}', 'original_id': row['id'],
                        'query_time': row['query_time'] + delay, 'additional_wait_seconds': delay,
                        'cohort': cohort} for delay in waits]
            selected.append({'source_group': row['source_group'], 'video': row['video'], 'queries': queries})
            added += 1
            if added == count:
                break
        if added != count:
            raise ValueError(f'insufficient independent sources for {cohort}: {added}/{count}')
    return selected


def prepare(args):
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    def read(p):
        return [json.loads(x) for x in p.read_text().splitlines()]
    activity = read(args.activity / 'dev.jsonl')
    ovo = [r for r in read(args.ovo) if r['split'] == 'val']
    sources = choose_sources(activity, ovo, args.sources)
    train = json.loads((args.plain.parent / 'run.json').read_text())
    train_hashes = set(train['videos'].values())
    checkpoint = checked_checkpoint(args.plain, train['id'])
    cache = VideoProbeCache(root / 'media', stage_media=True)
    seen_hashes = set()
    for source in sources:
        local, meta, checksum = cache.source(source['video'], stage=True)
        if checksum in train_hashes or checksum in seen_hashes:
            raise ValueError('training overlap or duplicated video bytes in diagnostic')
        seen_hashes.add(checksum)
        metadata = root / 'metadata' / f'{checksum}.metadata.json'
        write_json(metadata, meta)
        source.update({'original_video': source['video'], 'video': str(local),
                       'video_sha256': checksum, 'metadata': str(metadata),
                       'metadata_sha256': sha256(metadata)})
        for query in source['queries']:
            query['video'] = str(local)
        schedule(meta, max(q['query_time'] for q in source['queries']))
    identity = {'protocol': 'bounded-longstream-diagnostic-v1', 'scope': SCOPE,
                'activity_manifest_sha256': sha256(args.activity / 'dev.jsonl'),
                'ovo_manifest_sha256': sha256(args.ovo), 'model': str(args.model),
                'model_identity': checkpoint_identity(args.model), 'plain': str(args.plain),
                'checkpoint': checkpoint, 'capacity': 32, 'seed': 17, 'pixels': 65536,
                'sources': sources, 'source_code_sha256': sha256(Path(__file__)),
                'questions': sum(len(s['queries']) for s in sources)}
    identity['id'] = digest(identity)
    ensure_run_manifest(root / 'inputs.json', identity)
    write_json(root / 'data-audit.json', {'source_videos': len(sources), 'questions': identity['questions'],
        'all_staged_on_data': all(str(Path(s['video']).resolve()).startswith('/data/') for s in sources),
        'training_video_hash_overlap': False, 'source_counts': {
            c: sum(s['queries'][0]['cohort'] == c for s in sources) for c in ('activitynet_delayed', 'ovo_long')},
        'scope': SCOPE})
    print(f'PREPARED {len(sources)} sources, {identity["questions"]} questions', flush=True)


def verify_native_step(bb, layout, raw, record):
    """The native parity helper's prefix needs the layout bound to its record."""
    return check_native_forward(bb, layout, [raw], {**record, 'native_layout': layout})


@torch.no_grad()
def text_answer(bb, query, video_hash):
    question, _ = question_and_answer({**query, 'answer': 0})
    prompt = bb.processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'text', 'text': question}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = bb.tokenizer(prompt, add_special_tokens=False)['input_ids']
    embeds = bb.embed_ids(ids)
    positions = torch.arange(len(ids), device=bb.device).expand(3, -1)
    hidden = bb.lm(embeds, positions, cache=None, use_cache=False)[:, -1:]
    candidates = [bb.ids(chr(65 + i)) for i in range(len(query['options']))]
    if any(len(ids) != 1 for ids in candidates):
        raise ValueError('single-token answer options required')
    logits = bb.logits(hidden)[0, 0, [i[0] for i in candidates]].float()
    fp32 = bb.lm_head.weight[[i[0] for i in candidates]].float() @ hidden[0, 0].float()
    return {'choice': int(logits.argmax()), 'choice_fp32': int(fp32.argmax()),
            'option_logits': logits.tolist(), 'option_logits_fp32': fp32.tolist(),
            'visual_tokens': 0, 'read_frame_indices': [],
            'read_times': [], 'real_feature_rgb_hashes': [], 'source_video_sha256': video_hash,
            'layout_uses_zero_pixels_only_for_packing': False, 'query_time_encoder_calls': 0}


@torch.no_grad()
def answer(bb, qa, selected, query, metadata, hw, video_hash):
    if not selected or len(selected) > 8 or any(s.end > query['query_time'] for s in selected):
        raise ValueError('invalid bounded/causal read')
    # Only placeholder positions are constructed with zeros. These pixels NEVER
    # enter the vision tower; genuine streamed feature hashes are recorded below.
    images = np.zeros((len(selected), 2, *hw, 3), dtype=np.uint8)
    times = np.asarray([s.times for s in selected])
    indices = [i for s in selected for i in s.indices]
    audit = {'source_fps': float(Fraction(metadata['streams'][0]['avg_frame_rate'])),
             'source_frame_count': len(metadata['frames']), 'frame_indices': indices,
             'decoded_rgb_sha256': hashlib.sha256(images.tobytes()).hexdigest()}
    safe = {**query, 'answer': 0}  # gold is not passed into prompt construction
    safe['native_layout'] = prepare_native(bb.processor, images, times, safe, audit,
                                           allow_native_timestamp_approximation=True)
    visual = [expand_pooled(s) for s in selected]
    embeds, positions, _, _, info = qa.prefix(visual, safe)
    hidden = bb.lm(embeds, positions, cache=None, use_cache=False)[:, -1:]
    candidates = [bb.ids(chr(65 + i)) for i in range(len(query['options']))]
    if any(len(ids) != 1 for ids in candidates):
        raise ValueError('single-token answer options required')
    logits = bb.logits(hidden)[0, 0, [i[0] for i in candidates]].float()
    choice = int(logits.argmax())
    fp32 = bb.lm_head.weight[[i[0] for i in candidates]].float() @ hidden[0, 0].float()
    if info['visual_tokens_total'] != len(selected) * 16 or info['visual_tokens_total'] > 128:
        raise ValueError('read budget exceeded')
    return {'choice': choice, 'choice_fp32': int(fp32.argmax()), 'option_logits': logits.tolist(),
            'option_logits_fp32': fp32.tolist(),
            'visual_tokens': info['visual_tokens_total'], 'read_frame_indices': indices,
            'read_times': times.tolist(), 'real_feature_rgb_hashes': [s.rgb_sha256 for s in selected],
            'source_video_sha256': video_hash, 'layout_uses_zero_pixels_only_for_packing': True,
            'query_time_encoder_calls': 0}


def evaluate_source(bb, inputs, source, out, verify):
    if sha256(source['video']) != source['video_sha256'] or sha256(source['metadata']) != source['metadata_sha256']:
        raise ValueError('staged data changed')
    metadata = json.loads(Path(source['metadata']).read_text())
    queries = sorted(source['queries'], key=lambda q: q['query_time'])
    indices, times, dropped = schedule(metadata, queries[-1]['query_time'])
    stream = metadata['streams'][0]
    hw = frame_size(stream['height'], stream['width'], inputs['pixels'], 32)
    bank = BoundedBank(inputs['capacity'], source.get('memory_seed', int(digest(source['source_group'])[:8], 16)),
                       keep_prefix=inputs.get('position_controls', False))
    qa = NativeVideoQALoss(bb, mode='grid', pool_grid=(4, 4))
    evidence = queries[0].get('evidence')
    privileged = set()
    if evidence:
        possible = [i for i, pair in enumerate(times) if any(evidence[0] <= t <= evidence[1] for t in pair)]
        if possible:
            privileged = set(np.asarray(possible)[np.linspace(0, len(possible) - 1, min(4, len(possible))).round().astype(int)].tolist())
    oracle, qindex, encoded, results = [], 0, 0, []
    started = time.monotonic()

    def ask(query):
        state = bank.audit()
        read_arms = ARMS + (('prefix4',) if inputs.get('position_controls') else ())
        predictions = {arm: answer(bb, qa, bank.read(arm), query, metadata, hw, source['video_sha256'])
                       for arm in read_arms}
        predictions['text_only'] = text_answer(bb, query, source['video_sha256'])
        if oracle:
            selected = sorted(oracle + list(bank.recent)[-4:], key=lambda s: s.end)
            # The query arrival is always later than original evidence+recent.
            selected = list({s.indices: s for s in selected}.values())
            predictions['evidence_oracle'] = answer(bb, qa, selected, query, metadata, hw, source['video_sha256'])
            predictions['stored_oracle'] = answer(bb, qa, bank.read('stored_oracle', evidence), query, metadata, hw, source['video_sha256'])
            if inputs.get('position_controls'):
                predictions['fifo_stored_oracle'] = answer(bb, qa, bank.read('fifo_stored_oracle', evidence), query, metadata, hw, source['video_sha256'])
        if not results:
            replay = answer(bb, qa, bank.read('recent8'), query, metadata, hw, source['video_sha256'])
            if replay != predictions['recent8']:
                raise ValueError('fresh independent question replay changed output')
        if bank.audit() != state:
            raise ValueError('query mutated streaming memory')
        result = {'id': query['id'], 'input_id': inputs['id'],
                  'source_group': source['source_group'], 'cohort': query['cohort'],
                  'gold': query['answer'], 'query_time': query['query_time'],
                  'additional_wait_seconds': query['additional_wait_seconds'],
                  'evidence_gap_seconds': query['query_time'] - evidence[1] if evidence else None,
                  'stream_encoder_calls': encoded, 'memory': state,
                  'privileged_oracle_retained_tokens': len(oracle) * 16,
                  'predictions': predictions, 'elapsed_seconds': time.monotonic() - started}
        results.append(result)
        write_json(out / 'questions' / f'{digest(query["id"])}.json', result)
        print(json.dumps({'question': query['id'], 'steps': encoded, 'arms': len(predictions)}), flush=True)

    pixels = stream_pixels(source['video'], indices, *hw)
    try:
        for i, frames in enumerate(pixels):
            pair = tuple(float(t) for t in times[i])
            while qindex < len(queries) and queries[qindex]['query_time'] < pair[-1]:
                ask(queries[qindex])
                qindex += 1
            if verify and i == 0:
                audit = {'source_fps': float(Fraction(stream['avg_frame_rate'])),
                         'source_frame_count': len(metadata['frames']), 'frame_indices': indices[:2].tolist(),
                         'decoded_rgb_sha256': hashlib.sha256(frames[None].tobytes()).hexdigest()}
                row = {**queries[0], 'answer': 0}
                layout = prepare_native(bb.processor, frames[None], np.asarray([pair]), row, audit,
                                        allow_native_timestamp_approximation=True)
            values, grid = bb.preprocess(frames)
            dense = bb.encode(values, grid)
            encoded += 1
            shape = bb.merged_grid(grid)
            raw = VisualStep(dense, shape, pair[-1], pair)
            if verify and i == 0:
                write_json(out / f'native-parity-{digest(source["source_group"])}.json', verify_native_step(bb, layout, raw, row))
            pooled = grid_pool(dense, shape, (4, 4))[0].detach().clone()
            step = StoredStep(pooled, shape, pair, tuple(int(x) for x in indices[2 * i:2 * i + 2]),
                              hashlib.sha256(frames.tobytes()).hexdigest())
            bank.append(step)
            if i in privileged:
                oracle.append(step)
            if len(bank.fifo) > 32 or len(bank.reservoir) > 32 or len(bank.recent) > 8:
                raise ValueError('unbounded retained streaming features')
            if i % 30 == 0:
                write_json(out / f'progress-{os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")}.json',
                           {'source': source['source_group'], 'steps': encoded, 'target_steps': len(times),
                            'queries_completed_this_source': len(results), 'updated_unix': time.time()})
        while qindex < len(queries):
            ask(queries[qindex])
            qindex += 1
    finally:
        pixels.close()
    write_json(out / 'sources' / f'{digest(source["source_group"])}.json', {
        'completed': True, 'input_id': inputs['id'], 'source': source['source_group'],
        'queries': [r['id'] for r in results],
        'stream_encoder_calls': encoded, 'sampled_steps': len(times), 'dropped_bad_pts_frames': dropped,
        'peak_two_frame_rgb_buffer_bytes': 2 * hw[0] * hw[1] * 3,
        'shared_input_pts_index_bytes': len(metadata['frames']) * 8,
        'verification_extra_encoder_calls': int(verify),
        'scope': 'Source PTS index/FFmpeg buffers are shared input infrastructure, not readable historical memory. Experiment JSON traces on disk are not used by memory policies.'})


def worker(args):
    if not 0 <= args.shard < args.shards:
        raise ValueError('invalid worker shard')
    inputs = json.loads((args.out / 'inputs.json').read_text())
    if sha256(Path(__file__)) != inputs['source_code_sha256']:
        raise ValueError('source code differs from prepared run')
    if checked_checkpoint(Path(inputs['plain']), inputs['checkpoint']['run']) != inputs['checkpoint']:
        raise ValueError('frozen Plain checkpoint changed')
    selected = inputs['sources'][args.shard::args.shards]
    if args.smoke:
        smoke_groups = inputs.get('smoke_source_groups')
        selected = [s for s in selected if s['source_group'] in smoke_groups] if smoke_groups else [
            s for s in selected if s in inputs['sources'][:1] + inputs['sources'][len(inputs['sources']) // 2:len(inputs['sources']) // 2 + 1]]
    if not selected:
        return
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    bb = Backbone(BackboneConfig(name=inputs['model'], dtype='bfloat16', local_files_only=True),
                  StreamConfig(pixels_per_frame=inputs['pixels']), device='cuda:0')
    bb.add_lora(8, 16, dropout=0., adapter=str(Path(inputs['plain']) / 'adapter'))
    bb.model.eval().requires_grad_(False)
    for number, source in enumerate(selected):
        done = args.out / 'sources' / f'{digest(source["source_group"])}.json'
        if done.exists():
            previous = json.loads(done.read_text())
            if previous.get('input_id') != inputs['id'] or not previous.get('completed'):
                raise ValueError('completed source belongs to another run')
            continue
        evaluate_source(bb, inputs, source, args.out, verify=number == 0)


def report(args):
    inputs = json.loads((args.out / 'inputs.json').read_text())
    sources = inputs['sources']
    if args.smoke:
        sources = sources[:1] + sources[len(sources) // 2:len(sources) // 2 + 1]
    rows = []
    for source in sources:
        done = json.loads((args.out / 'sources' / f'{digest(source["source_group"])}.json').read_text())
        if (not done['completed'] or done.get('input_id') != inputs['id']
                or done['stream_encoder_calls'] != done['sampled_steps']):
            raise ValueError('source did not finish its single encoding pass')
        for query in source['queries']:
            row = json.loads((args.out / 'questions' / f'{digest(query["id"])}.json').read_text())
            if (row['id'] != query['id'] or row['gold'] != query['answer']
                    or row.get('input_id') != inputs['id'] or not set(BASELINES) <= set(row['predictions'])):
                raise ValueError('question identity drift')
            if any(p['visual_tokens'] > 128 or p['query_time_encoder_calls'] != 0
                   or max((t for pair in p['read_times'] for t in pair), default=0) > row['query_time']
                   for p in row['predictions'].values()):
                raise ValueError('causality, read budget or encoder replay violation')
            rows.append(row)
    groups = {}
    for cohort in ('activitynet_delayed', 'ovo_long'):
        population = [r for r in rows if r['cohort'] == cohort]
        for wait in sorted({r['additional_wait_seconds'] for r in population}):
            subset = [r for r in population if r['additional_wait_seconds'] == wait]
            groups[f'{cohort}/wait{wait}'] = {'n': len(subset), 'scores': {
                arm: {'n': len([r for r in subset if arm in r['predictions']]),
                      'correct': sum(r['predictions'][arm]['choice'] == r['gold'] for r in subset if arm in r['predictions'])}
                for arm in BASELINES + ORACLES}, 'actual_evidence_gaps_seconds': [r['evidence_gap_seconds'] for r in subset]}
    write_json(args.out / ('smoke-ok.json' if args.smoke else 'report.json'), {
        'completed': True, 'smoke': args.smoke, 'input_id': inputs['id'], 'questions': len(rows),
        'sources': len(sources), 'results': groups, 'scope': SCOPE,
        'architecture_accepted': False, 'no_training': True,
        'limitations': ['Small previously used development population.',
                        'Caption labels are not visually certified.',
                        'No claimed novel learned memory or 10-minute/hour-scale retention.',
                        'No persistent decoder KV; history lives only in bounded visual banks.']})
    print(f'VERIFIED {"SMOKE" if args.smoke else "FULL"} LONGSTREAM: {len(rows)} questions', flush=True)


def run(args):
    import sys

    prepare(args)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'TRACE_WANDB': '0', 'TRACE_SWANLAB': '0', 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    run_jobs([('gpu-check', [sys.executable, '-c',
              'import torch; assert torch.cuda.device_count()==4; '
              'assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24'], list(range(4)))],
             env, args.out / 'logs')
    cmd = [sys.executable, '-m', 'trace_vl.core_vlm.longstream', 'worker', '--out', str(args.out)]
    print('SMOKE: one delayed source and one >=300s OVO source, all inference paths', flush=True)
    run_jobs([(f'smoke-{i}', [*cmd, '--shard', str(i), '--smoke'], [i]) for i in range(4)],
             env, args.out / 'logs')
    args.smoke = True
    report(args)
    smoke = json.loads((args.out / 'smoke-ok.json').read_text())
    if smoke['questions'] != 4 or smoke['sources'] != 2:
        raise ValueError('real-video smoke did not cover both cohorts')
    print('FULL: verified smoke passed; continue the fixed 16-source / 32-question plan', flush=True)
    run_jobs([(f'full-{i}', [*cmd, '--shard', str(i)], [i]) for i in range(4)], env, args.out / 'logs')
    args.smoke = False
    report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'worker', 'report', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/data/yyy/tracevl-data/core-pool-probe-2efc7cc/model'))
    parser.add_argument('--plain', type=Path, default=Path('/data/yyy/tracevl-runs/core-pool-2efc7cc-001/train/plain/checkpoint-002500'))
    parser.add_argument('--activity', type=Path, default=Path('/nas-data/vlm/yyy/tracevl-runs/history-reader-3e5cb79-001'))
    parser.add_argument('--ovo', type=Path, default=Path('/nas-data/vlm/yyy/tracevl-data/core-transfer-v1-f1c19a7/ovo.jsonl'))
    parser.add_argument('--sources', type=int, default=8)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'worker':
        worker(args)
    elif args.stage == 'run':
        run(args)
    else:
        report(args)


if __name__ == '__main__':
    main()
