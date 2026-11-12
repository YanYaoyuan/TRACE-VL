from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import bind, digest, read_json, sha, write_json


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)
    write_json(path.with_suffix('.sha.json'), {'sha256': sha(path)})


def valid_tensor(path):
    path = Path(path)
    receipt = path.with_suffix('.sha.json')
    return path.is_file() and receipt.is_file() and read_json(receipt)['sha256'] == sha(path)


def model_complete(root):
    root = Path(root)
    if not (root/'config.json').is_file():
        return False
    index = root/'model.safetensors.index.json'
    weights = set(read_json(index)['weight_map'].values()) if index.is_file() else {'model.safetensors'}
    return bool(weights) and all((root/f).is_file() for f in weights)


def model_identity(root):
    root = Path(root)
    files = sorted(p for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts
                   and p.suffix in {'.json', '.safetensors', '.txt', '.jinja'})
    if not model_complete(root):
        raise ValueError(f'No model weights in {root}')
    return {str(p.relative_to(root)): sha(p) for p in files}


def causal_frames(path, targets):
    """Sequential decode and presentation timestamps; never select a frame after target."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    previous, lookahead, previous_time = None, None, -1.0
    try:
        for target in targets:
            while True:
                if lookahead is None:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if not math.isfinite(timestamp) or timestamp <= previous_time:
                        raise ValueError(f'Unreliable/non-increasing decoded timestamps: {path}')
                    previous_time = timestamp
                    lookahead = (timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if lookahead[0] > target+1e-7:
                    break
                previous, lookahead = lookahead, None
            if previous is None or target-previous[0] > 1:
                raise ValueError(f'No recent causal frame at {target}: {path}')
            yield target, previous[0], previous[1]
    finally:
        cap.release()


def encode_shard(rows, model_root, out, cfg, shard=0, shards=1, device='cuda:0'):
    import numpy as np

    from trace_vl.v3.backbone import Backbone
    from trace_vl.v3.config import BackboneConfig, StreamConfig
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['source_video_id']].append(r)
    groups = sorted(grouped)[shard::shards]
    if not groups:
        write_json(out/f'profile-{shard}.json', {'videos': 0})
        return
    required = [out/f'video-{digest(key)[:24]}.pt' for key in groups]
    required += [out/f"query-{r['id']}.pt" for key in groups for r in grouped[key]]
    if all(valid_tensor(p) for p in required):
        print(f'FEATURE SKIP verified shard={shard}; original timing profile retained', flush=True)
        return
    start = time.monotonic()
    model = Backbone(BackboneConfig(name=str(model_root), local_files_only=True),
                     StreamConfig(pixels_per_frame=cfg['pixels']), device)
    torch.cuda.reset_peak_memory_stats(device)
    stats = {'videos': 0, 'ticks': 0, 'conditions': 0, 'width': model.hidden, 'grid_thw_examples': []}
    for key in groups:
        records = grouped[key]
        ticks = sorted({t for r in records for t in r['ticks']})
        name = digest(key)[:24]
        video_path = out/f'video-{name}.pt'
        if not valid_tensor(video_path):
            target_times = sorted({t-.5 for t in ticks} | set(ticks))
            iterator = iter(causal_frames(records[0]['media_path'], target_times))
            pending, encoded, observed = {}, [], []
            for t in ticks:
                for target in (t-.5, t):
                    while target not in pending:
                        want, actual, frame = next(iterator)
                        pending[want] = (actual, frame)
                frames = np.stack([pending[t-.5][1], pending[t][1]])
                with torch.inference_mode():
                    pixels, grid = model.preprocess(model.resize_step(frames))
                    values = model.encode(pixels, grid)
                    h, w = model.merged_grid(grid)
                    if int(grid[0, 0]) != 1 or h*w != values.shape[0]:
                        raise ValueError('Unexpected temporal/grid shape; cannot pool safely')
                    pooled = F.adaptive_avg_pool2d(values.T.reshape(1, model.hidden, h, w).float(),
                                                  (cfg['grid'], cfg['grid']))
                    encoded.append(pooled.flatten(2)[0].T.to(torch.bfloat16).cpu())
                observed.append([pending[t-.5][0], pending[t][0]])
                pending = {k: v for k, v in pending.items() if k > t}
                if len(stats['grid_thw_examples']) < 3:
                    stats['grid_thw_examples'].append(grid.tolist())
            save_tensor(video_path, {'visual': torch.stack(encoded), 'ticks': ticks,
                                     'actual_frame_times': observed, 'media_sha256': records[0]['media_sha256']})
        for r in records:
            p = out/f"query-{r['id']}.pt"
            if not valid_tensor(p):
                with torch.inference_mode():
                    q = model.encode_texts([r['condition']])[0].to(torch.bfloat16).cpu()
                save_tensor(p, {'query': q, 'condition_sha256': digest(r['condition'])})
        stats['videos'] += 1
        stats['ticks'] += len(ticks)
        stats['conditions'] += len(records)
        print(f"FEATURE shard={shard} {stats['videos']}/{len(groups)} video={key}", flush=True)
    torch.cuda.synchronize(device)
    stats.update(seconds=time.monotonic()-start, peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
    write_json(out/f'profile-{shard}.json', stats)


def bind_features(out, rows, cfg, model_hashes, code):
    bind(Path(out)/'identity.json', {'code': code, 'manifest': digest(rows), 'model': model_hashes,
                                  'pixels': cfg['pixels'], 'grid': cfg['grid'], 'extractor': 'causal-pts-v1'})
