"""Local, hash-checked replay of the exact observations from a completed run."""
from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from .data import decode_frame_indices, sha256
from .native import encode_native
from .prepare import digest, write_json
from .qa import VisualStep
from .train import tensor_digest


def observation_key(audit):
    return digest({k: audit[k] for k in ('video_sha256', 'frame_indices', 'frame_times',
                                       'frame_hw', 'decoded_rgb_sha256')})


def validate_observation(row, audit, steps):
    indices = np.asarray(audit['frame_indices'])
    times = np.asarray(audit['frame_times'], dtype=float)
    if (indices.shape != (2 * steps,) or times.shape != (steps, 2)
            or not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any()
            or (np.diff(indices) <= 0).any() or not np.isfinite(times).all()
            or (times < 0).any() or (np.diff(times.ravel()) <= 0).any()):
        raise ValueError('invalid reference frame selection')
    if times.max() > float(audit['query_time']):
        raise ValueError('reference contains future frames')
    if row.get('query_time') is not None and float(row['query_time']) != audit['query_time']:
        raise ValueError('reference query time differs')
    if audit['sampling'] != 'uniform-frame-index-within-observation-v1':
        raise ValueError('unsupported reference sampling protocol')


def copy_verified(source, target, expected):
    source, target = Path(source), Path(target)
    if target.is_file():
        if sha256(target) != expected:
            raise ValueError(f'local file changed: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f'.{os.getpid()}.tmp')
    try:
        shutil.copyfile(source, tmp)
        if sha256(tmp) != expected:
            raise ValueError(f'source does not match recorded bytes: {source}')
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def validate_images(images, audit):
    expected = (len(audit['frame_times']), 2, *audit['frame_hw'], 3)
    if images.shape != expected or images.dtype != np.uint8:
        raise ValueError('cached RGB layout changed')
    if hashlib.sha256(images.tobytes()).hexdigest() != audit['decoded_rgb_sha256']:
        raise ValueError('cached decoded RGB differs from original evaluation')


def read_frames(path, audit):
    with np.load(path, allow_pickle=False) as saved:
        images = saved['images']
    validate_images(images, audit)
    return images


def prepare_frames(local_video, path, audit):
    path = Path(path)
    if path.is_file():
        read_frames(path, audit)
        return
    h, w = audit['frame_hw']
    # Stopping after the final selected frame preserves the selected RGB bytes.
    # The original run already audited full decoding; verify its exact RGB hash.
    images = decode_frame_indices(local_video, audit['frame_indices'], h, w, 600,
                                  stop_after_selection=True).reshape(-1, 2, h, w, 3)
    validate_images(images, audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as handle:
            np.savez(handle, images=images)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def parallel(items, function, workers, label):
    with ThreadPoolExecutor(workers) as pool:
        pending = [pool.submit(function, item) for item in items]
        for number, future in enumerate(as_completed(pending), 1):
            future.result()
            if number % 25 == 0 or number == len(items):
                print(f'{label} {number}/{len(items)}', flush=True)


def stage_data(cases, model_source, model_identity, root, workers=16, reserve_gib=30):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    videos, observations = {}, {}
    for item in cases:
        row, audit = item['row'], item['reference']['observation']
        key = audit['video_sha256']
        videos.setdefault(key, (Path(row['video']), root / 'videos' / key, key))
        observations.setdefault(observation_key(audit), audit)
    copies = list(videos.values())
    for name, expected in model_identity.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('invalid relative model filename')
        copies.append((Path(model_source) / name, root / 'model' / name, expected))
    required = sum(source.stat().st_size for source, target, _ in copies if not target.exists())
    rgb_bytes = sum(2 * len(a['frame_times']) * int(np.prod(a['frame_hw'])) * 3
                    for a in observations.values())
    # Dense BF16 features plus RGB; reserve remains available after staging.
    required += 4 * rgb_bytes
    if shutil.disk_usage(root).free < required + reserve_gib * 2**30:
        raise ValueError(f'insufficient local space: need estimated {required / 2**30:.1f} GiB '
                         f'plus {reserve_gib} GiB reserve at {root}')
    parallel(copies, lambda x: copy_verified(*x), workers, 'LOCAL COPY')
    tasks = [(videos[a['video_sha256']][1], root / 'frames' / f'{key}.npz', a)
             for key, a in observations.items()]
    parallel(tasks, lambda x: prepare_frames(*x), workers, 'EXACT RGB')
    result = {'videos': len(videos), 'observations': len(observations),
              'questions': len(cases), 'rgb_bytes': rgb_bytes,
              'model': str(root / 'model'), 'frames': str(root / 'frames'),
              'verification': 'recorded source SHA256 and exact decoded-RGB SHA256',
              'no_ffprobe': True}
    write_json(root / 'media.json', result)
    return result


class FrozenVisionCache:
    """Share only frozen vision features, never LM or question-answer state."""

    def __init__(self, root, identity):
        self.root = Path(root) / digest(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = self.writes = 0

    def get(self, backbone, layout, *, verify_fresh=False):
        key = observation_key(layout.audit)
        path = self.root / f'{key}.pt'
        fresh = None
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['key'] != key or tensor_digest({'features': saved['features']}) != saved['sha256']:
                raise ValueError('frozen feature cache changed')
            self.hits += 1
        else:
            fresh = encode_native(backbone, layout)
            features = torch.stack([s.tokens.cpu() for s in fresh])
            saved = {'key': key, 'grid': fresh[0].grid, 'features': features,
                     'sha256': tensor_digest({'features': features})}
            tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
            try:
                torch.save(saved, tmp)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
            self.writes += 1
        if verify_fresh:
            fresh = fresh or encode_native(backbone, layout)
            if not torch.equal(saved['features'], torch.stack([s.tokens.cpu() for s in fresh])):
                raise ValueError('fresh and cached frozen vision features differ')
        return [VisualStep(x, tuple(saved['grid']), pair[-1], tuple(pair))
                for x, pair in zip(saved['features'], layout.frame_times, strict=True)]
