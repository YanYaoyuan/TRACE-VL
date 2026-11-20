"""Reuse the audited training mixture and external populations without rescanning archives."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .answerability import reviewed_training_pairs
from .data import read_split, sha256
from .mixed_data import frozen_external
from .prepare import digest, write_json


def reuse_data(cfg, out):
    out = Path(out)
    root = Path(cfg['mixture_data'])
    complete = json.loads((root / 'complete.json').read_text())
    inputs = json.loads((root / 'inputs.json').read_text())
    if complete['inputs'] != inputs['id'] or digest({k: v for k, v in inputs.items() if k != 'id'}) != inputs['id']:
        raise ValueError('mixture provenance differs')
    for name in ('manifest.jsonl', 'inventory.json', 'media.json', 'annotation-audit.json'):
        if sha256(root / name) != complete['files'][name]:
            raise ValueError('frozen mixture/audit changed')
    for key in ('steps', 'pixels', 'pool_grid', 'seed'):
        if cfg[key] != inputs['config'][key]:
            raise ValueError(f'mixture observation protocol differs: {key}')
    external, external_complete = frozen_external(cfg['external_data'])
    if inputs['external'] != external_complete:
        raise ValueError('frozen external data changed since training mixture exclusion')
    rows = read_split(root / 'manifest.jsonl', 'train')
    paths = {r['video'] for r in rows}
    if paths != set(complete['videos']):
        raise ValueError('mixture video inventory differs')
    with ThreadPoolExecutor(min(8, cfg['workers'])) as pool:
        for i, (p, h) in enumerate(pool.map(lambda p: (p, sha256(p)), sorted(paths)), 1):
            if h != complete['videos'][p]:
                raise ValueError('frozen training media changed')
            if i % 100 == 0 or i == len(paths):
                print(f'REUSED MEDIA CHECK {i}/{len(paths)} (hash only; no archive scan)', flush=True)
    if set(r.get('mixture_family') for r in rows) != {'general', 'temporal'}:
        raise ValueError('expected general and temporal mixture')
    inventory = json.loads((root / 'inventory.json').read_text())
    if inventory['source_overlap'] != 0 or inventory['exact_byte_overlap'] != 0:
        raise ValueError('original mixture overlap audit failed')
    review = cfg.get('reviewed_pairs')
    extra = reviewed_training_pairs(review, rows, steps=cfg['steps'], pixels=cfg['pixels']) if review else []
    identity = {'original_complete': sha256(root / 'complete.json'), 'original_inputs': sha256(root / 'inputs.json'),
                'manifest': sha256(root / 'manifest.jsonl'), 'external_complete': external_complete,
                'reviewed_pairs_sha256': sha256(review) if review else None,
                'extra_rows': digest(extra), 'implementation': sha256(__file__)}
    identity['id'] = digest(identity)
    out.mkdir(parents=True, exist_ok=True)
    ensure_run_manifest(out / 'reuse.json', identity)
    final = sorted(rows + extra, key=lambda r: r['id'])
    if len({r['id'] for r in final}) != len(final):
        raise ValueError('review IDs collide with original training')
    text = ''.join(json.dumps(r) + '\n' for r in final)
    path = out / 'manifest.jsonl'
    if path.exists() and path.read_text() != text:
        raise ValueError('saved training mixture changed')
    atomic_text(path, text)
    write_json(out / 'supervision.json', {'families': dict(Counter(r['mixture_family'] for r in final)),
        'reviewed_pairs': len(extra) // 2, 'review_file': review,
        'answerability_supervision_active': bool(extra),
        'note': 'No synthetic unanswerable labels. With no reviewed file this is preservation-only. '
                'Both labels in each reviewed training pair; no OVO/final-test supervision.'})
    if review:
        atomic_text(out / 'reviewed-pairs.jsonl', Path(review).read_text())
    return path, external
