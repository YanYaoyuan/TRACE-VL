"""Explicit, pre-training missing-media amendment; never resample or edit old runs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_vl.v3.run_io import atomic_text, ensure_run_manifest

from .data import sha256
from .prepare import digest


def reuse_failed_preparation(cfg, out):
    prior = Path(cfg['repair_from']).resolve()
    out = Path(out).resolve()
    if prior == out or prior in out.parents or out in prior.parents:
        raise ValueError('repair requires a separate output directory')
    old = json.loads((prior / 'preparation.json').read_text())
    if old['config'] != {k: v for k, v in cfg.items() if k not in ('repair_from', 'exclude_training_videos')}:
        raise ValueError('repair must preserve the original scale configuration')
    if (prior / 'mixture/complete.json').exists() or (prior / 'release.json').exists():
        raise ValueError('repair only an unfinished, untrained preparation')
    names = cfg.get('exclude_training_videos', [])
    if not names or len(set(names)) != len(names) or any(Path(n).name != n or not n.endswith('.mp4') for n in names):
        raise ValueError('explicit unique missing training filenames required')
    media = json.loads((prior / 'mixture/media.json').read_text())
    if set(media['still_missing']) != set(names) or any(media[k] for k in ('ambiguous', 'extracted_but_unreadable', 'archives_skipped')):
        raise ValueError('declared exclusions must exactly match the prior missing-only failure')
    inputs = json.loads((prior / 'mixture/inputs.json').read_text())
    general = prior / 'general'
    complete = json.loads((general / 'complete.json').read_text())
    run = json.loads((general / 'run.json').read_text())
    verification = json.loads((general / 'verification.json').read_text())['videos']
    manifest = (general / 'manifest.jsonl').read_text()
    if (complete['run'] != run['id'] or complete['manifest_sha256'] != sha256(general / 'manifest.jsonl')
            or complete['manifest_sha256'] != inputs['general_manifest']
            or complete['id'] != digest([run['id'], {p: v['sha256'] for p, v in verification.items()}, manifest])):
        raise ValueError('prior general QA completion/provenance changed')
    record = {'prior_run': str(prior), 'excluded_videos': sorted(names),
              'prior_files': {n: sha256(prior / n) for n in ('preparation.json', 'mixture/media.json',
                  'mixture/inputs.json', 'general/complete.json', 'general/manifest.jsonl', 'general/verification.json')},
              'policy': 'post-selection exclusion of declared missing training media; no replacement; no evaluation changes'}
    record['id'] = digest(record)
    ensure_run_manifest(out / 'repair.json', record)
    # Metadata only: all absolute media paths and verification records stay valid.
    # The new mixture re-hashes selected media before trusting decode records.
    for name in ('manifest.jsonl', 'complete.json', 'run.json', 'verification.json', 'report.json',
                 'missing.json', 'protected-exclusions.json'):
        src, dest = general / name, out / 'general' / name
        if src.is_file():
            if dest.exists() and sha256(dest) != sha256(src):
                raise ValueError('reused general metadata changed: ' + name)
            atomic_text(dest, src.read_text())
    return record


def exclude_declared_missing(rows, cfg, identity):
    repair = cfg.get('media_repair')
    if not repair:
        return rows, None
    if repair.get('id') != digest({k: v for k, v in repair.items() if k != 'id'}):
        raise ValueError('repair record changed')
    prior = Path(repair['prior_run'])
    for name, expected in repair['prior_files'].items():
        if sha256(prior / name) != expected:
            raise ValueError('repair source changed: ' + name)
    media = json.loads((prior / 'mixture/media.json').read_text())
    if (set(media['still_missing']) != set(repair['excluded_videos'])
            or any(media[k] for k in ('ambiguous', 'extracted_but_unreadable', 'archives_skipped'))):
        raise ValueError('repair must only exclude recorded missing media')
    old = json.loads((prior / 'mixture/inputs.json').read_text())
    for key in ('train_csv', 'val_csv', 'general_manifest', 'external', 'blocked_ids'):
        if old[key] != identity[key]:
            raise ValueError('repair selection input changed: ' + key)
    for key in ('seed', 'questions_per_family'):
        if old['config'][key] != cfg[key]:
            raise ValueError('repair selection configuration changed: ' + key)
    names = set(repair['excluded_videos'])
    removed = [r for r in rows if r['mixture_family'] == 'temporal' and r['video'] in names]
    if {r['video'] for r in removed} != names or any(r['split'] != 'train' for r in removed):
        raise ValueError('declared missing video is not in the fixed temporal training selection')
    ids = {r['id'] for r in removed}
    kept = [r for r in rows if r['id'] not in ids]
    audit = {'repair': repair, 'selected_before': len(rows), 'retained_after': len(kept),
             'selection_sha256': digest(rows), 'retained_selection_sha256': digest(kept),
             'excluded_questions': [{k: r[k] for k in ('id', 'video', 'mixture_family', 'split')} for r in removed],
             'excluded_by_family': dict(Counter(r['mixture_family'] for r in removed)),
             'replacement_questions': 0}
    return kept, audit
