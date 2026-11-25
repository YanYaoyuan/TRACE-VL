"""Outcome-independent anchored past-state diagnostic; frozen model, no training."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream
from . import longstream_delayed as core
from .control import checkpoint_identity
from .data import sha256
from .experiment import run_jobs
from .prepare import digest, write_json
from .train import checked_checkpoint

OLD_DEVELOPMENT_MANIFEST = Path('/nas-data/vlm/yyy/tracevl-runs/streaming-vision-a566104-001/streamingbench.jsonl')
TASKS = {'Action Perception', 'Object Perception'}
SCOPE = (
    'Controlled anchored past-state diagnostic on existing development videos. All250 old sources '
    'considered using fullPTS; Concurrent Action/Object questions, cutoff>=60,300s natural continuation; '
    'one earliest eligible target per source, never selected by outcomes. Original questions/options/'
    'gold unchanged except explicit original-cutoff anchoring. All targets at0/15/60/180/300s. '
    'Frozen Plain LoRA2500;32 historical+4 recent slots,128 read tokens. Privileged snapshot8 is not '
    'a semantic oracle; recent8_now is a later-frame control. FP32 letter head over bf16 hidden primary; '
    'bf16 secondary. No training/new reserved sources/audio/subtitles/query-time vision/LM KV memory. '
    'Not official StreamingBench or unseen validation. No architecture promotion from this diagnostic.'
)


def validate_rows(rows, allowed, spec):
    if (not rows or len(rows) != spec['original_targets'] or len(rows) != spec['original_sources']
            or len({q['id'] for q in rows}) != len(rows)
            or len({q['source_group'] for q in rows}) != len(rows)
            or not {q['source_group'] for q in rows} <= allowed
            or sorted(q['id'] for q in rows) != spec['question_ids']
            or sorted(q['source_group'] for q in rows) != spec['source_groups']):
        raise ValueError('complete one-target-per-source population required')
    if any(not math.isfinite(q['query_time']) or q['query_time'] < 60
           or q.get('temporal_clue_type') != 'Concurrent' or q['task'] not in TASKS
           or q.get('frames_required') not in ('single', 'multiple') or q.get('evidence') is not None for q in rows):
        raise ValueError('frozen state task/metadata changed')


def choose_smoke(sources):
    if len(sources) < 2:
        raise ValueError('two distinct sources required')
    ordered = sorted(sources, key=lambda s: (s['planned_stream_encoder_calls'], s['source_group']))
    return sorted([ordered[0]['source_group'], ordered[-1]['source_group']])


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.reference / 'inputs.json').read_text())
    if base['id'] != core.REFERENCE_ID or digest({k: v for k, v in base.items() if k != 'id'}) != core.REFERENCE_ID:
        raise ValueError('frozen natural model reference changed')
    ready = json.loads((args.data / 'ready.json').read_text())
    if (not ready['completed'] or ready['id'] != args.data_id
            or digest({k: v for k, v in ready.items() if k != 'id'}) != args.data_id
            or sha256(args.data / 'questions.jsonl') != ready['questions_sha256']):
        raise ValueError('fully verified state data changed')
    spec = json.loads((args.data / 'candidate-spec.json').read_text())
    rule = json.loads((args.data / 'selection-rule.json').read_text())
    if (digest({k: v for k, v in spec.items() if k != 'id'}) != spec['id'] or spec['id'] != ready['candidate_spec_id']
            or digest({k: v for k, v in rule.items() if k != 'id'}) != rule['id']
            or rule['id'] != ready['selection_rule_id'] or rule['id'] != spec['selection_rule_id']
            or spec['completed_inventory_sources'] != 250 or rule['tasks'] != sorted(TASKS)
            or rule['selection_uses_model_outcomes'] or spec['model_outcomes_used']
            or rule['delays_seconds'] != list(core.DELAYS)):
        raise ValueError('outcome-independent state selection changed')
    if sha256(OLD_DEVELOPMENT_MANIFEST) != spec['old_development_manifest_sha256']:
        raise ValueError('existing development boundary changed')
    allowed = {q['source_group'] for q in map(json.loads, OLD_DEVELOPMENT_MANIFEST.read_text().splitlines())}
    rows = [json.loads(line) for line in (args.data / 'questions.jsonl').read_text().splitlines()]
    validate_rows(rows, allowed, spec)
    if checked_checkpoint(Path(base['plain']), base['checkpoint']['run']) != base['checkpoint']:
        raise ValueError('frozen checkpoint changed')
    if checkpoint_identity(Path(base['model'])) != base['model_identity']:
        raise ValueError('base weights changed')
    training_hashes = set(json.loads((Path(base['plain']).parent / 'run.json').read_text())['videos'].values())
    if sorted(s['source_group'] for s in ready['sources']) != spec['source_groups']:
        raise ValueError('data source population changed')
    sources = []
    for s in sorted(ready['sources'], key=lambda s: s['source_group']):
        if (sha256(s['video']) != s['video_sha256'] or sha256(s['metadata']) != s['metadata_sha256']
                or s['video_sha256'] in training_hashes or not s['video'].startswith('/data/')):
            raise ValueError('source bytes/PTS/location or training disjointness changed')
        original = next(q for q in rows if q['source_group'] == s['source_group'])
        if original['video'] != s['video']:
            raise ValueError('question media does not match source')
        queries = core.delayed_queries([original], s['last_valid_source_time'])
        if len(queries) != 5:
            raise ValueError('every target must have all5 delays')
        metadata = json.loads(Path(s['metadata']).read_text())
        indices, times, dropped = longstream.schedule(metadata, max(q['query_time'] for q in queries))
        for q in queries:
            q['planned_encoder_calls'] = int(sum(pair[-1] <= q['query_time'] for pair in times))
            q['target_encoder_calls'] = int(sum(pair[-1] <= q['target_time'] for pair in times))
        sources.append({'source_group': s['source_group'], 'video': s['video'], 'video_sha256': s['video_sha256'],
                        'metadata': s['metadata'], 'metadata_sha256': s['metadata_sha256'], 'queries': queries,
                        'last_valid_source_time': s['last_valid_source_time'],
                        'sampled_frame_indices': indices.tolist(), 'sampled_frame_times': times.tolist(),
                        'planned_stream_encoder_calls': len(times), 'dropped_bad_pts_frames': dropped,
                        'memory_seed': int(digest(['natural-longprefix-writer-v1', s['source_group']])[:8], 16),
                        'events': [list(e) for e in core.events_for(queries)]})
    inputs = {**{k: copy.deepcopy(base[k]) for k in ('model', 'model_identity', 'plain', 'checkpoint', 'pixels',
                                                   'capacity', 'seed', 'primary_score', 'secondary_score')},
              'protocol': 'natural-delayed-states-v1', 'scope': SCOPE, 'sources': sources,
              'questions': 5 * len(rows), 'original_targets': len(rows), 'candidate_data': str(args.data),
              'candidate_data_id': ready['id'], 'candidate_ready_sha256': sha256(args.data / 'ready.json'),
              'state_source_code_sha256': sha256(__file__), 'delayed_source_code_sha256': sha256(core.__file__),
              'source_code_sha256': sha256(longstream.__file__), 'selection_rule_id': rule['id'],
              'cohorts': {str(d): {'questions': len(rows), 'sources': len(rows)} for d in core.DELAYS},
              'smoke_source_groups': choose_smoke(sources), 'primary_tasks': sorted(TASKS),
              'normal_history_slots': 32, 'normal_recent_slots': 4, 'protected_target_limit': 1,
              'protected_total_step_limit': 16, 'natural_reference': str(args.reference),
              'natural_reference_input_id': core.REFERENCE_ID, 'task_counts': dict(Counter(q['task'] for q in rows))}
    inputs['id'] = digest(inputs)
    ensure_run_manifest(args.out / 'inputs.json', inputs)
    write_json(args.out / 'data-audit.json', {'completed': True, 'input_id': inputs['id'], 'questions': 5 * len(rows),
               'original_targets': len(rows), 'source_videos': len(rows), 'all_targets_have_all5_delays': True,
               'model_outcomes_used_for_eligibility': False, 'new_reserved_source_groups_consumed': 0, 'scope': SCOPE})


def check_code(args):
    inputs = json.loads((args.out / 'inputs.json').read_text())
    if inputs['state_source_code_sha256'] != sha256(__file__) or inputs['candidate_data_id'] != args.data_id:
        raise ValueError('state protocol source changed')
    return inputs


def worker(args):
    check_code(args)
    core.worker(args)


def report(args):
    inputs = check_code(args)
    core.report(args)
    path = args.out / ('smoke-ok.json' if args.smoke else 'report.json')
    result = json.loads(path.read_text())
    selected = [s for s in inputs['sources'] if not args.smoke or s['source_group'] in inputs['smoke_source_groups']]
    for source in selected:
        for q in source['queries']:
            record = json.loads((args.out / 'questions' / f"{digest(q['id'])}.json").read_text())
            if record['memory']['per_policy_unique_retained_steps']['protected_snapshot8'] > 16:
                raise ValueError('one-target protected storage budget exceeded')
    result.update(scope=SCOPE, accuracy_used_as_gate=False, task_counts=inputs['task_counts'],
                  candidate_data_id=inputs['candidate_data_id'])
    write_json(path, result)


def run(args):
    prepare(args)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TRACE_WANDB': '0',
           'TRACE_SWANLAB': '0', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    run_jobs([('gpu-check', [sys.executable, '-c', 'import torch; assert torch.cuda.device_count()==4; '
                            'assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24'], list(range(4)))],
             env, args.out / 'logs')
    command = [sys.executable, '-m', 'trace_vl.core_vlm.longstream_delayed_states', 'worker', '--out', str(args.out),
               '--data-id', args.data_id]
    run_jobs([(f'smoke-{i}', [*command, '--shard', str(i), '--smoke'], [i]) for i in range(4)], env, args.out / 'logs')
    args.smoke = True
    report(args)
    smoke = json.loads((args.out / 'smoke-ok.json').read_text())
    if smoke['questions'] != 10 or smoke['source_videos'] != 2 or smoke['original_targets'] != 2:
        raise ValueError('smoke must cover shortest/longest distinct sources and all delays')
    run_jobs([(f'full-{i}', [*command, '--shard', str(i)], [i]) for i in range(4)], env, args.out / 'logs')
    args.smoke = False
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'worker', 'run', 'report'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('/data/yyy/tracevl-data/streaming-delayed-states-20261125-001'))
    p.add_argument('--data-id', required=True)
    p.add_argument('--reference', type=Path, default=Path('/data/yyy/tracevl-runs/core-longstream-natural-c2f39df-001'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    {'prepare': prepare, 'worker': worker, 'run': run, 'report': report}[args.stage](args)


if __name__ == '__main__':
    main()
