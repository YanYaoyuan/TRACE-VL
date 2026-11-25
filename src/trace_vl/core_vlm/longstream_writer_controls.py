"""Same77-source frozen-reader writer ablation, with exact shared-baseline replay."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from trace_vl.v3.run_io import ensure_run_manifest

from . import longstream
from . import longstream_delayed as core
from . import persistent_writers as writers
from .control import checkpoint_identity
from .data import sha256
from .experiment import run_jobs
from .prepare import digest, write_json
from .train import checked_checkpoint

REFERENCE_ID = '090db3b555dd368f8d7478a2de7195408a4aa6788e451e02636fc4d91cc59e4b'
REFERENCE_REPORT_SHA256 = 'f2cd9ec0c4c401cca7af35c1093dfa453dfb1700c2b7688dd668ece50da6dd88'
REFERENCE_ARCHIVE_SHA256 = '98459c2f386e734917529bf9ace54a2f14d96c825f6aaf94433540370acaf439'
EXTRA_ARMS = ('time_coverage8', 'spatial_change8', 'reservoir_seed29_time8', 'reservoir_seed43_time8')
ARMS = (*core.ARMS, *EXTRA_ARMS)
SCOPE = (
    'Development writer ablation on all77 frozen state targets/385queries; no outcome filtering, '
    'same natural0/15/60/180/300s, weights, sampling and causal time reader. Normal writers32history+'
    '4recent/128readtokens. Time coverage and spatial-change value retention, plus two additional '
    'prespecified reservoir memory seeds29/43 and exact replay of all6 prior arms. One stream '
    'encoder pass/source; writer receives observations only. No training/new reserved sources/'
    'query-time vision/LM KV. Original snapshot privileged and current8 a later-frame control. '
    'Explicit per-policy state, bounded transient workspace, wall time and write-only audit traces. '
    'Memory seeds are not training seeds. Not official benchmark/unseen validation or architecture promotion.'
)


class ReservoirWriter(writers.PersistentWriter):
    def __init__(self, seed):
        super().__init__('time_coverage', 32)
        self.policy = 'reservoir'
        self.seed = seed
        self.rng = random.Random(seed)

    def _admit(self, step):
        if len(self.history) < self.capacity:
            return [*self.history, step]
        index = self.rng.randrange(self.old_seen+1)
        result = list(self.history)
        if index < self.capacity:
            result[index] = step
        return result

    def audit(self):
        return {**super().audit(), 'memory_seed': self.seed}


def extra_seed(group, label):
    if label not in (29, 43):
        raise ValueError('only prespecified extra memory seed labels allowed')
    return int(digest(['writer-controls-extra-reservoir-v1', group, label])[:8], 16)


class WriterExtension:
    arms = EXTRA_ARMS

    def __init__(self, group, out):
        self.path = out/'writer-traces'/f'{digest(group)}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w', buffering=1)
        self.trace_events = 0
        self.trace_digest = hashlib.sha256()
        self.seconds = {arm: 0. for arm in self.arms}
        self.banks = {
            'time_coverage8': writers.PersistentWriter('time_coverage', on_eviction=self.trace),
            'spatial_change8': writers.PersistentWriter('spatial_change', on_eviction=self.trace),
            'reservoir_seed29_time8': ReservoirWriter(extra_seed(group, 29)),
            'reservoir_seed43_time8': ReservoirWriter(extra_seed(group, 43)),
        }
        self.tokens_per_step = None

    def trace(self, event):
        raw = (json.dumps(event, separators=(',', ':'), allow_nan=False)+'\n').encode()
        self.file.write(raw.decode())
        self.trace_digest.update(raw)
        self.trace_events += 1

    def append(self, step):
        self.tokens_per_step = step.tokens.numel()
        for arm, bank in self.banks.items():
            start = time.perf_counter()
            bank.append(step)
            self.seconds[arm] += time.perf_counter()-start

    def read(self, arm, target):
        return self.banks[arm].read(target)

    def audit(self):
        states = {arm: bank.audit() for arm, bank in self.banks.items()}
        for state in states.values():
            if state['history_steps'] > 32 or state['recent_steps'] > 4:
                raise ValueError('new writer storage capacity exceeded')
        return {'banks': states, 'writer_append_wall_seconds': dict(self.seconds),
                'trace_events_written': self.trace_events, 'trace_sha256_so_far': self.trace_digest.hexdigest(),
                'conservative_spatial_cpu_fp32_workspace_bound_bytes': 8*33*(self.tokens_per_step or 0)*4,
                'accounting': 'Per-policy owned feature values,32+4slots and96logicalB/slot; no persistent descriptors. Transient CPU tensor expression bound is conservative, not measured allocator/RSS. Append wall time includes device synchronization. Audit log write-only during inference and never a memory/read tier; returned values are original stored16tokens.'}

    def close(self):
        self.file.close()

    def receipt(self):
        if not self.file.closed:
            raise ValueError('writer trace must close before final receipt')
        return {'path': str(self.path), 'bytes': self.path.stat().st_size, 'sha256': sha256(self.path),
                'events': self.trace_events, 'scope': 'Admission indices/times/computed scalar costs only, no feature/pixel archive. Reader never accesses trace. Semantic costs logged by inference; independent re-encoding needed to verify their feature origin.'}


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.reference/'inputs.json').read_text())
    if (base['id'] != REFERENCE_ID or digest({k:v for k,v in base.items() if k!='id'}) != REFERENCE_ID
            or sha256(args.reference/'report.json') != REFERENCE_REPORT_SHA256
            or sha256(args.reference/'reports.tar.gz') != REFERENCE_ARCHIVE_SHA256):
        raise ValueError('completed77-source baseline reference changed')
    report = json.loads((args.reference/'report.json').read_text())
    if not report['completed'] or report['questions'] != 385 or report['source_videos'] != 77:
        raise ValueError('completed all77 state reference required')
    if checked_checkpoint(Path(base['plain']), base['checkpoint']['run']) != base['checkpoint']:
        raise ValueError('frozen checkpoint changed')
    if checkpoint_identity(Path(base['model'])) != base['model_identity']:
        raise ValueError('base model changed')
    if len(base['sources']) != 77 or any(len(s['queries']) != 5 for s in base['sources']):
        raise ValueError('all77 targets/all5delays must survive')
    for source in base['sources']:
        if sha256(source['video']) != source['video_sha256'] or sha256(source['metadata']) != source['metadata_sha256']:
            raise ValueError('frozen source bytes/PTS changed')
    inputs = copy.deepcopy(base)
    inputs.pop('id')
    inputs.pop('state_source_code_sha256')
    inputs.update(protocol='natural-delayed-writer-controls-v1', scope=SCOPE,
                  reference=str(args.reference), reference_input_id=REFERENCE_ID,
                  reference_report_sha256=REFERENCE_REPORT_SHA256,
                  writer_protocol_source_sha256=sha256(__file__), writer_class_source_sha256=sha256(writers.__file__),
                  delayed_source_code_sha256=sha256(core.__file__), source_code_sha256=sha256(longstream.__file__),
                  arms=list(ARMS), extra_memory_seed_labels=[29,43], normal_history_slots=32, normal_recent_slots=4,
                  analysis_plan={'primary_endpoint_delay':300, 'primary_contrasts':'each new deterministic writer minus per-source mean of all3 reservoir memory seeds',
                                 'bootstrap_unit':'77source videos; seeds averaged within source, not231independent videos',
                                 'draws':4000, 'seed':7, 'secondary':'all5delay curves, seed-wise outcomes, old-original reservoir exact replay',
                                 'accuracy_smoke_gate':False, 'architecture_promotion':False})
    inputs['id'] = digest(inputs)
    ensure_run_manifest(args.out/'inputs.json', inputs)
    write_json(args.out/'data-audit.json', {'completed':True, 'input_id':inputs['id'], 'questions':385,
               'source_videos':77, 'original_targets':77, 'new_reserved_sources':0,
               'all_original_questions_options_gold_delays_unchanged':True, 'no_outcome_filtering':True, 'scope':SCOPE})


def check_code(args):
    inputs = json.loads((args.out/'inputs.json').read_text())
    if (inputs['writer_protocol_source_sha256'] != sha256(__file__)
            or inputs['writer_class_source_sha256'] != sha256(writers.__file__)):
        raise ValueError('writer experiment code changed')
    return inputs


def worker(args):
    check_code(args)
    core.worker(args, extension_factory=lambda s,out: WriterExtension(s['source_group'], out))


def check_new_reads(record, query, bytes_per_step=81920):
    states = record['memory']['writer_controls']['banks']
    if set(states) != set(EXTRA_ARMS):
        raise ValueError('missing new writer state')
    for arm, state in states.items():
        if (state['seen'] != record['stream_encoder_calls'] or state['old_seen'] != max(0,state['seen']-4)
                or state['capacity'] != 32 or state['history_steps'] != min(32,state['old_seen'])
                or state['recent_steps'] != min(4,state['seen'])
                or state['persistent_auxiliary_feature_bytes'] != 0):
            raise ValueError('new writer counters or capacity changed')
        steps = [s for key in ('history','recent') for s in state['retained'][key] if s['times'][-1] <= query['target_time']]
        selected = sorted(steps,key=lambda s:s['times'][-1])[-8:]
        p = record['predictions'][arm]
        if (p['read_frame_indices'] != [i for s in selected for i in s['indices']]
                or p['read_times'] != [s['times'] for s in selected]
                or p['real_feature_rgb_hashes'] != [s['rgb_sha256'] for s in selected]):
            raise ValueError('new writer read does not match actual stored values')
        slots = state['history_steps']+state['recent_steps']
        if (state['feature_payload_bytes'] != slots*bytes_per_step
                or state['feature_and_fixed_slot_metadata_bytes'] != slots*(bytes_per_step+96)
                or p['empty_visual_memory'] != (not selected)):
            raise ValueError('new writer logical payload or empty-read accounting changed')


def report(args):
    inputs = check_code(args)
    pending = args.out/'writer-report.pending.json'
    core.report(args, arms=ARMS, output_path=pending)
    path = args.out/('smoke-ok.json' if args.smoke else 'report.json')
    result = json.loads(pending.read_text())
    selected = [s for s in inputs['sources'] if not args.smoke or s['source_group'] in inputs['smoke_source_groups']]
    replays = 0
    maxima = {arm:0 for arm in EXTRA_ARMS}
    seconds = {arm:0. for arm in EXTRA_ARMS}
    for source in selected:
        done = json.loads((args.out/'sources'/f"{digest(source['source_group'])}.json").read_text())
        trace = done['extension_trace']
        if sha256(trace['path']) != trace['sha256']:
            raise ValueError('write-only admission trace changed')
        for arm, value in done['final_memory']['writer_controls']['writer_append_wall_seconds'].items():
            seconds[arm] += value
        for q in source['queries']:
            r = json.loads((args.out/'questions'/f"{digest(q['id'])}.json").read_text())
            old = json.loads((args.reference/'questions'/f"{digest(q['id'])}.json").read_text())
            if {a:r['predictions'][a] for a in core.ARMS} != old['predictions']:
                raise ValueError('shared six-arm original predictions did not exactly replay')
            check_new_reads(r,q)
            for arm,state in r['memory']['writer_controls']['banks'].items():
                maxima[arm] = max(maxima[arm],state['feature_and_fixed_slot_metadata_bytes'])
            replays += 1
    result.update(scope=SCOPE, exact_baseline_query_replays=replays, exact_baseline_arm_predictions=6*replays,
                  normal_history_slots=32, normal_recent_slots=4, primary_score=inputs['primary_score'],
                  maximum_new_writer_logical_payload_bytes=maxima, writer_append_wall_seconds=seconds,
                  accuracy_used_as_gate=False, new_reserved_sources=0, memory_seed_labels=[29,43],
                  memory_seed_scope='3 reservoir memory seeds total incl exact original, one frozen training/model seed; deterministic new writers once.')
    write_json(path,result)


def run(args):
    prepare(args)
    env = {**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TRACE_WANDB':'0','TRACE_SWANLAB':'0',
           'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    run_jobs([('gpu-check',[sys.executable,'-c','import torch; assert torch.cuda.device_count()==4; '
                          'assert min(torch.cuda.mem_get_info(i)[0]/2**30 for i in range(4))>24'],list(range(4)))],env,args.out/'logs')
    command = [sys.executable,'-m','trace_vl.core_vlm.longstream_writer_controls','worker','--out',str(args.out),'--reference',str(args.reference)]
    run_jobs([(f'smoke-{i}',[*command,'--shard',str(i),'--smoke'],[i]) for i in range(4)],env,args.out/'logs')
    args.smoke = True
    report(args)
    smoke = json.loads((args.out/'smoke-ok.json').read_text())
    if smoke['questions']!=10 or smoke['source_videos']!=2 or smoke['exact_baseline_query_replays']!=10:
        raise ValueError('two-source all-delay exact replay smoke incomplete')
    run_jobs([(f'full-{i}',[*command,'--shard',str(i)],[i]) for i in range(4)],env,args.out/'logs')
    args.smoke = False
    report(args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','worker','run','report'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('/data/yyy/tracevl-runs/core-longstream-states-3dc8c83-001'))
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    {'prepare':prepare,'worker':worker,'run':run,'report':report}[args.stage](args)


if __name__=='__main__':
    main()
