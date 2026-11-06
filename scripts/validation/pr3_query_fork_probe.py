"""Verify query/marker/generation forks do not close or mutate the live video cache."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from trace_vl.v3.backbone import Backbone
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.session import StreamSession

p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sha',required=True)
p.add_argument('--device',default='cuda:0')
args=p.parse_args()
cfg, _ = load_v3_config(args.config)
torch.set_num_threads(4)
bb = Backbone(cfg.backbone, cfg.stream, args.device)
rng = np.random.default_rng(3)
clips = [rng.integers(0,255,(2,448,448,3),dtype=np.uint8) for _ in range(3)]

def snapshot(session):
    result = {}
    for i, layer in enumerate(session.cache.layers):
        for field in ['keys','values','recurrent_states','conv_states']:
            value = getattr(layer, field, None)
            if value is None:
                continue
            items = list(value.values()) if isinstance(value,dict) else value if isinstance(value,(list,tuple)) else [value]
            for j, item in enumerate(items):
                if torch.is_tensor(item):
                    result[f'{i}.{field}.{j}'] = item.detach().cpu().clone()
    return result

def equal(a,b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[k],b[k]) for k in a)

session = StreamSession(bb,cfg,mode='native')
session.step(clips[0], .25)
before = snapshot(session)
assert len(before) == len(bb.full_layers) * 2 + len(bb.linear_layers) * 2
pos_before = dict(vars(session.pos))
stats_before = session.stats()
session.ask('What is shown?', ['a person','a car'], marker=(clips[0],.25))
assert equal(before,snapshot(session))
assert vars(session.pos) == pos_before
assert session.stats() == stats_before
session.describe('Describe the scene.',max_new_tokens=2)
assert equal(before,snapshot(session))
assert vars(session.pos) == pos_before
for i in [1,2]:
    session.step(clips[i], i+.25)
reference = StreamSession(bb,cfg,mode='native')
for i in range(3):
    reference.step(clips[i],i+.25)
assert equal(snapshot(session),snapshot(reference))
choice_a,scores_a = session.ask('What is shown?',['a person','a car'])
choice_b,scores_b = reference.ask('What is shown?',['a person','a car'])
assert scores_a == scores_b
report = {'tested_commit':args.sha, 'pass':True,
    'marker_query_preserves_all_cache_tensors':True,'describe_preserves_all_cache_tensors':True,
    'stream_after_queries_equals_uninterrupted_stream':True,'answer_scores_exactly_equal':True,
    'tensor_count_checked':len(before),'steps':3,'mode':'native'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
