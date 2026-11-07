import argparse
from pathlib import Path

import yaml
from trace_vl.v3.config import load_v3_config

p=argparse.ArgumentParser(description='Write a strict resolved Stage-1 config without changing model dimensions.')
p.add_argument('--base',default='configs/v3/stage1.yaml')
p.add_argument('--model',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=200)
p.add_argument('--accum',type=int,default=1)
p.add_argument('--warmup',type=int,default=20)
p.add_argument('--save-every',type=int,default=50)
p.add_argument('--max-eval-steps',type=int,default=0)
args=p.parse_args()
assert args.steps>0 and args.accum>0 and args.save_every>0
cfg,_=load_v3_config(args.base)
cfg.backbone.name=str(args.model.resolve(strict=True))
cfg.backbone.local_files_only=True
cfg.train.steps=args.steps
cfg.train.grad_accum=args.accum
cfg.train.warmup=args.warmup
cfg.train.save_every=args.save_every
cfg.memory.policy='full'
cfg.stream.max_steps=args.max_eval_steps
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
print(args.output)
