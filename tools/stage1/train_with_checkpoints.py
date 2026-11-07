"""Unchanged upstream trainer, with paired numbered checkpoints and per-rank resource records."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.model import TraceV3Modules
from trace_vl.v3.train import StudentLoss, train

p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--records',required=True)
p.add_argument('--video-root',required=True)
p.add_argument('--output',required=True)
p.add_argument('--clip-seconds',type=float,default=30)
p.add_argument('--max-steps',type=int)
args=p.parse_args()
rank=int(os.environ.get('RANK','0'))
output=Path(args.output).resolve()
if rank==0:
    # Rank 1 may start after rank 0 created it; freshness is owned by rank 0 only.
    output.mkdir(parents=True,exist_ok=False)
cfg,_=load_v3_config(args.config)
saved_module=TraceV3Modules.save
saved_peft=PeftModel.save_pretrained
saved_init=StudentLoss.__init__
current_step=0

def module_save(self,path,extra=None):
    global current_step
    current_step=int((extra or {})['step'])
    saved_module(self,path,extra)
    saved_module(self,output/'checkpoints'/f'step-{current_step:05d}'/'modules.pt',extra)

def peft_save(self,path,*a,**kw):
    result=saved_peft(self,path,*a,**kw)
    target=output/'checkpoints'/f'step-{current_step:05d}'
    saved_peft(self,target/'lora',*a,**kw)
    manifest={'step':current_step,'modules':'modules.pt','adapter':'lora',
              'paired_from_same_optimizer_state':True}
    (target/'checkpoint.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result

def student_init(self,bb,modules,config):
    saved_init(self,bb,modules,config)
    if rank==0:
        target=output/'checkpoints/step-00000'
        saved_module(modules,target/'modules.pt',{'step':0})
        saved_peft(bb.peft,target/'lora')
        (target/'checkpoint.json').write_text(json.dumps({'step':0,'modules':'modules.pt','adapter':'lora',
            'paired_from_same_optimizer_state':True})+'\n')

TraceV3Modules.save=module_save
PeftModel.save_pretrained=peft_save
StudentLoss.__init__=student_init
started=time.perf_counter()
status='failed'
try:
    train(cfg,args.records,args.video_root,str(output),clip_seconds=args.clip_seconds,max_steps=args.max_steps)
    status='completed'
finally:
    local=int(os.environ.get('LOCAL_RANK','0'))
    device=f'cuda:{local}'
    report={'rank':rank,'local_rank':local,'status':status,'world_size':int(os.environ.get('WORLD_SIZE','1')),
            'peak_allocated_gb':torch.cuda.max_memory_allocated(device)/1e9,
            'peak_reserved_gb':torch.cuda.max_memory_reserved(device)/1e9,
            'device_name':torch.cuda.get_device_name(device),'elapsed_including_load':time.perf_counter()-started,
            'records_sha256':hashlib.sha256(Path(args.records).read_bytes()).hexdigest()}
    (output/f'rank-{rank}-resources.json').write_text(json.dumps(report,indent=2)+'\n')
