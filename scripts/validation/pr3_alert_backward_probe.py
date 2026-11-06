import argparse
import json
import random
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
from trace_vl.v3 import train as training
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.train import gradient_check

p = argparse.ArgumentParser()
p.add_argument('--state', choices=['on', 'off'], required=True)
p.add_argument('--encode-outside-autocast', action='store_true')
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--sha', default='0171f0782a0a11ba4bb00b44eb3ec006758f6462')
args = p.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=True)
cfg, _ = load_v3_config(args.config)
cfg.alert.state_every = 1 if args.state == 'on' else 1000000000
cfg.alert.state_dropout = 0
random.seed(7)
torch.manual_seed(7)
torch.set_num_threads(4)
try:
    if args.encode_outside_autocast:
        original = training.encode_questions
        def isolated_encode(*args, **kwargs):
            with torch.autocast('cuda', enabled=False):
                return original(*args, **kwargs)
        with patch.object(training, 'encode_questions', isolated_encode):
            result = gradient_check(cfg, steps=6, device='cuda:0')
    else:
        result = gradient_check(cfg, steps=6, device='cuda:0')
except Exception as exc:
    result = {'pass': False, 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
result['tested_commit'] = args.sha
result['state_read_requested'] = args.state
result['encode_outside_autocast'] = args.encode_outside_autocast
suffix = '-isolated-encode' if args.encode_outside_autocast else ''
(root / f'alert-backward-state-{args.state}{suffix}.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'error', 'traceback'}}, indent=2), flush=True)
raise SystemExit(0 if result['pass'] else 1)
