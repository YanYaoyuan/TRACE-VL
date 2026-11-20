#!/usr/bin/env bash
# One entry point for gate statistics, budget calibration, validated evaluation and paired intervals.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: TRACE_RUNS=... TRACE_DATA_DIR=... TRACE_VIDEO_ROOT=... bash scripts/evaluate_a100_pilot.sh [--check-only]'
  echo 'Optional: TRACE_EVAL_GPU=0 TRACE_EVAL_STEP=100 TRACE_EVAL_OUTPUT=...'
  exit 0
fi
check_only=0
if [[ "${1:-}" == --check-only ]]; then check_only=1; shift; fi
[[ $# == 0 ]] || { echo 'Unexpected argument; use --help' >&2; exit 2; }
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_RUNS:?Set TRACE_RUNS to the directory containing pilot-transition and pilot-b4}"
: "${TRACE_DATA_DIR:?Set TRACE_DATA_DIR to the directory containing train.jsonl and val.jsonl}"
: "${TRACE_VIDEO_ROOT:?Set TRACE_VIDEO_ROOT to the extracted-video root}"
export TRACE_EVAL_OUTPUT="${TRACE_EVAL_OUTPUT:-$TRACE_RUNS/pilot-eval}"
export TRACE_EVAL_STEP="${TRACE_EVAL_STEP:-100}"
export TRACE_TRANSITION_CONFIG="${TRACE_TRANSITION_CONFIG:-$root/configs/v3/a100-pilot-transition.yaml}"
export TRACE_B4_CONFIG="${TRACE_B4_CONFIG:-$root/configs/v3/a100-pilot-b4.yaml}"
export CUDA_VISIBLE_DEVICES="${TRACE_EVAL_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRACE_PIPELINE_CHECK_ONLY="$check_only"
runner=("${TRACE_UV_BIN:-uv}" run --locked --no-sync python)
if [[ $check_only == 0 ]]; then
  mkdir -p "$TRACE_EVAL_OUTPUT"
  exec 9>"$TRACE_EVAL_OUTPUT/.pipeline.lock"
  flock -n 9 || { echo 'Another evaluation is using this output directory' >&2; exit 2; }
  trap 'status=$?; if ((status)); then echo "Evaluation failed (exit $status). Existing predictions are preserved for a retry." >&2; fi' EXIT
fi

"${runner[@]}" - <<'PY'
import hashlib, json, math, os
from pathlib import Path
import torch
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.harness import run_fingerprint

runs, data, videos, out = [Path(os.environ[k]).resolve() for k in ('TRACE_RUNS','TRACE_DATA_DIR','TRACE_VIDEO_ROOT','TRACE_EVAL_OUTPUT')]
step = int(os.environ['TRACE_EVAL_STEP'])
if step < 1: raise ValueError('TRACE_EVAL_STEP must be positive')
def read(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
train, val = read(data/'train.jsonl'), read(data/'val.jsonl')
for split, rows in [('train',train),('val',val)]:
    if not rows or any(r.get('split') != split for r in rows): raise ValueError(f'Invalid {split} records')
    if len({r['id'] for r in rows}) != len(rows): raise ValueError(f'Duplicate {split} question ids')
    if any(not (videos/r['video']).is_file() for r in rows): raise ValueError(f'Missing {split} videos; check TRACE_VIDEO_ROOT')
if {r['video'] for r in train} & {r['video'] for r in val}: raise ValueError('Training/validation video overlap')
train_sha = hashlib.sha256((data/'train.jsonl').read_bytes()).hexdigest()
cfgs, ckpts, adapters, stats, worlds = {}, {}, {}, {}, set()
for arm, env in [('transition','TRACE_TRANSITION_CONFIG'),('b4','TRACE_B4_CONFIG')]:
    cfg, _ = load_v3_config(os.environ[env]); cfgs[arm] = cfg
    run = runs/f'pilot-{arm}'
    meta = json.loads((run/'run.json').read_text())
    if meta['world'] < 1: raise ValueError('Invalid training world size')
    worlds.add(meta['world'])
    if meta['config'] != cfg.to_dict(): raise ValueError(f'{arm}: config changed since training')
    for rank in range(meta['world']):
        resource = json.loads((run/f'rank-{rank}-resources.json').read_text())
        if resource['status'] != 'completed' or resource['records_sha256'] != train_sha:
            raise ValueError(f'{arm}/rank{rank}: training incomplete or training manifest changed')
    target = run/'checkpoints'/f'step-{step:05d}'
    pair = json.loads((target/'checkpoint.json').read_text())
    if pair['step'] != step or not pair['paired_from_same_optimizer_state']: raise ValueError('Unpaired checkpoint')
    ckpts[arm], adapters[arm] = str(target/'modules.pt'), str(target/'lora')
    if not Path(ckpts[arm]).is_file() or not (target/'lora/adapter_config.json').is_file(): raise ValueError('Missing checkpoint or adapter')
    if not list((target/'lora').glob('*.safetensors')): raise ValueError('Missing LoRA weights')
    rows = read(run/'metrics.jsonl')
    if not rows or max(r['step'] for r in rows) < step: raise ValueError('Training metrics do not cover checkpoint')
    rows = [r for r in rows if r['step'] <= step]
    if len(rows)!=step or {r['step'] for r in rows}!=set(range(1,step+1)):
        raise ValueError('Missing or duplicate training metric steps')
    vals = [r['p_step_tokens_mean'] for r in rows if r.get('p_step_tokens_mean') is not None]
    gist = cfg.transition.gist
    if any(not math.isfinite(v) or v<gist for v in vals): raise ValueError('Invalid P-token statistics')
    stats[arm] = {'logged_updates':len(rows), 'records_with_P_steps':len(vals),
        'gist_only_records':sum(abs(v-gist)<1e-6 for v in vals),
        'records_with_transition_tokens':sum(v>gist+1e-6 for v in vals),
        'P_mean_min':min(vals) if vals else None, 'P_mean_max':max(vals) if vals else None}
if cfgs['transition'].transition.p_step_mode != 'transition' or cfgs['b4'].transition.p_step_mode != 'pooled':
    raise ValueError('Expected transition and pooled B4 configurations')
a,b=cfgs['transition'].to_dict(),cfgs['b4'].to_dict()
b['transition']['p_step_mode']='transition'
if a!=b or len(worlds)!=1: raise ValueError('Paired configurations/world sizes differ beyond P-step mode')
identities = {}
for arm in ['native','transition','b4']:
    cfg = cfgs['b4' if arm == 'b4' else 'transition']
    identities[arm] = run_fingerprint(cfg,str(data/'val.jsonl'),'native' if arm=='native' else 'v3',
        ckpts.get(arm),adapters.get(arm),str(videos),val)
    previous = out/arm/'run.json'
    if previous.exists() and json.loads(previous.read_text()).get('id') != identities[arm]['id']:
        raise ValueError(f'{arm}: different existing evaluation; set a NEW TRACE_EVAL_OUTPUT')
code = hashlib.sha256()
for path in sorted(Path('src/trace_vl/v3').glob('*.py')): code.update(path.name.encode()+path.read_bytes())
workflow = {'step':step,'eval_runs':identities,'train_sha256':train_sha,'source_sha256':code.hexdigest()}
previous = out/'workflow.json'
if previous.exists() and json.loads(previous.read_text()) != workflow:
    raise ValueError('Workflow inputs changed; set a NEW TRACE_EVAL_OUTPUT')
if os.environ['TRACE_PIPELINE_CHECK_ONLY']=='1':
    print(json.dumps({'inputs_ok':True,'validation_questions':len(val),'gate_summary':stats},indent=2))
else:
    if not torch.cuda.is_available(): raise RuntimeError('Selected CUDA device is unavailable')
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').unlink(missing_ok=True)
    (out/'workflow.json').write_text(json.dumps(workflow,indent=2)+'\n')
    (out/'pilot-gate-summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    print('Preflight passed:',len(val),'validation questions; gate statistics are rank-0 logged samples only')
PY
[[ $check_only == 0 ]] || exit 0

step_dir="$(printf 'step-%05d' "$TRACE_EVAL_STEP")"
budget_tmp="$(mktemp "$TRACE_EVAL_OUTPUT/.budget.XXXXXX")"
echo '[1/5] Fixed training-video budget calibration'
"${runner[@]}" -m trace_vl v3 budget \
  --config "$TRACE_TRANSITION_CONFIG" --config-b4 "$TRACE_B4_CONFIG" \
  --checkpoint "$TRACE_RUNS/pilot-transition/checkpoints/$step_dir/modules.pt" \
  --checkpoint-b4 "$TRACE_RUNS/pilot-b4/checkpoints/$step_dir/modules.pt" \
  --records "$TRACE_DATA_DIR/train.jsonl" --video-root "$TRACE_VIDEO_ROOT" \
  --max-videos 8 --max-seconds 30 --device cuda:0 > "$budget_tmp"
"${runner[@]}" - "$budget_tmp" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if len(report.get('arms',{})) != 2 or any(a['steps']<=0 or a['total_visual_tokens']<=0 for a in report['arms'].values()):
    raise ValueError('Empty/invalid budget audit')
if not isinstance(report.get('matched'),bool): raise ValueError('Missing budget gate')
PY
mv "$budget_tmp" "$TRACE_EVAL_OUTPUT/budget.json"

validate_and_score() {
  "${runner[@]}" - "$1" "$TRACE_DATA_DIR/val.jsonl" "$TRACE_EVAL_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
from trace_vl.v3.harness import score
arm, records, root = sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3])
out=root/arm
gold={r['id']:r for r in (json.loads(s) for s in records.read_text().splitlines() if s.strip())}
errors=list(out.glob('errors.*.json'))
if not errors or any(json.loads(p.read_text()) for p in errors): raise ValueError('Evaluation not complete, or errors present')
rows=[json.loads(s) for p in out.glob('predictions.*.jsonl') for s in p.read_text().splitlines() if s.strip()]
if len(rows)!=len(gold) or len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!=set(gold):
    raise ValueError('Incomplete, duplicate or unexpected predictions')
expected=json.loads((root/'workflow.json').read_text())['eval_runs'][arm]['id']
if any(r.get('run')!=expected or r['answer']!=gold[r['id']]['answer'] for r in rows): raise ValueError('Wrong run identity or gold labels')
if any(r['video_key']!=(gold[r['id']].get('stream') or gold[r['id']]['video']) for r in rows): raise ValueError('Wrong video grouping')
report=score(str(out),str(out/'score.json'))
if report['n']!=len(gold): raise ValueError('Invalid score coverage')
print(json.dumps({'arm':arm,**report},indent=2))
PY
}
index=1
for arm in native transition b4; do
  index=$((index+1))
  echo "[$index/5] Evaluate $arm (existing valid predictions can be resumed)"
  cfg="$TRACE_TRANSITION_CONFIG"
  eval_args=(--mode native)
  if [[ "$arm" != native ]]; then
    [[ "$arm" != b4 ]] || cfg="$TRACE_B4_CONFIG"
    eval_args=(--mode v3 --checkpoint "$TRACE_RUNS/pilot-$arm/checkpoints/$step_dir/modules.pt" \
      --adapter "$TRACE_RUNS/pilot-$arm/checkpoints/$step_dir/lora")
  fi
  "${runner[@]}" -m trace_vl v3 eval --config "$cfg" --records "$TRACE_DATA_DIR/val.jsonl" \
    --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_EVAL_OUTPUT/$arm" --device cuda:0 "${eval_args[@]}"
  validate_and_score "$arm"
done
echo '[5/5] Summarize results and paired video-bootstrap intervals'
"${runner[@]}" - <<'PY'
import json,os,random
from pathlib import Path
root=Path(os.environ['TRACE_EVAL_OUTPUT'])
preds={arm:{r['id']:r for p in (root/arm).glob('predictions.*.jsonl') for r in (json.loads(s) for s in p.read_text().splitlines() if s.strip())}
       for arm in ['native','transition','b4']}
def paired(a,b):
    groups={}
    for key,r in preds[a].items():
        other=preds[b][key]
        group=groups.setdefault(r['video_key'],[0,0])
        group[0]+=int(r['prediction']==r['answer'])-int(other['prediction']==other['answer'])
        group[1]+=1
    units=list(groups.values()); rng=random.Random(7); draws=[]
    point=100*sum(v[0] for v in units)/sum(v[1] for v in units)
    if len(units)<2: return {'delta_accuracy_pp':point,'ci95_pp':None,'videos':len(units),'bootstrap_samples':0,'seed':7}
    for _ in range(4000):
        chosen=rng.choices(units,k=len(units))
        draws.append(100*sum(v[0] for v in chosen)/sum(v[1] for v in chosen))
    draws.sort()
    return {'delta_accuracy_pp':point,
            'ci95_pp':[draws[100],draws[3899]],'videos':len(units),'bootstrap_samples':4000,'seed':7}
budget=json.loads((root/'budget.json').read_text())
summary={'completed':True,'budget_matched':budget['matched'],'budget_gap':budget['budget_gap'],
         'scores':{a:json.loads((root/a/'score.json').read_text()) for a in preds},
         'paired':{'transition_minus_b4':paired('transition','b4'),'transition_minus_native':paired('transition','native')},
         'gate_summary':json.loads((root/'pilot-gate-summary.json').read_text()),
         'note':'Gate statistics describe rank-0 logged samples. Budget mismatch confounds an equal-budget architecture claim.'}
temporary=root/'.summary.json.tmp'
temporary.write_text(json.dumps(summary,indent=2)+'\n'); temporary.replace(root/'summary.json')
print(json.dumps(summary,indent=2))
PY
echo "Completed. Report: $TRACE_EVAL_OUTPUT/summary.json"
