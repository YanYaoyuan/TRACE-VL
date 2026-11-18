#!/usr/bin/env bash
# Default: CPU data preparation only. GPU queue requires explicit reviewed recipe.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the reviewed scale commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-scale20k-${TRACE_REV:0:7}-001}"
export TRACE_DATA_WORKERS="${TRACE_DATA_WORKERS:-16}"
export TRACE_ARCHIVE_WORKERS="${TRACE_ARCHIVE_WORKERS:-4}"
export TRACE_SCALE_MODE="${TRACE_SCALE_MODE:-prepare}"
[[ "$TRACE_SCALE_MODE" == prepare || "$TRACE_SCALE_MODE" == train ]] || exit 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRACE_WANDB="${TRACE_WANDB:-1}" WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-trace-vl-core}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
py="${TRACE_PYTHON:-$TRACE_REPO/.venv/bin/python}"
[[ -x "$py" ]] || { echo "Missing UV Python: $py" >&2; exit 2; }
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
cd "$TRACE_REPO"
revision=$(git rev-parse --verify "${TRACE_REV}^{commit}")
export TRACE_REV="$revision"
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.launcher.lock"
flock -n 9 || { echo 'This scale preparation/queue is already running' >&2; exit 2; }
trap 'code=$?; if (( code )); then echo "STOPPED ($code). Keep $TRACE_OUT; no automatic fallback or sample replacement." >&2; fi' EXIT
if [[ ! -f "$TRACE_OUT/source/.snapshot-complete" ]]; then
  [[ ! -e "$TRACE_OUT/source" ]] || { echo 'Incomplete source snapshot: use a new TRACE_OUT' >&2; exit 2; }
  staging=$(mktemp -d "$TRACE_OUT/source.tmp.XXXXXX")
  git archive "$revision" | tar -xf - -C "$staging"
  printf '%s\n' "$revision" > "$staging/.snapshot-complete"
  mv "$staging" "$TRACE_OUT/source"
fi
[[ "$(cat "$TRACE_OUT/source/.snapshot-complete")" == "$revision" ]]
cd "$TRACE_OUT/source"
export PYTHONPATH="$PWD/src" TMPDIR="$TRACE_OUT/tmp"
mkdir -p "$TMPDIR"
# No CUDA contexts while preparing beside the currently running experiment.
export CUDA_VISIBLE_DEVICES=''
"$py" -m pytest -q tests/test_core_vlm_*.py tests/test_v3_run_io.py
"$py" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.core_vlm.prepare import write_json
from trace_vl.v3.run_io import ensure_run_manifest
out=Path(os.environ['TRACE_OUT'])
cfg=json.loads(Path('configs/core_vlm/scale20k_a100.json').read_text())
cfg.update(workers=int(os.environ['TRACE_DATA_WORKERS']),archive_workers=int(os.environ['TRACE_ARCHIVE_WORKERS']))
for var,key in [('TRACE_MODEL','model'),('TRACE_EXTERNAL_DATA','external_data'),('TRACE_PILOT_RUN','pilot_run')]:
    if os.environ.get(var): cfg[key]=os.environ[var]
if os.environ.get('TRACE_REPAIR_FROM') or os.environ.get('TRACE_EXCLUDE_TRAINING_VIDEO'):
    cfg['repair_from']=os.environ['TRACE_REPAIR_FROM']
    cfg['exclude_training_videos']=[os.environ['TRACE_EXCLUDE_TRAINING_VIDEO']]
ensure_run_manifest(out/'launcher.json',{'id':os.environ['TRACE_REV'],'config':cfg})
write_json(out/'config.json',cfg)
PY
args=(--config "$TRACE_OUT/config.json" --out "$TRACE_OUT" --mode "$TRACE_SCALE_MODE")
if [[ "$TRACE_SCALE_MODE" == train ]]; then
  : "${TRACE_RECIPE:?Set ce or preserved after reviewing pilot results}"
  : "${TRACE_PILOT_REPORT:?Set the completed pilot report.json path}"
  : "${TRACE_PILOT_SHA256:?Set the reviewed report SHA256}"
  : "${TRACE_REVIEW_NOTE:?Record the reason for selecting this recipe}"
  args+=(--recipe "$TRACE_RECIPE" --pilot-report "$TRACE_PILOT_REPORT" --pilot-sha256 "$TRACE_PILOT_SHA256" --review-note "$TRACE_REVIEW_NOTE")
fi
"$py" -m trace_vl.core_vlm.scale "${args[@]}"
"$py" - <<'PY'
import os, tarfile
from pathlib import Path
out=Path(os.environ['TRACE_OUT'])
# Keep media, model weights, archive indexes and large per-frame verification out.
roots=[out/('training-'+os.environ.get('TRACE_RECIPE',''))] if os.environ['TRACE_SCALE_MODE']=='train' else []
files=[out/'queue-plan.json',out/'config.json',out/'launcher.json',out/'preparation.json',out/'release.json',out/'capacity-preflight.json',out/'repair.json']
files += [out/'mixture'/name for name in ('inventory.json','inputs.json','complete.json','annotation-audit.json','capacity.json','media-exclusion.json')]
files += [out/'general'/name for name in ('report.json','missing.json','protected-exclusions.json')]
for root in roots:
    files.extend(p for p in root.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.log','.md'}
                 and not any(x.startswith('checkpoint-') for x in p.relative_to(root).parts))
target=out/('scale-'+os.environ['TRACE_SCALE_MODE']+'-reports.tar.gz')
with tarfile.open(target.with_suffix('.tmp'),'w:gz') as tar:
    for path in sorted(set(files)):
        if path.is_file(): tar.add(path,arcname=str(path.relative_to(out)))
target.with_suffix('.tmp').replace(target)
print('REPORT BUNDLE:',target)
PY
