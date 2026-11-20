#!/usr/bin/env bash
# Prepare and run the 40K matched long-training experiment.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the tested long-scale commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_PREVIOUS="${TRACE_PREVIOUS:-/nas-data/vlm/yyy/tracevl-runs/core-scale20k-b2919d8-001}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-scale40k-${TRACE_REV:0:7}-001}"
export TRACE_TRACKER=swanlab TRACE_SWANLAB=1 TRACE_WANDB=0
export SWANLAB_MODE="${SWANLAB_MODE:-online}" TRACE_SWANLAB_PROJECT="${TRACE_SWANLAB_PROJECT:-trace-vl-core}"
export TRACE_WANDB_GROUP="${TRACE_WANDB_GROUP:-$(basename "$TRACE_OUT")}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
py="${TRACE_PYTHON:-$TRACE_REPO/.venv/bin/python}"
[[ -x "$py" ]]
cd "$TRACE_REPO"
revision=$(git rev-parse --verify "${TRACE_REV}^{commit}")
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.launcher.lock"
flock -n 9 || { echo 'This scale queue is already running' >&2; exit 2; }
trap 'code=$?; if (( code )); then echo "STOPPED ($code). Keep $TRACE_OUT and rerun the same command to resume." >&2; fi' EXIT
if [[ ! -f "$TRACE_OUT/source/.snapshot-complete" ]]; then
  [[ ! -e "$TRACE_OUT/source" ]] || { echo 'Incomplete source snapshot; use a new TRACE_OUT' >&2; exit 2; }
  staging=$(mktemp -d "$TRACE_OUT/source.tmp.XXXXXX")
  git archive "$revision" | tar -xf - -C "$staging"
  printf '%s\n' "$revision" > "$staging/.snapshot-complete"
  mv "$staging" "$TRACE_OUT/source"
fi
[[ "$(cat "$TRACE_OUT/source/.snapshot-complete")" == "$revision" ]]
cd "$TRACE_OUT/source"
export PYTHONPATH="$PWD/src" TMPDIR="$TRACE_OUT/tmp"
mkdir -p "$TMPDIR"
"$py" -c 'import torch, transformers, peft, swanlab; print("Environment imports OK; SwanLab enabled")'
export CUDA_VISIBLE_DEVICES=''
# Regression tests select their own mock/offline backend, never the operator account.
TRACE_TRACKER=wandb TRACE_SWANLAB=0 TRACE_WANDB=0 WANDB_MODE=offline \
  "$py" -m pytest -q tests/test_core_vlm_*.py tests/test_v3_run_io.py
export CUDA_VISIBLE_DEVICES=0,1,2,3
args=(--previous "$TRACE_PREVIOUS" --out "$TRACE_OUT")
if [[ "${TRACE_PREPARE_ONLY:-0}" == 1 ]]; then
  "$py" -m trace_vl.core_vlm.long_scale "${args[@]}" --prepare-only
  exit 0
fi
"$py" -m trace_vl.core_vlm.long_scale "${args[@]}"
"$py" - <<'PY'
import os, tarfile
from pathlib import Path
out = Path(os.environ['TRACE_OUT'])
files = list(out.glob('*.json')) + list(out.glob('*.md'))
directories = ['data', 'logs', 'train', 'eval', 'smoke-train', 'smoke-eval', 'batch', 'mixture']
directories.extend(p.name for p in out.glob('eval-step-*') if p.is_dir())
for name in directories:
    for p in (out / name).rglob('*'):
        if (p.is_file() and p.suffix in {'.json', '.jsonl', '.log', '.md'}
                and not any(x.startswith(('checkpoint-', '.')) or x == 'wandb' for x in p.relative_to(out).parts)
                and (p.suffix != '.jsonl' or p.name in {'metrics.jsonl', 'predictions.jsonl'})):
            files.append(p)
target = out / 'reports.tar.gz'
with tarfile.open(out / 'reports.tmp', 'w:gz') as tar:
    for p in sorted(set(files)):
        tar.add(p, arcname=str(p.relative_to(out)))
    for p in sorted(out.rglob('offline-run-*')):
        if p.is_dir():
            tar.add(p, arcname=str(p.relative_to(out)))
(out / 'reports.tmp').replace(target)
print('REPORT BUNDLE:', target)
PY
upload_args=(--runs "$TRACE_OUT")
[[ -z "${TRACE_OPERATOR_LOG:-}" ]] || upload_args+=(--operator-log "$TRACE_OPERATOR_LOG")
"$py" -m trace_vl.core_vlm.swanlab_upload "${upload_args[@]}"
