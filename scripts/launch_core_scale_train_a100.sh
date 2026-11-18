#!/usr/bin/env bash
# Reuse completed scale data and run both matched batch-4 architecture arms.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the tested scale-training commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_PREPARED="${TRACE_PREPARED:-/nas-data/vlm/yyy/tracevl-runs/core-scale20k-b2919d8-001}"
export TRACE_BENCHMARK="${TRACE_BENCHMARK:-/nas-data/vlm/yyy/tracevl-runs/core-batch-43fdd72-001}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-scale20k-mb4-${TRACE_REV:0:7}-001}"
export TRACE_WANDB="${TRACE_WANDB:-1}" WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-trace-vl-core}"
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
"$py" -c 'import torch, transformers, peft, wandb; print("Environment imports OK; W&B offline")'
export CUDA_VISIBLE_DEVICES=''
"$py" -m pytest -q tests/test_core_vlm_*.py tests/test_v3_run_io.py
export CUDA_VISIBLE_DEVICES=0,1,2,3
args=(--prepared "$TRACE_PREPARED" --benchmark "$TRACE_BENCHMARK" --out "$TRACE_OUT")
[[ -z "${TRACE_REVIEWED_PAIRS:-}" ]] || args+=(--reviewed-pairs "$TRACE_REVIEWED_PAIRS")
[[ "${TRACE_PREPARE_ONLY:-0}" != 1 ]] || args+=(--prepare-only)
"$py" -m trace_vl.core_vlm.scale_train "${args[@]}"
"$py" - <<'PY'
import os, tarfile
from pathlib import Path
out = Path(os.environ['TRACE_OUT'])
files = list(out.glob('*.json')) + list(out.glob('*.md'))
for name in ('data', 'logs', 'train', 'eval', 'smoke-train', 'smoke-eval'):
    for p in (out / name).rglob('*'):
        if (p.is_file() and p.suffix in {'.json', '.jsonl', '.log', '.md'}
                and not any(x.startswith('checkpoint-') or x == 'wandb' for x in p.relative_to(out).parts)):
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
