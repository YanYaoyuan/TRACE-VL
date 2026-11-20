#!/usr/bin/env bash
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the tested batch commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_PREPARED="${TRACE_PREPARED:-/nas-data/vlm/yyy/tracevl-runs/core-scale20k-b2919d8-001}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-batch-${TRACE_REV:0:7}-001}"
export TRACE_WANDB="${TRACE_WANDB:-1}" WANDB_PROJECT="${WANDB_PROJECT:-trace-vl-core}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRACE_WANDB_INIT_TIMEOUT="${TRACE_WANDB_INIT_TIMEOUT:-15}"
export TRACE_WANDB_GROUP="${TRACE_WANDB_GROUP:-$(basename "$TRACE_OUT")}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
py="${TRACE_PYTHON:-$TRACE_REPO/.venv/bin/python}"
[[ -x "$py" ]]
cd "$TRACE_REPO"
revision=$(git rev-parse --verify "${TRACE_REV}^{commit}")
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.launcher.lock"
flock -n 9 || { echo 'This batch benchmark is already running' >&2; exit 2; }
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
"$py" - <<'PY'
import os
import torch
from trace_vl.v3.tracking import enabled_from_env
if enabled_from_env():
    import wandb
    print('W&B SDK', wandb.__version__, 'mode', os.environ.get('WANDB_MODE', 'online'))
assert torch.cuda.device_count() == 4, 'Four visible GPUs required'
free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(4)]
print('Free GPU GiB', free)
assert min(free) >= 30, 'Free at least 30 GiB on each A100 before benchmarking'
PY
"$py" -m pytest -q tests/test_core_vlm_batch.py tests/test_core_vlm_preservation.py tests/test_core_vlm_train.py
"$py" -m trace_vl.core_vlm.batch_benchmark --prepared "$TRACE_PREPARED" --out "$TRACE_OUT" \
  --steps "${TRACE_BENCH_STEPS:-12}" --warmup 2 --recipe "${TRACE_RECIPE:-preserved}"
tar -czf "$TRACE_OUT/batch-reports.tar.gz" -C "$TRACE_OUT" batch-report.json trials.json run.json logs
echo "REPORT BUNDLE: $TRACE_OUT/batch-reports.tar.gz"
