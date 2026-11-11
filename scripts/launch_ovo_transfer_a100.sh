#!/usr/bin/env bash
# OVO-Bench Stage-2 transfer check (experiments.md §14.3). EVALUATION ONLY: frozen checkpoints, no training.
# Code a26cffc (exact snapshot) in a NEW output directory; the temporal-diagnostic, memory-pilot, evalrepair and
# pilot-005 runs, the OVO downloads and media are only read. ONE command runs prepare -> inventory -> flow -> full and
# advances automatically when each predeclared check passes; it stops with a report (inventory exit 3, flow exit 4)
# otherwise. Re-running resumes. TRACE_STAGE=inventory stops before any GPU work.
#   bash launch_ovo_transfer_a100.sh
set -Eeuo pipefail
export TRACE_STAGE="${TRACE_STAGE:-all}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_DIAG_RUN="${TRACE_DIAG_RUN:-/nas-data/vlm/yyy/tracevl-runs/temporal-diag-9653e2a-001}"
export TRACE_PILOT_RUN="${TRACE_PILOT_RUN:-/nas-data/vlm/yyy/tracevl-runs/memory-pilot-448adf1-001}"
export TRACE_S1_CKPT="${TRACE_S1_CKPT:-/nas-data/vlm/yyy/tracevl-runs/pilot005-6a46fb5-001/train/b4}"
export TRACE_EVALREPAIR_RUN="${TRACE_EVALREPAIR_RUN:-/nas-data/vlm/yyy/tracevl-runs/memory-evalrepair-af2f961-001}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_2b}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/ovo-transfer-a26cffc-001}"
export TRACE_OVO_ROOT="${TRACE_OVO_ROOT:-/nas-data/vlm/data/OVO_Bench}"  # existing downloads, searched only for missing media
export TRACE_MIN_FREE_GB="${TRACE_MIN_FREE_GB:-30}"
export TRACE_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRACE_WANDB=0
rev=a26cffcf0ed549bfa53db7b146ad0b3707310c40
mkdir -p "$TRACE_OUT"
exec 8>"$TRACE_OUT/.launcher.lock"
flock -n 8 || { echo 'This launcher is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
for f in "$TRACE_DIAG_RUN/split/manifest.json" "$TRACE_DIAG_RUN/split/diagnostic.jsonl" "$TRACE_DIAG_RUN/split/final_frozen.jsonl" \
         "$TRACE_DIAG_RUN/conversion.json" "$TRACE_DIAG_RUN/ovo_all.jsonl" "$TRACE_PILOT_RUN/expected.json" \
         "$TRACE_PILOT_RUN/train-identity.expected.json" "$TRACE_PILOT_RUN/train/complete.json" "$TRACE_S1_CKPT/modules.pt"; do
  [[ -f "$f" ]] || { echo "Missing frozen input: $f" >&2; exit 2; }
done
if [[ -f "$TRACE_OUT/base-revision.txt" && "$(cat "$TRACE_OUT/base-revision.txt")" != "$rev" ]]; then
  echo "$TRACE_OUT belongs to another revision; use a new TRACE_OUT" >&2; exit 2
fi
if ! git -C "$TRACE_REPO" cat-file -e "$rev^{commit}" 2>/dev/null; then
  git -C "$TRACE_REPO" fetch origin design/v3-transition-streaming
fi
git -C "$TRACE_REPO" cat-file -e "$rev^{commit}"
if [[ ! -f "$TRACE_OUT/source/.complete" ]]; then
  rm -rf "$TRACE_OUT/source" && mkdir -p "$TRACE_OUT/source"
  git -C "$TRACE_REPO" archive "$rev" | tar -x -C "$TRACE_OUT/source"
  touch "$TRACE_OUT/source/.complete"
fi
printf '%s\n' "$rev" > "$TRACE_OUT/base-revision.txt"
cd "$TRACE_OUT/source"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

"$TRACE_UV_BIN" run --locked --no-sync python - <<'PY'
import os
from pathlib import Path
import trace_vl
assert Path(trace_vl.__file__).resolve().is_relative_to(Path.cwd().resolve()), f'Wrong trace_vl import: {trace_vl.__file__}'
assert Path(os.environ['TRACE_MODEL'], 'config.json').is_file(), 'Cached model path is missing'
if os.environ['TRACE_STAGE'] != 'inventory':
    import torch
    assert torch.cuda.device_count() == 4, 'Expected four visible CUDA GPUs'
    low = []
    for i in range(4):
        free, total = torch.cuda.mem_get_info(i)
        if free / 1e9 < float(os.environ['TRACE_MIN_FREE_GB']):
            low.append(f'GPU {i}: {free / 1e9:.1f} GB free of {total / 1e9:.1f}')
    assert not low, 'Free GPU memory below TRACE_MIN_FREE_GB; stop other jobs first: ' + '; '.join(low)
print('Code a26cffc (evaluation only); stage', os.environ['TRACE_STAGE'])
PY

"$TRACE_UV_BIN" run --locked --no-sync python -c 'import pytest' 2>/dev/null \
  || { echo 'pytest is required for the launch gate' >&2; exit 2; }
CUDA_VISIBLE_DEVICES= "$TRACE_UV_BIN" run --locked --no-sync python -m pytest -q -p no:cacheprovider \
  tests/test_v3_ovo_transfer.py tests/test_v3_memory_pilot.py tests/test_v3_stage2.py

bash scripts/run_ovo_transfer.sh
echo "Submit from $TRACE_OUT: inventory-digest.txt, half-counts.json, media.json, source.json, transfer/inventory.json,"
echo "  transfer-v1-comparison/inventory.json, flow.json, inputs.json, transfer-digest.txt, transfer-report.json,"
echo "  eval/*/predictions.jsonl and the launcher log (whichever exist; STOPPED-inventory.txt if the inventory stopped)."
