#!/usr/bin/env bash
# Temporal diagnostic on the A100 host (experiments.md §13): pilot-005's EXISTING final checkpoints (transition-grid,
# after-grid, B4) + native on the OVO-Bench diagnostic half. No training. Code 9653e2a (exact snapshot).
# Reads the pilot-005 run and OVO-Bench; writes only into TRACE_OUT (a NEW directory).
# Time: depends on the selected OVO subset (printed after the split). Rough expectation for <= 100 questions per
# task x 9 tasks, streams <= 180 s: about 1.5-3 h wall time on 4 A100s for all four arms.
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_CKPT_RUN="${TRACE_CKPT_RUN:-/nas-data/vlm/yyy/tracevl-runs/pilot005-6a46fb5-001}"
export TRACE_OVO_ROOT="${TRACE_OVO_ROOT:-/nas-data/vlm/data/OVO_Bench}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/temporal-diag-9653e2a-001}"
export TRACE_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRACE_WANDB=0
rev=9653e2ac207dc667d0b10958f203a6e79d3a29b5
mkdir -p "$TRACE_OUT"
exec 8>"$TRACE_OUT/.launcher.lock"
flock -n 8 || { echo 'This launcher is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
[[ -f "$TRACE_CKPT_RUN/identity.json" ]] || { echo "No pilot run at $TRACE_CKPT_RUN" >&2; exit 2; }
[[ -d "$TRACE_OVO_ROOT" ]] || { echo "No OVO-Bench directory at $TRACE_OVO_ROOT" >&2; exit 2; }
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
import json, os
from pathlib import Path
import torch
import trace_vl
from trace_vl.v3.pilot import training_complete
assert Path(trace_vl.__file__).resolve().is_relative_to(Path.cwd().resolve()), f'Wrong trace_vl import: {trace_vl.__file__}'
assert torch.cuda.device_count() == 4, 'Expected four visible CUDA GPUs'
run = Path(os.environ['TRACE_CKPT_RUN'])
identity = json.loads((run/'identity.json').read_text())
assert identity.get('pilot') == 'pilot005', 'checkpoint run is not pilot-005'
for arm in ('transition', 'after', 'b4'):
    cfg = identity['configs'][arm]
    ok, reason = training_complete(run/'train'/arm, cfg, cfg['train']['steps'])
    assert ok, f'{arm}: {reason}'
print('Code 9653e2a; checkpoints: pilot-005 final (verified completion records); GPUs:', [torch.cuda.get_device_name(i) for i in range(4)])
PY

if "$TRACE_UV_BIN" run --locked --no-sync python -c 'import pytest' 2>/dev/null; then
  CUDA_VISIBLE_DEVICES= "$TRACE_UV_BIN" run --locked --no-sync python -m pytest -q -p no:cacheprovider tests/test_v3_temporal.py
fi

bash scripts/run_temporal_diagnostic.sh
echo "VERIFIED: complete coverage for native + 3 arms on the diagnostic half. Submit: $TRACE_OUT/temporal-digest.txt,"
echo "  temporal-report.json, conversion.json, split/manifest.json and the launcher log."
