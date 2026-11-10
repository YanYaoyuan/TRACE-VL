#!/usr/bin/env bash
# Stage-2 memory pilot on 4 A100s (experiments.md §14). Run SMOKE first, then PILOT, each in its own directory:
#   TRACE_MODE=smoke bash launch_memory_pilot_a100.sh   (~30-45 min incl. data build; plumbing checks)
#   TRACE_MODE=pilot bash launch_memory_pilot_a100.sh   (~4-5 h: data build, 300 updates, 8 dev evaluations)
# Code 77c5050 (exact snapshot). Reads the pilot-005 run, Stage-1 manifests and the extracted LLaVA media; writes only
# into TRACE_OUT. No other experiment directory is modified. The frozen test split and the OVO final half are unused.
set -Eeuo pipefail
export TRACE_MODE="${TRACE_MODE:?Set TRACE_MODE=smoke (first) or TRACE_MODE=pilot}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_S1_RUN="${TRACE_S1_RUN:-/nas-data/vlm/yyy/tracevl-runs/pilot005-6a46fb5-001}"
export TRACE_DATA_DIR="${TRACE_DATA_DIR:-/nas-data/vlm/yyy/tracevl-data/stage1-30s-001}"
export TRACE_VIDEO_ROOT="${TRACE_VIDEO_ROOT:-/nas-data/vlm/llava/media}"
export TRACE_LLAVA_ROOT="${TRACE_LLAVA_ROOT:-/nas-data/vlm/llava}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_2b}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/memory-${TRACE_MODE}-77c5050-001}"
export TRACE_SMOKE_CAPS="${TRACE_SMOKE_CAPS:-600 200 200}"  # smoke candidate pool (train dev test source videos)
export TRACE_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRACE_WANDB="${TRACE_WANDB:-0}"
rev=77c5050c0b951591855aec1355e1bff6188b2da5
mkdir -p "$TRACE_OUT"
exec 8>"$TRACE_OUT/.launcher.lock"
flock -n 8 || { echo 'This launcher is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
if [[ -f "$TRACE_OUT/base-revision.txt" && "$(cat "$TRACE_OUT/base-revision.txt")" != "$rev" ]]; then
  echo "$TRACE_OUT belongs to another revision; use a new TRACE_OUT" >&2; exit 2
fi
# Pilot authorization is machine-checked by the runner against the smoke run's smoke-ok.json (passed, same revision,
# resolved config, initial checkpoint and data sources); a digest file is not an authorization.
export TRACE_SMOKE_OK="${TRACE_SMOKE_OK:-/nas-data/vlm/yyy/tracevl-runs/memory-smoke-77c5050-001/smoke-ok.json}"
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
assert Path(os.environ['TRACE_MODEL'], 'config.json').is_file(), 'Cached model path is missing'
run = Path(os.environ['TRACE_S1_RUN'])
identity = json.loads((run/'identity.json').read_text())
assert identity.get('pilot') == 'pilot005', 'Stage-1 run is not pilot-005'
cfg = identity['configs']['b4']
ok, reason = training_complete(run/'train'/'b4', cfg, cfg['train']['steps'])
assert ok, f'b4: {reason}'
print('Code 77c5050; mode', os.environ['TRACE_MODE'], '; Stage-1 B4 checkpoint verified; GPUs:', [torch.cuda.get_device_name(i) for i in range(4)])
PY

"$TRACE_UV_BIN" run --locked --no-sync python -c 'import pytest' 2>/dev/null \
  || { echo 'pytest is required for the launch gate (uv pip install pytest into the environment)' >&2; exit 2; }
CUDA_VISIBLE_DEVICES= "$TRACE_UV_BIN" run --locked --no-sync python -m pytest -q -p no:cacheprovider \
  tests/test_v3_memory_pilot.py tests/test_v3_stage2.py tests/test_v3_pilot003.py

bash scripts/run_memory_pilot.sh
echo "VERIFIED ($TRACE_MODE). Submit: $TRACE_OUT/memory-digest.txt, memory-report.json, data/report.json, data/selection.json,"
echo "  data/splits/manifest.json, data/streams/build.json, train/metrics.jsonl, smoke-ok.json (smoke) and the launcher log."
