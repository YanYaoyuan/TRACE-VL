#!/usr/bin/env bash
# Recent-window control on the frozen OVO transfer population (experiments.md §14.4). EVALUATION ONLY: frozen
# checkpoints, no training, the final half untouched. Code REVSHORT (exact snapshot) in a NEW output directory; the
# completed transfer run (its manifest, media overlay and four arms), the memory pilot and pilot-005 are only read.
# ONE command: tests -> verify -> 4-stream GPU flow check with A-B-A isolation -> six evaluations -> report. It
# continues automatically when a check passes and stops otherwise (flow check: exit 4). Re-running resumes.
#   bash launch_recent_window_a100.sh
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_TRANSFER_RUN="${TRACE_TRANSFER_RUN:-/nas-data/vlm/yyy/tracevl-runs/ovo-transfer-cdbac0c-001}"
export TRACE_PILOT_RUN="${TRACE_PILOT_RUN:-/nas-data/vlm/yyy/tracevl-runs/memory-pilot-448adf1-001}"
export TRACE_S1_CKPT="${TRACE_S1_CKPT:-/nas-data/vlm/yyy/tracevl-runs/pilot005-6a46fb5-001/train/b4}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_2b}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/recent-window-REVSHORT-001}"
export TRACE_MIN_FREE_GB="${TRACE_MIN_FREE_GB:-30}"
export TRACE_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRACE_WANDB=0
rev=REVFULL
mkdir -p "$TRACE_OUT"
exec 8>"$TRACE_OUT/.launcher.lock"
flock -n 8 || { echo 'This launcher is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
for f in "$TRACE_TRANSFER_RUN/transfer/streams.jsonl" "$TRACE_TRANSFER_RUN/transfer/questions.jsonl" \
         "$TRACE_TRANSFER_RUN/transfer/inventory.json" "$TRACE_TRANSFER_RUN/inputs.json" "$TRACE_TRANSFER_RUN/source.json" \
         "$TRACE_TRANSFER_RUN/transfer-report.json" "$TRACE_TRANSFER_RUN/flow_streams.jsonl" \
         "$TRACE_TRANSFER_RUN/flow/s1_fifo/predictions.jsonl" "$TRACE_TRANSFER_RUN/eval/s1_fifo/predictions.jsonl" \
         "$TRACE_TRANSFER_RUN/eval/s2_fifo/predictions.jsonl" "$TRACE_PILOT_RUN/expected.json" \
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
import torch
import trace_vl
assert Path(trace_vl.__file__).resolve().is_relative_to(Path.cwd().resolve()), f'Wrong trace_vl import: {trace_vl.__file__}'
assert Path(os.environ['TRACE_MODEL'], 'config.json').is_file(), 'Cached model path is missing'
assert torch.cuda.device_count() == 4, 'Expected four visible CUDA GPUs'
low = []
for i in range(4):
    free, total = torch.cuda.mem_get_info(i)
    if free / 1e9 < float(os.environ['TRACE_MIN_FREE_GB']):
        low.append(f'GPU {i}: {free / 1e9:.1f} GB free of {total / 1e9:.1f}')
assert not low, 'Free GPU memory below TRACE_MIN_FREE_GB; stop other jobs first: ' + '; '.join(low)
print('Code REVSHORT (evaluation only): recent-window control')
PY

"$TRACE_UV_BIN" run --locked --no-sync python -c 'import pytest' 2>/dev/null \
  || { echo 'pytest is required for the launch gate' >&2; exit 2; }
CUDA_VISIBLE_DEVICES= "$TRACE_UV_BIN" run --locked --no-sync python -m pytest -q -p no:cacheprovider \
  tests/test_v3_recent_window.py tests/test_v3_ovo_transfer.py tests/test_v3_stage2.py

bash scripts/run_recent_window.sh
echo "Submit from $TRACE_OUT: recent-window-digest.txt, recent-window-report.json, inputs.json, flow.json,"
echo "  eval/*/predictions.jsonl, flow/*/predictions.jsonl and the launcher log."
