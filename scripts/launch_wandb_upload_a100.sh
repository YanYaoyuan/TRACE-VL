#!/usr/bin/env bash
# Upload a COMPLETED pilot run directory to Weights & Biases: training curves per arm (replayed from
# metrics.jsonl), evaluation summary, accuracy / paired-difference / selector-diagnostic tables, all reports and
# logs as one artifact, and demos (GIF + keyframes + answers for the same validation questions in every arm).
# Code e728712 (exact snapshot in its own directory). Reads the run; writes only demos/ and wandb/ inside it.
# Does not train. Demos use GPU 0 for a few minutes per arm (set TRACE_DEMOS=0 to skip them).
#
# One-time setup on the host:
#   uv pip install --python /data/yyy/TRACE-VL/.venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple \
#       "wandb==0.31.0"
#   /data/yyy/TRACE-VL/.venv/bin/wandb login            # paste the API key once (never put it in scripts)
# If the host cannot reach wandb.ai: export WANDB_MODE=offline, run this script, then copy
#   $TRACE_RUNS/wandb/ to a machine with access and run: wandb sync $TRACE_RUNS/wandb/wandb/offline-run-*
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_RUNS="${TRACE_RUNS:?Set TRACE_RUNS to a completed run, e.g. /nas-data/vlm/yyy/tracevl-runs/pilot004-a3f8c29-001}"
export TRACE_DATA_DIR="${TRACE_DATA_DIR:-/nas-data/vlm/yyy/tracevl-data/stage1-30s-001}"
export TRACE_VIDEO_ROOT="${TRACE_VIDEO_ROOT:-/nas-data/vlm/llava/media}"
export TRACE_DEMOS="${TRACE_DEMOS:-12}"
export WANDB_PROJECT="${WANDB_PROJECT:-trace-vl}"
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rev=e728712a917fd11145ee67b5062e3b1479aa8a8d
src="$TRACE_RUNS/wandb-src-${rev:0:7}"
[[ -f "$TRACE_RUNS/summary.json" && -f "$TRACE_RUNS/identity.json" ]] || { echo "No completed run at $TRACE_RUNS" >&2; exit 2; }
exec 8>"$TRACE_RUNS/.wandb-upload.lock"
flock -n 8 || { echo 'An upload for this run is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import wandb' 2>/dev/null || {
  echo 'wandb is not installed in the environment; see the one-time setup at the top of this script.' >&2; exit 2; }
if [[ "${WANDB_MODE:-online}" != offline && -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" ~/.netrc; then
  echo 'Not logged in: run "$UV_PROJECT_ENVIRONMENT/bin/wandb login" once, or export WANDB_MODE=offline.' >&2; exit 2
fi
if ! git -C "$TRACE_REPO" cat-file -e "$rev^{commit}" 2>/dev/null; then
  git -C "$TRACE_REPO" fetch origin design/v3-transition-streaming
fi
git -C "$TRACE_REPO" cat-file -e "$rev^{commit}"
if [[ ! -f "$src/.complete" ]]; then
  rm -rf "$src" && mkdir -p "$src"
  git -C "$TRACE_REPO" archive "$rev" | tar -x -C "$src"
  touch "$src/.complete"
fi
cd "$src"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$TRACE_UV_BIN" run --locked --no-sync python -m trace_vl.cli v3 wandb-upload --runs "$TRACE_RUNS" \
  --demos "$TRACE_DEMOS" --data-dir "$TRACE_DATA_DIR" --video-root "$TRACE_VIDEO_ROOT" --device cuda:0 \
  | tee "$TRACE_RUNS/wandb-upload.json"
echo "UPLOADED: group $(basename "$TRACE_RUNS") in project $WANDB_PROJECT (demos: $TRACE_RUNS/demos/)"
