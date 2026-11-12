#!/usr/bin/env bash
# One launcher: CPU data audit/extraction -> frozen Qwen features -> queued experiments -> report bundle.
# First pass tests annotated response timing; it does NOT claim surveillance false alarms/hour.
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
# Pinned tested code; local development can set TRACE_REV=working.
rev="${TRACE_REV:-fd11ff0001bb6ca9a55bbdbdc9cb4f871a92c106}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/event-state-${rev:0:7}-001}"
export TRACE_DATA="${TRACE_DATA:-/nas-data/vlm/yyy/tracevl-data/event-state-v1}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_4b}"
export TRACE_JOY="${TRACE_JOY:-/nas-data/vlm/data/JoyAI-VL-Interaction/event_grounding/event_grounding.json}"
export TRACE_MEDIA="${TRACE_MEDIA:-/nas-data/vlm/llava}"
export TRACE_DEVICES="${TRACE_DEVICES:-0,1,2,3}"
# CPU controls: TRACE_DATA_WORKERS=32 for full decode, TRACE_ARCHIVE_WORKERS=8 for archive reads.
# Optional TRACE_REUSE_PREPARATION points to a stopped previous run/data; inputs are verified before reuse.
export TRACE_PARALLEL_ARMS="${TRACE_PARALLEL_ARMS:-4}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset CUDA_VISIBLE_DEVICES  # TRACE_DEVICES names physical GPUs for child jobs
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.launcher.lock"
flock -n 9 || { echo "Already running: $TRACE_OUT" >&2; exit 2; }
exec > >(tee -a "$TRACE_OUT/launcher.log") 2>&1
trap 'rc=$?; echo "STOPPED (exit $rc). Logs and checkpoints retained in $TRACE_OUT. Rerun this command to resume." >&2; exit "$rc"' ERR
if [[ "$rev" == working ]]; then
  source_dir="$TRACE_REPO"
  code="local-$(git -C "$TRACE_REPO" rev-parse --short HEAD)"
else
  git -C "$TRACE_REPO" cat-file -e "$rev^{commit}" 2>/dev/null || git -C "$TRACE_REPO" fetch origin design/v3-transition-streaming
  code="$(git -C "$TRACE_REPO" rev-parse "$rev^{commit}")"
  source_dir="$TRACE_OUT/source"
  if [[ -f "$TRACE_OUT/source-revision.txt" ]]; then
    [[ "$(cat "$TRACE_OUT/source-revision.txt")" == "$code" ]] || { echo 'Source version mismatch' >&2; exit 2; }
  fi
  if [[ ! -f "$TRACE_OUT/source-revision.txt" ]]; then
    mkdir -p "$source_dir"
    git -C "$TRACE_REPO" archive "$code" | tar -x -C "$source_dir"
    printf '%s\n' "$code" > "$TRACE_OUT/source-revision.txt"
  fi
fi
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
  # Existing CUDA envs are never re-synced implicitly. Only create a missing UV environment.
  "$TRACE_UV_BIN" sync --project "$source_dir" --python 3.10 --locked --extra dev --extra vlm --extra data
fi
export PYTHONPATH="$source_dir/src"
cd "$source_dir"
"$TRACE_UV_BIN" run --locked --no-sync python -c 'import torch, transformers, cv2, pytest, pyarrow; print("Environment imports OK")'
env -u TRACE_REUSE_PREPARATION "$TRACE_UV_BIN" run --locked --no-sync python -m pytest -q tests/test_event_state.py
[[ -f "$TRACE_JOY" ]] || { echo "Missing official JoyAI annotation: $TRACE_JOY. Set TRACE_JOY to its existing path." >&2; exit 2; }
config="${TRACE_CONFIG:-$source_dir/configs/event_state/pilot.json}"
args=(--config "$config" --out "$TRACE_OUT" --code "$code" --model "$TRACE_MODEL"
      --annotation "$TRACE_JOY" --media-root "$TRACE_MEDIA" --data-root "$TRACE_DATA"
      --devices "$TRACE_DEVICES" --parallel-arms "$TRACE_PARALLEL_ARMS"
      --min-free-gib "${TRACE_MIN_FREE_GIB:-24}")
# Optional second media root, e.g. originals distributed separately with JoyAI.
if [[ -n "${TRACE_EXTRA_MEDIA:-}" ]]; then args+=(--media-root "$TRACE_EXTRA_MEDIA"); fi
if [[ "${TRACE_NO_DOWNLOAD:-0}" == 1 ]]; then args+=(--no-download); fi
"$TRACE_UV_BIN" run --locked --no-sync python -m trace_vl.event_state.pipeline "${args[@]}"
