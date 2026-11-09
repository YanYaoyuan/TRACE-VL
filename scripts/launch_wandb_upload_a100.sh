#!/usr/bin/env bash
# Upload a COMPLETED pilot run directory to Weights & Biases: training curves per arm (replayed from
# metrics.jsonl), evaluation summary, accuracy / paired-difference / selector-diagnostic tables, all reports and
# logs as one artifact, and demos (GIF + keyframes + answers for the same validation questions in every arm).
# Code 6a46fb5 (exact snapshot in its own directory). Reads the run; writes only demos/ and wandb/ inside it.
# Does not train. Demos are OFF by default (first verification); TRACE_DEMOS=12 renders them on GPU 0.
# The operator log outside the run directory is included as operator/<name> (TRACE_OPERATOR_LOG).
# Network problems never fail silently: an online-init failure falls back to offline collection (reported), and
# offline data is synced later with: trace-vl v3 wandb-sync --runs $TRACE_RUNS
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
export TRACE_DEMOS="${TRACE_DEMOS:-0}"
export TRACE_OPERATOR_LOG="${TRACE_OPERATOR_LOG:-$TRACE_REPO/pilot-004.log}"
export WANDB_PROJECT="${WANDB_PROJECT:-trace-vl}"
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rev=6a46fb52fa38358089a8d6487d8205f4efc98fdf
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
extra=()
if [[ -f "$TRACE_OPERATOR_LOG" ]]; then extra=(--extra-file "$TRACE_OPERATOR_LOG"); else echo "Note: operator log $TRACE_OPERATOR_LOG not found"; fi
"$TRACE_UV_BIN" run --locked --no-sync python -m trace_vl.cli v3 wandb-upload --runs "$TRACE_RUNS" \
  --demos "$TRACE_DEMOS" --data-dir "$TRACE_DATA_DIR" --video-root "$TRACE_VIDEO_ROOT" --device cuda:0 "${extra[@]}" \
  --output "$TRACE_RUNS/wandb-upload.json"
"$TRACE_UV_BIN" run --locked --no-sync python - "$TRACE_RUNS/wandb-upload.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
reports = r["artifacts"].get("reports") or []
scores = [e for e in reports if e.endswith("score.json")]
print(f"mode={r.get('mode')} train_runs={r['train_runs']} eval_run={r['eval_run']}")
print(f"report artifact: {len(reports)} files, {len(scores)} score.json (unique {len(set(scores))}), "
      f"operator log included: {any(e.startswith('operator/') for e in reports)}")
print("offline runs to sync:", len(r.get("offline_runs", [])), "| errors:", r["errors"] or "none")
assert len(scores) == len(set(scores)), "artifact path collision"
print("UPLOAD CHECKED:", sys.argv[1])
PY
