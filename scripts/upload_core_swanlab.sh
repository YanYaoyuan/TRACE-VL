#!/usr/bin/env bash
# Record an existing experiment on SwanLab; never modifies training checkpoints.
set -Eeuo pipefail
: "${TRACE_RUNS:?Set TRACE_RUNS to the experiment/report directory}"
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$script_root/src${PYTHONPATH:+:$PYTHONPATH}"
export SWANLAB_MODE="${SWANLAB_MODE:-online}"
export TRACE_SWANLAB_PROJECT="${TRACE_SWANLAB_PROJECT:-trace-vl-core}"
py="${TRACE_PYTHON:-$script_root/.venv/bin/python}"
args=(--runs "$TRACE_RUNS")
[[ -z "${TRACE_OPERATOR_LOG:-}" ]] || args+=(--operator-log "$TRACE_OPERATOR_LOG")
[[ -z "${TRACE_SWANLAB_WATCH:-}" ]] || args+=(--watch "$TRACE_SWANLAB_WATCH")
"$py" -m trace_vl.core_vlm.swanlab_upload "${args[@]}"
