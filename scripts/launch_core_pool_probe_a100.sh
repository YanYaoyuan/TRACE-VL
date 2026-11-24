#!/usr/bin/env bash
# Same checkpoint, same observations; three intervention settings, no training.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the diagnostic revision}"
repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
base="${TRACE_BASE:-/data/yyy/tracevl-runs/core-pool-2efc7cc-001}"
data="${TRACE_DATA:-/data/yyy/tracevl-data/core-pool-probe-2efc7cc}"
rev=$(git -C "$repo" rev-parse --verify "$TRACE_REV^{commit}")
out="${TRACE_OUT:-/data/yyy/tracevl-runs/core-pool-probe-${rev:0:7}-001}"
mkdir -p "$out"
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo 'Diagnostic launcher is already running' >&2; exit 2; }
source="$out/source"
if [[ ! -f "$source/.revision" ]]; then
    stage=$(mktemp -d "$out/source.tmp.XXXXXX")
    git -C "$repo" archive "$rev" | tar -xf - -C "$stage"
    printf '%s\n' "$rev" >"$stage/.revision"
    mv -T "$stage" "$source"
fi
[[ "$(cat "$source/.revision")" == "$rev" ]]
export PYTHONPATH="$source/src" TRACE_REV="$rev"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TRACE_WANDB=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export SWANLAB_MODE="${SWANLAB_MODE:-online}" TRACE_SWANLAB_PROJECT="${TRACE_SWANLAB_PROJECT:-trace-vl-core}"
cd "$source"
"$py" -m pytest -q tests/test_core_pool_probe.py tests/test_core_vlm_native.py \
    tests/test_core_vlm_diagnostics.py
args=()
[[ "${TRACE_PREPARE_ONLY:-0}" != 1 ]] || args+=(--prepare-only)
[[ "${TRACE_SMOKE_ONLY:-0}" != 1 ]] || args+=(--smoke-only)
"$py" -m trace_vl.core_vlm.pool_probe --base "$base" --out "$out" --data "$data" \
    --workers "${TRACE_WORKERS:-16}" --gpus ${TRACE_GPUS:-0 1 2 3} \
    --minimum-free-gib "${TRACE_MIN_FREE_GIB:-12}" "${args[@]}"
if [[ -f "$out/report.json" ]]; then
    tar -czf "$out/reports.tar.gz" -C "$out" \
        inputs.json run.json preparation.json smoke-ok.json report.json digest.txt eval logs
    echo "REPORT BUNDLE: $out/reports.tar.gz"
    if [[ "${TRACE_SWANLAB:-1}" == 1 ]]; then
        TRACE_RUNS="$out" bash "$source/scripts/upload_core_swanlab.sh" \
            >"$out/swanlab-upload.log" 2>&1 \
            || echo "SwanLab upload needs retry; local report complete: $out/swanlab-upload.log"
    fi
fi
