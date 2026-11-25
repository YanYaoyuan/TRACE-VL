#!/usr/bin/env bash
# Add one independently trained static arm; preserve source runs and their caches.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the static-control revision}"
repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
base="${TRACE_BASE:-/data/yyy/tracevl-runs/core-pool-2efc7cc-001}"
observations="${TRACE_OBSERVATIONS:-/data/yyy/tracevl-runs/core-pool-probe-a68217e-001}"
rev=$(git -C "$repo" rev-parse --verify "$TRACE_REV^{commit}")
out="${TRACE_OUT:-/data/yyy/tracevl-runs/core-static-pool-${rev:0:7}-001}"
mkdir -p "$out"
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo 'Static control launcher already running' >&2; exit 2; }
source="$out/source"
if [[ ! -f "$source/.revision" ]]; then
    stage=$(mktemp -d "$out/source.tmp.XXXXXX")
    git -C "$repo" archive "$rev" | tar -xf - -C "$stage"
    printf '%s\n' "$rev" > "$stage/.revision"
    mv -T "$stage" "$source"
fi
[[ "$(cat "$source/.revision")" == "$rev" ]]
export PYTHONPATH="$source/src" TRACE_REV="$rev"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TRACE_WANDB=0 TRACE_SWANLAB=0
cd "$source"
"$py" -m pytest -q tests/test_core_static_pool.py tests/test_core_pool_probe.py \
    tests/test_core_vlm_native.py tests/test_core_vlm_train.py tests/test_core_vlm_preservation.py
args=()
[[ "${TRACE_SMOKE_ONLY:-0}" != 1 ]] || args+=(--smoke-only)
"$py" -m trace_vl.core_vlm.static_pool --base "$base" --observations "$observations" \
    --out "$out" "${args[@]}"
if [[ -f "$out/report.json" ]]; then
    tar -czf "$out/reports.tar.gz" -C "$out" inputs.json smoke-ok.json \
        train-smoke-audit.json train-audit.json report.json digest.txt eval logs
    echo "STATIC CONTROL REPORT BUNDLE: $out/reports.tar.gz"
fi
