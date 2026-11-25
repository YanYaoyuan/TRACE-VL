#!/usr/bin/env bash
# Issue #7 Test A. Uses only TRACE_GPUS (default 2,3); never touches another track's jobs.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the immutable revision under test}"
repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
weights="${TRACE_WEIGHTS:-plain}"
gpus="${TRACE_GPUS:-2,3}"
rev=$(git -C "$repo" rev-parse --verify "$TRACE_REV^{commit}")
out="${TRACE_OUT:-/data/yyy/tracevl-runs/complementary-test-a-$weights-${rev:0:7}-001}"
mkdir -p "$out"
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo "launcher already running" >&2; exit 2; }
source="$out/source"
if [[ ! -f "$source/.revision" ]]; then
    stage=$(mktemp -d "$out/source.tmp.XXXXXX")
    git -C "$repo" archive "$rev" | tar -xf - -C "$stage"
    printf '%s\n' "$rev" > "$stage/.revision"
    mv -T "$stage" "$source"
fi
[[ "$(cat "$source/.revision")" == "$rev" ]]
cd "$source"
export PYTHONPATH="$source/src" PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRACE_WANDB=0 TRACE_SWANLAB=0
CUDA_VISIBLE_DEVICES= "$py" -m pytest -q tests/test_complementary_state.py tests/test_core_vlm_native.py \
    tests/test_core_longstream_delayed.py
args=()
[[ "${TRACE_SMOKE_ONLY:-0}" != 1 ]] || args+=(--smoke-only)
"$py" -m trace_vl.complementary_memory.native_state run --out "$out" --weights "$weights" --gpus "$gpus" "${args[@]}"
files=(inputs.json smoke-ok.json questions sources logs)
[[ ! -f "$out/report.json" ]] || files+=(report.json)
tar -czf "$out/reports.tar.gz" -C "$out" "${files[@]}"
echo "TEST A BUNDLE: $out/reports.tar.gz"
