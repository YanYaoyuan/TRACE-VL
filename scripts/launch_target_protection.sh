#!/usr/bin/env bash
# Original Test A and B0 stay read-only. CPU tests before any GPU inference.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the immutable revision under test}"
repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
rev=$(git -C "$repo" rev-parse --verify "$TRACE_REV^{commit}")
out="${TRACE_OUT:-/data/yyy/tracevl-runs/target-protected32-${rev:0:7}-001}"
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
CUDA_VISIBLE_DEVICES= "$py" -m pytest -q tests/test_target_protection.py tests/test_complementary_state.py \
    tests/test_core_vlm_native.py tests/test_core_longstream_delayed.py
args=()
[[ "${TRACE_SMOKE_ONLY:-0}" != 1 ]] || args+=(--smoke-only)
"$py" -m trace_vl.target_protection.run run --out "$out" --gpus "${TRACE_GPUS:-0,1,2,3}" "${args[@]}"
files=(inputs.json smoke-ok.json logs)
[[ ! -f "$out/report.json" ]] || files+=(report.json questions sources)
tar -czf "$out/reports.tar.gz" -C "$out" "${files[@]}"
echo "TARGET PROTECTION BUNDLE: $out/reports.tar.gz"
