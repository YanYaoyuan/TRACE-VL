#!/usr/bin/env bash
# Matched training and evaluation on the verified local 40K video mixture.
set -Eeuo pipefail
: "${TRACE_REV:?set the comparison revision}"
repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
previous="${TRACE_PREVIOUS:-/nas-data/vlm/yyy/tracevl-runs/core-scale40k-54e161a-001}"
data="${TRACE_LOCAL_DATA:-/data/yyy/tracevl-data/core-scale40k-54e161a}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
rev=$(git -C "$repo" rev-parse --verify "$TRACE_REV^{commit}")
smoke="${TRACE_SMOKE:-0}"
suffix=001
args=()
if [[ "$smoke" == 1 ]]; then
    suffix=smoke-001
    args+=(--smoke)
    export TRACE_SWANLAB=0
else
    export TRACE_SWANLAB="${TRACE_SWANLAB:-1}"
fi
out="${TRACE_OUT:-/data/yyy/tracevl-runs/core-pool-${rev:0:7}-$suffix}"
mkdir -p "$out"
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo 'Comparison is already running' >&2; exit 2; }
source="$out/source"
if [[ ! -f "$source/.revision" ]]; then
    staging=$(mktemp -d "$out/source.tmp.XXXXXX")
    git -C "$repo" archive "$rev" | tar -xf - -C "$staging"
    printf '%s\n' "$rev" >"$staging/.revision"
    mv -T "$staging" "$source"
fi
[[ "$(cat "$source/.revision")" == "$rev" ]]
export PYTHONPATH="$source/src" TRACE_REV="$rev"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRACE_TRACKER=swanlab TRACE_WANDB=0
export SWANLAB_MODE="${SWANLAB_MODE:-online}" TRACE_SWANLAB_PROJECT="${TRACE_SWANLAB_PROJECT:-trace-vl-core}"
exec "$py" -m trace_vl.core_vlm.pool_experiment --previous "$previous" --local-data "$data" \
    --out "$out" --updates "${TRACE_UPDATES:-2500}" "${args[@]}"
