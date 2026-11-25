#!/usr/bin/env bash
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to a tested immutable revision}"
repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="$repo/.venv/bin/python"
rev=$(git -C "$repo" rev-parse --verify "$TRACE_REV^{commit}")
out="${TRACE_OUT:-/data/yyy/tracevl-runs/trained-reader-${rev:0:7}-001}"
mkdir -p "$out"
exec 9>"$out/.launcher.lock"
flock -n 9 || exit 2
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
"$py" -m pytest -q tests/test_longstream_training.py tests/test_core_vlm_native.py tests/test_core_longstream.py
"$py" -m trace_vl.longstream_training.run --out "$out" \
    --data "${TRACE_DATA:-/data/yyy/tracevl-data/trained-reader-20261125-001}" \
    --model /data/yyy/tracevl-data/core-pool-probe-2efc7cc/model \
    --adapter /data/yyy/tracevl-runs/core-pool-2efc7cc-001/train/plain/checkpoint-002500/adapter
# Frozen visual tensors and optimizer states stay on the experiment disk.
tar -czf "$out/reports.tar.gz" -C "$out" report.json smoke-ok.json logs reservoir/run.json reservoir/status.json reservoir/metrics.jsonl reservoir/dev-predictions.json learned/run.json learned/status.json learned/metrics.jsonl learned/dev-predictions.json
echo "MATCHED LONGSTREAM READER PILOT COMPLETE: $out/report.json"
