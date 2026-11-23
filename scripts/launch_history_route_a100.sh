#!/usr/bin/env bash
# Frozen-reader diagnostic on the existing ActivityNet development videos.
set -Eeuo pipefail

repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
cd "$repo"
rev=$(git rev-parse --verify "${TRACE_ROUTE_REV:-HEAD}^{commit}")
run_root=/nas-data/vlm/yyy/tracevl-runs
base="${TRACE_HISTORY_BASE:-$run_root/history-reader-3e5cb79-001}"
out="${TRACE_OUT:-$run_root/history-route-${rev:0:7}-001}"
scratch="${TRACE_HISTORY_SCRATCH:-$repo/.cache/history-reader/$(basename "$base")}"
IFS=, read -r -a devices <<< "${TRACE_CUDA_DEVICES:-0,1,2,3}"
for path in "$py" "$base/prepared.json" "$base/learned.pt"; do
    [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done
mkdir -p "$out/logs"
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo "Already running: $out" >&2; exit 2; }
trap 'code=$?; if (( code )); then echo "STOPPED ($code). See $out/logs." >&2; fi' EXIT

source="$out/source"
if [[ ! -f "$source/.revision" ]]; then
    stage=$(mktemp -d "$out/source.tmp.XXXXXX")
    git archive "$rev" | tar -xf - -C "$stage"
    printf '%s\n' "$rev" > "$stage/.revision"
    mv -T "$stage" "$source"
fi
[[ "$(cat "$source/.revision")" == "$rev" ]]
export PYTHONPATH="$source/src" TRACE_ROUTE_REV="$rev"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

module=trace_vl.history_reader.route_probe
args=(--base "$base" --out "$out" --scratch "$scratch" --shards "${#devices[@]}")
echo "HISTORY ROUTES $rev, GPUs ${devices[*]}, output $out"
CUDA_VISIBLE_DEVICES='' "$py" -m "$module" plan "${args[@]}"

pids=()
for i in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES="${devices[i]}" "$py" -m "$module" eval "${args[@]}" --shard "$i" \
        >>"$out/logs/shard-$i.log" 2>&1 &
    pids+=("$!")
done
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
        tail -n 25 "$out/logs/shard-$i.log" >&2
        failed=1
    fi
done
(( failed == 0 ))
CUDA_VISIBLE_DEVICES='' "$py" -m "$module" report "${args[@]}"
tar -czf "$out/reports.tmp" -C "$out" plan.json report.json predictions.jsonl logs
mv "$out/reports.tmp" "$out/reports.tar.gz"
echo "REPORT BUNDLE: $out/reports.tar.gz"
