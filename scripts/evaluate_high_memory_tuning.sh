#!/usr/bin/env bash
# Run after both high-memory training arms have finished and released GPU memory.
set -euo pipefail
cd "$(dirname "$0")/.."
root="${1:-outputs/tuning-20261105}"
for arm in lr1e4 lr3e5; do
  test -f "$root/$arm/resources.json"
  test -f "$root/$arm/last.pt"
done
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_ddp_milestone.sh \
  "$root/lr1e4/last.pt" "$root/lr1e4/val" configs/tuning/high_memory_lr1e4.yaml \
  > "$root/lr1e4/evaluation.log" 2>&1 &
first=$!
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate_ddp_milestone.sh \
  "$root/lr3e5/last.pt" "$root/lr3e5/val" configs/tuning/high_memory_lr3e5.yaml \
  > "$root/lr3e5/evaluation.log" 2>&1 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
if (( status )); then
  echo "Evaluation failed; inspect each arm's evaluation.log" >&2
  exit "$status"
fi
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_ddp_milestone.sh \
  outputs/llava-2b-ddp-5epochs-20261105/best.pt "$root/baseline-4frames" \
  configs/tuning/baseline_4frames.yaml > "$root/baseline-evaluation.log" 2>&1
