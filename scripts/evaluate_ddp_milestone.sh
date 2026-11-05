#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
checkpoint="${1:?Pass a checkpoint path}"
output="${2:?Pass a new evaluation directory}"
config="${3:-configs/local_2b_ddp_5epochs.yaml}"
uv run --locked trace-vl evaluate --config "$config" --checkpoint "$checkpoint" \
  --split val --output "$output"
uv run --locked python scripts/diagnose_visual_dependency.py --config "$config" \
  --checkpoint "$checkpoint" --output "$output/visual_dependency_full.json" \
  --max-videos 20 --all-questions --generate
