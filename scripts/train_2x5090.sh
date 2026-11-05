#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
uv run --locked torchrun --standalone --nproc_per_node=2 -m trace_vl train \
  --config "${1:-configs/hardware/2x5090_smoke.yaml}" --output "${2:-outputs/ddp-smoke}"
