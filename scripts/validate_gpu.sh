#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
uv run --locked trace-vl doctor
uv run --locked trace-vl data synth --video
uv run --locked pytest -q
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl train --config configs/hardware/2x5090_smoke.yaml --output outputs/5090-single
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl evaluate --config configs/hardware/2x5090_smoke.yaml --checkpoint outputs/5090-single/last.pt --output outputs/5090-single-eval
CUDA_VISIBLE_DEVICES=0,1 uv run --locked torchrun --standalone --nproc_per_node=2 -m trace_vl train --config configs/hardware/2x5090_smoke.yaml --output outputs/5090-ddp
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl evaluate --config configs/hardware/2x5090_smoke.yaml --checkpoint outputs/5090-ddp/last.pt --output outputs/5090-ddp-eval
