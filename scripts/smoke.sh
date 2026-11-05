#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
uv run --locked trace-vl data synth --video
uv run --locked trace-vl data validate data/synthetic/manifest.jsonl
uv run --locked pytest -q
uv run --locked trace-vl train --config configs/tiny_cpu.yaml --max-steps 3 --output outputs/smoke
uv run --locked trace-vl evaluate --config configs/tiny_cpu.yaml --checkpoint outputs/smoke/last.pt --output outputs/smoke-eval
