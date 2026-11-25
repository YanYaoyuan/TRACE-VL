#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
uv_run=(uv run --locked --extra dev --extra hf --extra lora --extra track)
"${uv_run[@]}" trace-vl data synth --video
"${uv_run[@]}" trace-vl data validate data/synthetic/manifest.jsonl
"${uv_run[@]}" pytest -q
"${uv_run[@]}" trace-vl train --config configs/tiny_cpu.yaml --max-steps 3 --output outputs/smoke
"${uv_run[@]}" trace-vl evaluate --config configs/tiny_cpu.yaml --checkpoint outputs/smoke/last.pt --output outputs/smoke-eval
