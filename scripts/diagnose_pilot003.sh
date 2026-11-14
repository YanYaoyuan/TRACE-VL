#!/usr/bin/env bash
# Diagnostics for a completed pilot-003 run (PR #3 review), one command, resumable:
#   1. gist-only ablation evals of the TRAINED transition and after arms (does the LM use the 12 slot tokens?)
#   2. provenance, per-question agreement, validation NLL, training curves (CPU)
#   3. weight changes vs the replicated initialization and an activation probe (GPU 0)
#   TRACE_RUNS=/nas-data/vlm/yyy/tracevl-runs/pilot003-cc00d60-001 TRACE_DATA_DIR=... TRACE_VIDEO_ROOT=... \
#   nohup bash scripts/diagnose_pilot003.sh > diagnose-003.log 2>&1 &
# Output: $TRACE_RUNS/diagnostics.json (post it on issue #4). Optional: TRACE_GPUS=4 TRACE_PROBE_VIDEOS=8.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_RUNS:?Set TRACE_RUNS to the completed pilot-003 run directory}"
: "${TRACE_DATA_DIR:?Set TRACE_DATA_DIR}"
: "${TRACE_VIDEO_ROOT:?Set TRACE_VIDEO_ROOT}"
TRACE_GPUS="${TRACE_GPUS:-4}"
uv_bin="${TRACE_UV_BIN:-uv}"
[[ -f "$TRACE_RUNS/identity.json" && -f "$TRACE_RUNS/summary.json" ]] || { echo "not a completed pilot-003 run" >&2; exit 2; }
exec 9>"$TRACE_RUNS/.diagnose.lock"
flock -n 9 || { echo "diagnostics already running for $TRACE_RUNS" >&2; exit 2; }
log() { echo "[$(date -Is)] $*"; }

for arm in transition after; do
  cfg="$TRACE_RUNS/configs/${arm}_gistonly.yaml"
  printf 'extends: %s\ntransition:\n  emit_slots: false\n' "$TRACE_RUNS/configs/$arm.yaml" > "$cfg"
  out="$TRACE_RUNS/eval/${arm}_gistonly"
  log "[1/2] gist-only ablation of trained $arm (${TRACE_GPUS} shards)"
  pids=()
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval --config "$cfg" --mode v3 \
      --checkpoint "$TRACE_RUNS/train/$arm/modules.pt" --adapter "$TRACE_RUNS/train/$arm/lora" \
      --records "$TRACE_DATA_DIR/val.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$out" \
      --shard "$i/$TRACE_GPUS" --device cuda:0 > "$TRACE_RUNS/eval-${arm}_gistonly-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "ablation eval failed; see $TRACE_RUNS/eval-${arm}_gistonly-shard*.log" >&2; exit 1; }; done
  "$uv_bin" run --locked --no-sync python - "$out" "$TRACE_DATA_DIR/val.jsonl" <<'PY'
import json, sys
from pathlib import Path
out, gold = Path(sys.argv[1]), [json.loads(s) for s in Path(sys.argv[2]).read_text().splitlines() if s.strip()]
ids = [json.loads(s)["id"] for p in out.glob("predictions.*.jsonl") for s in p.read_text().splitlines() if s.strip()]
if len(ids) != len(set(ids)) or set(ids) != {r["id"] for r in gold}:
    raise SystemExit(f"{out}: incomplete or duplicate predictions")
if any(json.loads(p.read_text()) for p in out.glob("errors.*.json")):
    raise SystemExit(f"{out}: evaluation errors")
PY
done

log "[2/2] provenance, agreement, NLL, curves, weight changes, activation probe (GPU 0)"
CUDA_VISIBLE_DEVICES=0 "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 diagnose-pilot --runs "$TRACE_RUNS" \
  --data-dir "$TRACE_DATA_DIR" --video-root "$TRACE_VIDEO_ROOT" --probe-videos "${TRACE_PROBE_VIDEOS:-8}" \
  --device cuda:0 --output "$TRACE_RUNS/diagnostics.json"
log "Completed: $TRACE_RUNS/diagnostics.json"
