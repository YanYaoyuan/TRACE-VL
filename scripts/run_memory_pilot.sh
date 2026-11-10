#!/usr/bin/env bash
# Stage-2 memory pilot (experiments.md §14; trace_vl.v3.memory_pilot). Modes:
#   smoke : 2 training updates, 2 dev streams per policy (plumbing: loading, eviction, scorer updates, parity)
#   pilot : 300 updates, 40 dev streams per policy (the decision run)
#   TRACE_MODE=smoke|pilot TRACE_S1_RUN=<pilot-005 run> TRACE_DATA_DIR=<stage1 manifests> TRACE_VIDEO_ROOT=<media>
#   TRACE_LLAVA_ROOT=<llava root> TRACE_OUT=<NEW dir> bash scripts/run_memory_pilot.sh
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_MODE:?smoke or pilot}" "${TRACE_S1_RUN:?}" "${TRACE_DATA_DIR:?}" "${TRACE_VIDEO_ROOT:?}" "${TRACE_LLAVA_ROOT:?}" "${TRACE_OUT:?}"
export TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_LLAVA_MC_JSON="${TRACE_LLAVA_MC_JSON:-}" TRACE_MODEL="${TRACE_MODEL:-}"
case "$TRACE_MODE" in
  smoke) steps=2; dev_streams=2; caps=(60 20 20) ;;
  pilot) steps=300; dev_streams=40; caps=(3000 400 400) ;;
  *) echo "TRACE_MODE must be smoke or pilot" >&2; exit 2 ;;
esac
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.lock"
flock -n 9 || { echo "Another memory pilot is using $TRACE_OUT" >&2; exit 2; }
log() { echo "[$(date -Is)] $*"; }
s1="$TRACE_S1_RUN/train/b4"
[[ -f "$s1/modules.pt" && -f "$s1/lora/adapter_config.json" ]] || { echo "missing Stage-1 B4 checkpoint in $s1" >&2; exit 2; }

log "[1/6] Config (memory_pilot_b4 + local model path)"
cfg="$TRACE_OUT/memory_pilot_b4.yaml"
{ echo "extends: $root/configs/v3/memory_pilot_b4.yaml"
  [[ -n "$TRACE_MODEL" ]] && printf 'backbone:\n  name: %s\n  local_files_only: true\n' "$TRACE_MODEL"; true; } > "$cfg"

log "[2/6] Data: convert the extracted subset's MC questions; freeze splits by source video (Stage-1 videos excluded); stitch streams per split"
if [[ ! -f "$TRACE_OUT/data/streams/dev.jsonl" ]]; then
  mkdir -p "$TRACE_OUT/data/streams"
  mapfile -t mc < <(if [[ -n "$TRACE_LLAVA_MC_JSON" ]]; then echo "$TRACE_LLAVA_MC_JSON"; else ls "$TRACE_LLAVA_ROOT"/0_30_s_youtube_v0_1/*_mc_*.json; fi)
  ((${#mc[@]})) || { echo "no LLaVA MC json found; set TRACE_LLAVA_MC_JSON" >&2; exit 2; }
  "${py[@]}" -m trace_vl.cli v3 convert-llava-mc "${mc[@]}" --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_OUT/data/mc_all.jsonl" --val-fraction 0
  "${py[@]}" -m trace_vl.cli v3 stage2-splits "$TRACE_OUT/data/mc_all.jsonl" --exclude "$TRACE_DATA_DIR/train.jsonl" "$TRACE_DATA_DIR/val.jsonl" \
    --output "$TRACE_OUT/data/splits" --max-videos "${caps[@]}"
  "${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.data import build_stage2_streams
out = Path(os.environ["TRACE_OUT"]) / "data"
report = {}
for split in ("train", "dev", "test"):
    report[split] = build_stage2_streams(str(out / "splits" / f"{split}.jsonl"), str(out / "streams" / f"{split}.tmp"),
                                         os.environ["TRACE_VIDEO_ROOT"], split=split, stream_seconds=300.0,
                                         delay_min=45.0, delay_max=240.0, immediate_fraction=0.2, distractor_fraction=0.4)
    # delay_min 45 s > the 16 s window: delayed evidence has left the recent window; immediate ones are the control
for split in ("train", "test", "dev"):  # dev last: its presence marks a complete build
    (out / "streams" / f"{split}.tmp").replace(out / "streams" / f"{split}.jsonl")
(out / "streams" / "build.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
fi

log "[3/6] Stage-2 training from the Stage-1 B4 checkpoint (${steps} updates, ${TRACE_GPUS} GPUs)"
s2="$TRACE_OUT/train"
if [[ ! -f "$s2/modules.pt" || $(wc -l < "$s2/metrics.jsonl" 2>/dev/null || echo 0) -lt $steps ]]; then
  [[ -e "$s2" ]] && mv "$s2" "$s2.incomplete-$(date +%s)"
  "$uv_bin" run --locked --no-sync torchrun --standalone --nproc_per_node="$TRACE_GPUS" -m trace_vl.cli v3 train-stage2 \
    --config "$cfg" --streams "$TRACE_OUT/data/streams/train.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$s2" \
    --init "$s1/modules.pt" --adapter "$s1/lora" --max-steps "$steps"
fi

run_eval() {  # name, checkpoint dir, extra args...
  local name="$1" ckpt="$2"; shift 2
  local pids=()
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval-stage2 --config "$cfg" \
      --streams "$TRACE_OUT/data/streams/dev.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_OUT/eval/$name" \
      --mode v3 --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev --limit "$dev_streams" \
      --shard "$i/$TRACE_GPUS" --device cuda:0 "$@" > "$TRACE_OUT/eval-$name-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "eval $name failed; see $TRACE_OUT/eval-$name-shard*.log" >&2; exit 1; }; done
  # assembly (no model load when every stream is done)
  CUDA_VISIBLE_DEVICES=0 "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval-stage2 --config "$cfg" \
    --streams "$TRACE_OUT/data/streams/dev.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_OUT/eval/$name" \
    --mode v3 --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev --limit "$dev_streams" "$@" > /dev/null
  log "  $name done"
}
log "[4/6] Dev evaluation of the Stage-2 model: only the memory policy changes (same checkpoint, same budget)"
run_eval s2_tkm "$s2" --policy tkm
run_eval s2_mass "$s2" --policy mass
run_eval s2_fifo "$s2" --policy fifo
run_eval s2_full "$s2" --policy full
run_eval s2_tkm_evict8 "$s2" --policy tkm --evict-every 8   # train/eval eviction-cadence parity
log "[5/6] Separate question: Stage-1 B4 checkpoint under the heuristic policies (does Stage-2 training help?)"
run_eval s1_mass "$s1" --policy mass
run_eval s1_fifo "$s1" --policy fifo
run_eval s1_full "$s1" --policy full

log "[6/6] Report"
"${py[@]}" -m trace_vl.cli v3 memory-report --eval-root "$TRACE_OUT/eval" --window-seconds 16 --output "$TRACE_OUT/memory-report.json" > /dev/null
"${py[@]}" - "$TRACE_OUT" "$TRACE_MODE" <<'PY'
import json, sys
from pathlib import Path
out, mode = Path(sys.argv[1]), sys.argv[2]
r = json.loads((out / "memory-report.json").read_text())
metrics = [json.loads(s) for s in (out / "train" / "metrics.jsonl").read_text().splitlines() if s.strip()]
lines = [f"MEMORY PILOT DIGEST (mode {mode}; dev split; window {r['window_seconds']} s)"]
lines.append(f"training: {len(metrics)} updates; retention update norm first/last "
             f"{metrics[0].get('update_norm_retention')}/{metrics[-1].get('update_norm_retention')}; "
             f"ret_pos/neg last {metrics[-1].get('ret_pos')}/{metrics[-1].get('ret_neg')}; q_acc last {metrics[-1].get('q_acc')}")
for name, s in r["runs"].items():
    lines.append(f"{name}: all {s['all']['accuracy_pct']} {s['all']['ci95_pct']} (n {s['all']['n']}) | beyond-window "
                 f"{s['beyond_window']['accuracy_pct']} {s['beyond_window']['ci95_pct']} (n {s['beyond_window']['n']}) | "
                 f"evidence kept {s['evidence_kept_beyond_window']} | KV tokens {s['kv_tokens_mean']} ({s['kv_mb_mean']} MB) | "
                 f"evicted {s['evicted_tokens_mean']} of {s['streamed_tokens_mean']} | age median {s['evidence_age_median']} s | "
                 f"{s['seconds_per_stream']} s/stream")
for pair, p in r["paired"].items():
    lines.append(f"{pair}: beyond-window {p['beyond_window']['delta_pp']} {p['beyond_window']['ci95_pp']} | all {p['all']['delta_pp']} {p['all']['ci95_pp']}")
lines.append(f"DECISION {json.dumps(r['decision'])}")
if mode == "smoke":
    checks = {"retention scorer updated": any((m.get("update_norm_retention") or 0) > 0 for m in metrics),
              "evictions happened": r["runs"]["s2_tkm"]["evicted_tokens_mean"] > 0,
              "all eight evaluations assembled": len(r["runs"]) == 8}
    lines.append(f"SMOKE CHECKS {checks}")
    if not all(checks.values()):
        raise SystemExit("\n".join(lines) + "\nSMOKE FAILED")
text = "\n".join(lines) + "\n"
(out / "memory-digest.txt").write_text(text)
print(text)
PY
log "Completed ($TRACE_MODE): $TRACE_OUT/memory-report.json, memory-digest.txt"
