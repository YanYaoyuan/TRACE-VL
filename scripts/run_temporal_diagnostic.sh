#!/usr/bin/env bash
# Evaluation-first temporal diagnostic (src/trace_vl/v3/temporal.py for the predeclared design and decision).
# Existing checkpoints only (no training): native + transition / after / b4 from TRACE_CKPT_RUN.
#   TRACE_CKPT_RUN=<completed pilot run dir> TRACE_OVO_ROOT=<OVO-Bench dir> TRACE_OUT=<NEW dir> bash scripts/run_temporal_diagnostic.sh
# Optional: TRACE_GPUS=4 TRACE_PER_TASK=100 TRACE_MAX_QUERY_SECONDS=180 TRACE_OVO_ANNOTATION=... TRACE_OVO_VIDEO_ROOT=...
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_CKPT_RUN:?Set TRACE_CKPT_RUN}" "${TRACE_OVO_ROOT:?Set TRACE_OVO_ROOT}" "${TRACE_OUT:?Set TRACE_OUT}"
export TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_PER_TASK="${TRACE_PER_TASK:-100}" TRACE_MAX_QUERY_SECONDS="${TRACE_MAX_QUERY_SECONDS:-180}"
export TRACE_OVO_ANNOTATION="${TRACE_OVO_ANNOTATION:-}" TRACE_OVO_VIDEO_ROOT="${TRACE_OVO_VIDEO_ROOT:-}"
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.lock"
flock -n 9 || { echo "Another temporal diagnostic is using $TRACE_OUT" >&2; exit 2; }
log() { echo "[$(date -Is)] $*"; }
for arm in transition after b4; do
  [[ -f "$TRACE_CKPT_RUN/train/$arm/modules.pt" && -f "$TRACE_CKPT_RUN/train/$arm/lora/adapter_config.json" ]] \
    || { echo "missing $arm checkpoint in $TRACE_CKPT_RUN" >&2; exit 2; }
done

log "[1/4] Locate OVO-Bench annotation and video root; convert (backward / real-time MC + forward REC/SSR/CRR)"
if [[ ! -f "$TRACE_OUT/ovo_all.jsonl" ]]; then
  "${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.data import convert_ovo

root, out = Path(os.environ["TRACE_OVO_ROOT"]), Path(os.environ["TRACE_OUT"])
annotations = [Path(os.environ["TRACE_OVO_ANNOTATION"])] if os.environ["TRACE_OVO_ANNOTATION"] else sorted(root.rglob("ovo_bench*.json"))
if not annotations:
    raise SystemExit(f"no ovo_bench*.json under {root}; set TRACE_OVO_ANNOTATION")
ann = annotations[0]
items = json.loads(ann.read_text())
if os.environ["TRACE_OVO_VIDEO_ROOT"]:
    candidates = [Path(os.environ["TRACE_OVO_VIDEO_ROOT"])]
else:
    candidates = [ann.parent, root, *sorted(p for p in root.rglob("*") if p.is_dir() and p.name in {"src_videos", "videos"})]
probe = items[:300]
coverage = {str(c): sum((c / i["video"]).is_file() for i in probe) / max(1, len(probe)) for c in candidates}
best = max(coverage, key=coverage.get)
print(json.dumps({"annotation": str(ann), "items": len(items), "video_root_coverage": coverage, "chosen": best}, indent=2))
if coverage[best] < 0.9:
    raise SystemExit("fewer than 90% of probed OVO videos found: set TRACE_OVO_VIDEO_ROOT (or extract the OVO videos)")
report = convert_ovo(str(ann), str(out / "ovo_all.jsonl.tmp"), best, forward=True)
(out / "ovo_all.jsonl.tmp").replace(out / "ovo_all.jsonl")
(out / "conversion.json").write_text(json.dumps({"annotation": str(ann), "video_root": best, **report}, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
fi
video_root="$("${py[@]}" -c "import json;print(json.load(open('$TRACE_OUT/conversion.json'))['video_root'])")"

log "[2/4] Fixed split: diagnostic half (used now) and frozen final half (not evaluated)"
if [[ ! -f "$TRACE_OUT/split/manifest.json" ]]; then
  "${py[@]}" -m trace_vl.cli v3 temporal-split "$TRACE_OUT/ovo_all.jsonl" --output "$TRACE_OUT/split" \
    --per-task "$TRACE_PER_TASK" --max-query-seconds "$TRACE_MAX_QUERY_SECONDS"
fi
"${py[@]}" -c "import json;m=json.load(open('$TRACE_OUT/split/manifest.json'));print('diagnostic:',m['diagnostic'])"

log "[3/4] Evaluate native and the three arms (${TRACE_GPUS} shards each; finished shards resume)"
run_eval() {
  local name="$1"; shift
  local pids=()
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval "$@" \
      --records "$TRACE_OUT/split/diagnostic.jsonl" --video-root "$video_root" --output "$TRACE_OUT/eval/$name" \
      --shard "$i/$TRACE_GPUS" --device cuda:0 > "$TRACE_OUT/eval-$name-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "eval $name failed; see $TRACE_OUT/eval-$name-shard*.log" >&2; exit 1; }; done
  log "  $name done"
}
run_eval native --config "$TRACE_CKPT_RUN/configs/transition.yaml" --mode native
for arm in transition after b4; do
  run_eval "$arm" --config "$TRACE_CKPT_RUN/configs/$arm.yaml" --mode v3 \
    --checkpoint "$TRACE_CKPT_RUN/train/$arm/modules.pt" --adapter "$TRACE_CKPT_RUN/train/$arm/lora"
done

log "[4/4] Per-category paired comparison and the predeclared decision"
"${py[@]}" -m trace_vl.cli v3 temporal-report --eval-root "$TRACE_OUT/eval" --records "$TRACE_OUT/split/diagnostic.jsonl" \
  --output "$TRACE_OUT/temporal-report.json" > /dev/null
"${py[@]}" - "$TRACE_OUT/temporal-report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
lines = [f"TEMPORAL DIAGNOSTIC DIGEST ({r['questions']} questions, {r['videos']} videos; diagnostic half only)"]
for arm, a in r["accuracy"].items():
    lines.append(f"{arm}: temporal {a['temporal']:.4f}, static {a['static']:.4f}, tokens/step {r['lm_tokens_per_step'][arm]:.2f}, per task {a['per_task']}")
for pair, p in r["paired"].items():
    lines.append(f"{pair}: temporal {p['temporal']['delta_pp']:+.2f} {p['temporal']['ci95_pp']} | static "
                 f"{p['static']['delta_pp']:+.2f} {p['static']['ci95_pp']} | interaction {p['interaction']['delta_pp']:+.2f} {p['interaction']['ci95_pp']}")
lines.append(f"DECISION continue_transition_track={r['decision']['continue_transition_track']} ({r['decision']['rule']})")
text = "\n".join(lines) + "\n"
open(sys.argv[1].replace("temporal-report.json", "temporal-digest.txt"), "w").write(text)
print(text)
PY
log "Completed: $TRACE_OUT/temporal-report.json, temporal-digest.txt"
