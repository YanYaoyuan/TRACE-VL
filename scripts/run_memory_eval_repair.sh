#!/usr/bin/env bash
# Evaluation-only budget repair of the 448adf1 memory pilot (experiments.md §14.1). No training.
#   TRACE_PILOT_RUN=<448adf1 pilot dir> TRACE_VIDEO_ROOT=... TRACE_OUT=<NEW dir> bash scripts/run_memory_eval_repair.sh
# 1) analysis of the EXISTING predictions (paired S2-S1, TKM/FIFO agreement); 2) budget pre-check with memory.fill
# exact on the 2 shortest + 2 longest manifest streams (stop unless KV spread <= 2%); 3) all eight evaluations with the
# frozen S1/S2 checkpoints on the identical 40-stream manifest, with the memory audit; 4) report and analysis.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_PILOT_RUN:?}" "${TRACE_VIDEO_ROOT:?}" "${TRACE_OUT:?}"
export TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_MODEL="${TRACE_MODEL:-}"
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.lock"
flock -n 9 || { echo "Another repair evaluation is using $TRACE_OUT" >&2; exit 2; }
log() { echo "[$(date -Is)] $*"; }
export TRACE_S2="$TRACE_PILOT_RUN/train" TRACE_MANIFEST="$TRACE_PILOT_RUN/data/streams/dev_selected.jsonl"
# Stage-1 B4 checkpoint (verified below against the pilot's recorded initialization identity)
export TRACE_S1="${TRACE_S1_CKPT:?Set TRACE_S1_CKPT to the Stage-1 B4 checkpoint dir (train/b4 of the pilot-005 run)}"
export TRACE_CFG="$TRACE_OUT/memory_pilot_b4_exactfill.yaml"
{ echo "extends: $root/configs/v3/memory_pilot_b4_exactfill.yaml"
  [[ -n "$TRACE_MODEL" ]] && printf 'backbone:\n  name: %s\n  local_files_only: true\n' "$TRACE_MODEL"; true; } > "$TRACE_CFG"

log "[0/4] Verify frozen inputs: Stage-2 completion record, Stage-1 identity, manifest hash"
"${py[@]}" - <<'PY'
import hashlib, json, os
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.memory_pilot import STAGE2_CHECKPOINT_STEPS, init_identity
from trace_vl.v3.pilot import training_complete
run, out = Path(os.environ["TRACE_PILOT_RUN"]), Path(os.environ["TRACE_OUT"])
identity = json.loads((run / "train-identity.expected.json").read_text())
cfg_old, _ = load_v3_config(run / "memory_pilot_b4.yaml")
ok, reason = training_complete(run / "train", cfg_old.to_dict(), identity["steps"], extra=identity,
                               checkpoint_steps=STAGE2_CHECKPOINT_STEPS)
assert ok, f"Stage-2 checkpoint not verified: {reason}"
assert init_identity(os.environ["TRACE_S1"]) == identity["init"], "Stage-1 checkpoint differs from the pilot's initialization"
cfg_new, _ = load_v3_config(os.environ["TRACE_CFG"])
a, b = cfg_old.to_dict(), cfg_new.to_dict()
assert b["memory"].pop("fill") == "exact" and a["memory"].pop("fill") == "groups"
assert a == b, "evaluation config differs from the pilot beyond memory.fill"
manifest = Path(os.environ["TRACE_MANIFEST"])
record = {"pilot_run": str(run), "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
          "stage2_identity": identity, "protocol_change": "memory.fill groups -> exact (evaluation only; training used groups)"}
target = out / "inputs.json"
if target.exists() and json.loads(target.read_text()) != record:
    raise SystemExit(f"{out} was started with different inputs; use a new TRACE_OUT")
target.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

log "[1/4] Analysis of the existing (invalid) pilot predictions: paired S2-S1, TKM/FIFO agreement"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.memory_pilot import memory_analysis
out = Path(os.environ["TRACE_OUT"])
a = memory_analysis(Path(os.environ["TRACE_PILOT_RUN"]) / "eval", 16.0)
(out / "analysis-448adf1.json").write_text(json.dumps(a, indent=2) + "\n")
print(json.dumps(a, indent=2))
PY

run_eval() {  # name, checkpoint dir, manifest, output root, extra args...
  local name="$1" ckpt="$2" manifest="$3" eroot="$4"; shift 4
  local pids=()
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval-stage2 --config "$TRACE_CFG" \
      --streams "$manifest" --video-root "$TRACE_VIDEO_ROOT" --output "$eroot/$name" --mode v3 \
      --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev --shard "$i/$TRACE_GPUS" --device cuda:0 "$@" \
      > "$TRACE_OUT/$(basename "$eroot")-$name-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "eval $name failed; see $TRACE_OUT/*-$name-shard*.log" >&2; exit 1; }; done
  CUDA_VISIBLE_DEVICES=0 "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval-stage2 --config "$TRACE_CFG" \
    --streams "$manifest" --video-root "$TRACE_VIDEO_ROOT" --output "$eroot/$name" --mode v3 \
    --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev "$@" > /dev/null
  log "  $name done"
}

log "[2/4] Budget pre-check (exact fill): 2 shortest + 2 longest manifest streams, TKM / mass / FIFO, audit on"
"${py[@]}" -c "from trace_vl.v3.memory_pilot import precheck_streams; print(precheck_streams('$TRACE_MANIFEST', '$TRACE_OUT/precheck_streams.jsonl'))"
for policy in tkm mass fifo; do run_eval "s2_$policy" "$TRACE_S2" "$TRACE_OUT/precheck_streams.jsonl" "$TRACE_OUT/precheck" --policy "$policy" --audit; done
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.memory_pilot import budget_precheck
out = Path(os.environ["TRACE_OUT"])
r = budget_precheck(out / "precheck")
(out / "precheck.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=2))
if not r["passed"]:
    raise SystemExit("budget pre-check failed (KV spread above 2%): not running the complete evaluation")
PY

log "[3/4] Complete evaluation: frozen checkpoints, identical 40-stream manifest, exact fill, audit on"
run_eval s2_tkm "$TRACE_S2" "$TRACE_MANIFEST" "$TRACE_OUT/eval" --policy tkm --audit
run_eval s2_mass "$TRACE_S2" "$TRACE_MANIFEST" "$TRACE_OUT/eval" --policy mass --audit
run_eval s2_fifo "$TRACE_S2" "$TRACE_MANIFEST" "$TRACE_OUT/eval" --policy fifo --audit
run_eval s2_full "$TRACE_S2" "$TRACE_MANIFEST" "$TRACE_OUT/eval" --policy full
run_eval s2_tkm_trainsched "$TRACE_S2" "$TRACE_MANIFEST" "$TRACE_OUT/eval" --policy tkm --evict-schedule train --audit
run_eval s1_mass "$TRACE_S1" "$TRACE_MANIFEST" "$TRACE_OUT/eval" --policy mass --audit
run_eval s1_fifo "$TRACE_S1" "$TRACE_MANIFEST" "$TRACE_OUT/eval" --policy fifo --audit
run_eval s1_full "$TRACE_S1" "$TRACE_MANIFEST" "$TRACE_OUT/eval" --policy full

log "[4/4] Report (unchanged decision rule) and analysis"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.memory_pilot import EXPECTED_RUNS, expected_questions, memory_analysis, memory_report
out = Path(os.environ["TRACE_OUT"])
cfg, _ = load_v3_config(os.environ["TRACE_CFG"])
ids = set(expected_questions(os.environ["TRACE_MANIFEST"], "dev", None, cfg))
r = memory_report(str(out / "eval"), cfg.memory.window_seconds, expected_runs=EXPECTED_RUNS, expected_ids=ids)
a = memory_analysis(out / "eval", cfg.memory.window_seconds)
(out / "memory-report.json").write_text(json.dumps(r, indent=2) + "\n")
(out / "analysis.json").write_text(json.dumps(a, indent=2) + "\n")
lines = ["MEMORY EVAL REPAIR DIGEST (frozen 448adf1 checkpoints; memory.fill exact; evaluation only)"]
for name, s in r["runs"].items():
    lines.append(f"{name}: all {s['all']['accuracy_pct']} {s['all']['ci95_pct']} | beyond {s['beyond_window']['accuracy_pct']} "
                 f"{s['beyond_window']['ci95_pct']} (n {s['beyond_window']['n']}) | KV {s['kv_tokens_mean']} | evidence kept "
                 f"{s['evidence_kept_beyond_window']} | audit {a['audit'].get(name)}")
for pair, p in r["paired"].items():
    lines.append(f"{pair}: beyond {p['beyond_window']['delta_pp']} {p['beyond_window']['ci95_pp']} | all {p['all']['delta_pp']} {p['all']['ci95_pp']}")
for policy, p in a["stage2_minus_stage1"].items():
    lines.append(f"s2-s1 {policy}: beyond {p['beyond_window']['delta_pp']} {p['beyond_window']['ci95_pp']} | all {p['all']['delta_pp']} {p['all']['ci95_pp']}")
lines.append(f"agreement {json.dumps(a['agreement'])}")
lines.append(f"DECISION {json.dumps(r['decision'])}")
text = "\n".join(lines) + "\n"
(out / "repair-digest.txt").write_text(text)
print(text)
PY
log "Completed: $TRACE_OUT/repair-digest.txt"
