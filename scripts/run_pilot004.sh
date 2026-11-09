#!/usr/bin/env bash
# Pilot-004: pilot-003's three arms (transition / after / b4, fixed 20-token P-steps) trained FRESH for 1,000
# updates (duration is the only change), evaluated at the predeclared checkpoints 100 / 300 / 1,000; the final
# checkpoint is the primary comparison. Selector diagnostics are logged every 10 updates (no model change).
# One command, resumable (completed arms and evaluations are skipped; inputs are checked on every start):
#   TRACE_DATA_DIR=... TRACE_VIDEO_ROOT=... TRACE_MODEL=... TRACE_RUNS=<NEW dir> bash scripts/run_pilot004.sh
# Optional: TRACE_GPUS=4 TRACE_CLIP_SECONDS=30 TRACE_UV_BIN=uv. Decision rule: docs/v3/experiments.md section 11.
# TRACE_PILOT_PREFIX=pilot005 runs pilot-005 (configs/v3/pilot005_*.yaml; same schedule and evaluation).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_DATA_DIR:?Set TRACE_DATA_DIR (train.jsonl and val.jsonl)}"
: "${TRACE_VIDEO_ROOT:?Set TRACE_VIDEO_ROOT (base of the manifest video paths)}"
: "${TRACE_RUNS:?Set TRACE_RUNS (a NEW directory for this pilot)}"
export TRACE_PILOT_PREFIX="${TRACE_PILOT_PREFIX:-pilot004}"
export TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_CLIP_SECONDS="${TRACE_CLIP_SECONDS:-30}" TRACE_MODEL="${TRACE_MODEL:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
arms=(transition after b4)
checkpoints=(100 300 1000)
mkdir -p "$TRACE_RUNS"
exec 9>"$TRACE_RUNS/.lock"
flock -n 9 || { echo "Another pilot-004 process is using $TRACE_RUNS" >&2; exit 2; }
trap 'status=$?; ((status)) && echo "Stopped (exit $status). Fix the error and rerun the same command; finished work is kept." >&2' EXIT
log() { echo "[$(date -Is)] $*"; }

log "[0/4] Preflight: data, resolved configs, media and source identity"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.pilot import preflight

report = preflight(os.environ["TRACE_DATA_DIR"], os.environ["TRACE_VIDEO_ROOT"], os.environ["TRACE_RUNS"], Path("configs/v3"),
                   os.environ["TRACE_MODEL"] or None, float(os.environ["TRACE_CLIP_SECONDS"]), int(os.environ["TRACE_GPUS"]),
                   Path("src/trace_vl/v3"), prefix=os.environ["TRACE_PILOT_PREFIX"])
identity = json.loads(Path(report["identity"]).read_text())
for arm, cfg in identity["configs"].items():
    t = cfg["train"]
    assert t["steps"] == 1000 and t["checkpoint_steps"] == [100, 300, 1000] and t["diag_every"] == 10, arm
print(json.dumps(report, indent=2))
PY

log "[1/4] Train arms (fresh, identical seed and schedule; ${TRACE_GPUS} GPUs each)"
for arm in "${arms[@]}"; do
  out="$TRACE_RUNS/train/$arm"
  if "${py[@]}" - "$out" "$arm" "$TRACE_RUNS/identity.json" <<'PY'
import json, sys
from pathlib import Path
from trace_vl.v3.pilot import training_complete

out, arm, identity = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text())
ok, reason = training_complete(out, identity["configs"][arm], identity["configs"][arm]["train"]["steps"])
if not ok and out.exists():
    print(f"  {arm}: not complete ({reason})", flush=True)
sys.exit(0 if ok else 1)
PY
  then log "  $arm: complete, skipped"; continue; fi
  if [[ -e "$out" ]]; then
    stale="$out.incomplete-$(date +%s)"; mv "$out" "$stale"; log "  $arm: incomplete previous attempt moved to $stale"
  fi
  log "  $arm: training"
  "$uv_bin" run --locked --no-sync torchrun --standalone --nproc_per_node="$TRACE_GPUS" -m trace_vl.cli v3 train \
    --config "$TRACE_RUNS/configs/$arm.yaml" --records "$TRACE_DATA_DIR/train.jsonl" \
    --video-root "$TRACE_VIDEO_ROOT" --output "$out" --clip-seconds "$TRACE_CLIP_SECONDS"
done

run_eval() {  # name, extra eval args...
  local name="$1"; shift
  local out="$TRACE_RUNS/eval/$name" pids=()
  log "  $name"
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval "$@" \
      --records "$TRACE_DATA_DIR/val.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$out" \
      --shard "$i/$TRACE_GPUS" --device cuda:0 > "$TRACE_RUNS/eval-${name//\//-}-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5  # the first shard writes run.json; later shards validate it instead of racing on it
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "eval $name failed; see $TRACE_RUNS/eval-${name//\//-}-shard*.log" >&2; exit 1; }; done
}

log "[2/4] Evaluate native and every arm at every predeclared checkpoint (${TRACE_GPUS} shards; finished shards resume)"
run_eval native --config "$TRACE_RUNS/configs/transition.yaml" --mode native
for step in "${checkpoints[@]}"; do
  tag="$(printf 'step-%05d' "$step")"
  for arm in "${arms[@]}"; do
    snap="$TRACE_RUNS/train/$arm/checkpoints/$tag"
    run_eval "$arm/$tag" --config "$TRACE_RUNS/configs/$arm.yaml" --mode v3 --checkpoint "$snap/modules.pt" --adapter "$snap/lora"
  done
done

log "[3/4] Validate coverage, run identity and per-question budgets; score"
log "[4/4] Paired whole-video bootstrap, predeclared decision, selector-diagnostic trajectories"
"${py[@]}" - <<'PY'
import json, math, os
from pathlib import Path
from trace_vl.v3.diagnose import paired_delta
from trace_vl.v3.harness import score

runs, data = Path(os.environ["TRACE_RUNS"]), Path(os.environ["TRACE_DATA_DIR"])
gold = {r["id"]: r for r in (json.loads(s) for s in (data / "val.jsonl").read_text().splitlines() if s.strip())}
arms, steps = ["transition", "after", "b4"], [100, 300, 1000]

def load(name):
    out = runs / "eval" / name
    rows = [json.loads(s) for p in sorted(out.glob("predictions.*.jsonl")) for s in p.read_text().splitlines() if s.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(gold):
        raise SystemExit(f"{name}: {len(set(ids))}/{len(gold)} questions predicted (duplicates or gaps); rerun")
    if any(json.loads(p.read_text()) for p in out.glob("errors.*.json")):
        raise SystemExit(f"{name}: evaluation errors in {out}")
    run_id = json.loads((out / "run.json").read_text())["id"]
    if any(r.get("run") != run_id or r["answer"] != gold[r["id"]]["answer"] for r in rows):
        raise SystemExit(f"{name}: wrong run identity or gold labels")
    return {r["id"]: r for r in rows}, score(str(out), str(out / "score.json"))

native, native_score = load("native")
per_step = {}
for step in steps:
    tag = f"step-{step:05d}"
    preds, scores = {"native": native}, {"native": native_score}
    for arm in arms:
        preds[arm], scores[arm] = load(f"{arm}/{tag}")
    for arm in arms[1:]:  # identical per-question budgets (fixed 20-token P-steps, fixed I-step schedule)
        for qid, row in preds["transition"].items():
            for field in ("steps", "i_step_fraction", "lm_tokens_per_step", "kv_tokens"):
                if not (math.isfinite(row[field]) and abs(row[field] - preds[arm][qid][field]) < 1e-6):
                    raise SystemExit(f"{tag} {arm}/{qid}: {field} differs from transition")
    acc = {a: scores[a]["accuracy"] for a in scores}
    pairs = {f"{a}_minus_{b}": paired_delta(preds, a, b) for a, b in
             (("transition", "b4"), ("after", "b4"), ("transition", "after"), ("transition", "native"),
              ("after", "native"), ("b4", "native"))}
    per_step[step] = {"accuracy": acc, "ci95": {a: scores[a]["ci95"] for a in scores}, "paired": pairs,
                      "lm_tokens_per_step": {a: scores[a]["lm_tokens_per_step"] for a in arms}}

final = per_step[1000]
def continuation(arm):
    p = final["paired"][f"{arm}_minus_b4"]
    return p["delta_pp"] >= 0 and p["ci95_pp"][0] > -1.0
decision = {
    "primary_checkpoint": 1000,
    "M2_gate (original; transition >= b4 + 2 pp and >= 95% native, final checkpoint)":
        final["paired"]["transition_minus_b4"]["delta_pp"] >= 2 and final["accuracy"]["transition"] >= 0.95 * final["accuracy"]["native"],
    "exploratory continuation (slot arm >= b4, CI lower bound > -1 pp, final checkpoint)":
        {arm: continuation(arm) for arm in ("transition", "after")},
}
diagnostics = {}
for arm in arms:
    rows = [json.loads(s) for s in (runs / "train" / arm / "metrics.jsonl").read_text().splitlines() if s.strip()]
    keys = sorted({k for r in rows for k in r if k.startswith("diag_")})
    diag_rows = [r for r in rows if any(k.startswith("diag_") for k in r)]
    diagnostics[arm] = {k: [[r["step"], r[k]] for r in diag_rows if k in r] for k in keys}
    diagnostics[arm]["ce"] = [[r["step"], r["ce"]] for r in rows if r["step"] in (1, 100, 300, 1000) or r["step"] % 100 == 0]
(runs / "selector-diagnostics.json").write_text(json.dumps(diagnostics) + "\n")
summary = {"per_checkpoint": per_step, "decision": decision,
           "note": "Fresh 1,000-update training (cosine schedule over 1,000 updates; the first 100 updates are not "
                   "pilot-003's schedule). Final checkpoint is primary; 100/300 are trajectory points, not selected. "
                   "One seed; 0-30 s multiple choice only. Video-cluster bootstrap is the primary uncertainty."}
tmp = runs / ".summary.tmp"
tmp.write_text(json.dumps(summary, indent=2) + "\n")
tmp.replace(runs / "summary.json")
print(json.dumps(summary, indent=2))
PY
log "Completed: $TRACE_RUNS/summary.json and $TRACE_RUNS/selector-diagnostics.json"
