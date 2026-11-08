#!/usr/bin/env bash
# Pilot-003: transition trios vs after-only slot pools vs B4 pooled regions at an identical, fixed P-step budget.
# One command, resumable (completed arms/shards are skipped; inputs are checked on every start):
#   TRACE_DATA_DIR=/nas-data/vlm/yyy/tracevl-data/stage1-30s-001 \
#   TRACE_VIDEO_ROOT=/nas-data/vlm/llava/media TRACE_MODEL=/nas-data/vlm/models/qwen3.5_2b \
#   TRACE_RUNS=/nas-data/vlm/yyy/tracevl-runs/pilot-003 nohup bash scripts/run_pilot003.sh > pilot-003.log 2>&1 &
# Optional: TRACE_GPUS=4 TRACE_CLIP_SECONDS=30 TRACE_UV_BIN=uv. Decision rule: docs/v3/experiments.md §10.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_DATA_DIR:?Set TRACE_DATA_DIR (train.jsonl and val.jsonl)}"
: "${TRACE_VIDEO_ROOT:?Set TRACE_VIDEO_ROOT (base of the manifest video paths)}"
: "${TRACE_RUNS:?Set TRACE_RUNS (a NEW directory for this pilot)}"
export TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_CLIP_SECONDS="${TRACE_CLIP_SECONDS:-30}" TRACE_MODEL="${TRACE_MODEL:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
arms=(transition after b4)
mkdir -p "$TRACE_RUNS"
exec 9>"$TRACE_RUNS/.lock"
flock -n 9 || { echo "Another pilot-003 process is using $TRACE_RUNS" >&2; exit 2; }
trap 'status=$?; ((status)) && echo "Stopped (exit $status). Fix the error and rerun the same command; finished work is kept." >&2' EXIT
log() { echo "[$(date -Is)] $*"; }

log "[0/4] Preflight: data, resolved configs, media and source identity"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.pilot import preflight

print(json.dumps(preflight(os.environ["TRACE_DATA_DIR"], os.environ["TRACE_VIDEO_ROOT"], os.environ["TRACE_RUNS"],
                           Path("configs/v3"), os.environ["TRACE_MODEL"] or None, float(os.environ["TRACE_CLIP_SECONDS"]),
                           int(os.environ["TRACE_GPUS"]), Path("src/trace_vl/v3")), indent=2))
PY

log "[1/4] Train arms (identical seed and schedule; ${TRACE_GPUS} GPUs each)"
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

log "[2/4] Evaluate native and arms (${TRACE_GPUS} shards in parallel; finished shards resume)"
for arm in native "${arms[@]}"; do
  out="$TRACE_RUNS/eval/$arm"
  if [[ $arm == native ]]; then
    args=(--config "$TRACE_RUNS/configs/transition.yaml" --mode native)
  else
    args=(--config "$TRACE_RUNS/configs/$arm.yaml" --mode v3 --checkpoint "$TRACE_RUNS/train/$arm/modules.pt"
          --adapter "$TRACE_RUNS/train/$arm/lora")
  fi
  log "  $arm"
  pids=()
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval "${args[@]}" \
      --records "$TRACE_DATA_DIR/val.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$out" \
      --shard "$i/$TRACE_GPUS" --device cuda:0 > "$TRACE_RUNS/eval-$arm-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5  # the first shard writes run.json; later shards validate it instead of racing on it
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "eval $arm failed; see $TRACE_RUNS/eval-$arm-shard*.log" >&2; exit 1; }; done
done

log "[3/4] Validate coverage, score, check budgets and gate logs"
log "[4/4] Paired whole-video bootstrap and predeclared decision"
"${py[@]}" - <<'PY'
import json, os, random
from pathlib import Path
from trace_vl.v3.harness import score

runs, data = Path(os.environ["TRACE_RUNS"]), Path(os.environ["TRACE_DATA_DIR"])
gold = {r["id"]: r for r in (json.loads(s) for s in (data / "val.jsonl").read_text().splitlines() if s.strip())}
arms = ["native", "transition", "after", "b4"]
preds, scores = {}, {}
for arm in arms:
    out = runs / "eval" / arm
    rows = [json.loads(s) for p in sorted(out.glob("predictions.*.jsonl")) for s in p.read_text().splitlines() if s.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(gold):
        raise SystemExit(f"{arm}: {len(set(ids))}/{len(gold)} questions predicted (duplicates or gaps); rerun")
    if any(json.loads(p.read_text()) for p in out.glob("errors.*.json")):
        raise SystemExit(f"{arm}: evaluation errors in {out}")
    run_id = json.loads((out / "run.json").read_text())["id"]
    if any(r.get("run") != run_id or r["answer"] != gold[r["id"]]["answer"] for r in rows):
        raise SystemExit(f"{arm}: wrong run identity or gold labels")
    preds[arm] = {r["id"]: r for r in rows}
    scores[arm] = score(str(out), str(out / "score.json"))
# Budget: identical by construction; verify on the evaluated streams (all v3 arms within 0.5%).
tokens = {a: scores[a]["lm_tokens_per_step"] for a in arms[1:]}
spread = (max(tokens.values()) - min(tokens.values())) / min(tokens.values())
# Training logs: every logged P-step mean must be exactly the fixed budget.
gate = {}
for arm in arms[1:]:
    rows = [json.loads(s) for s in (runs / "train" / arm / "metrics.jsonl").read_text().splitlines() if s.strip()]
    vals = [r["p_step_tokens_mean"] for r in rows if r.get("p_step_tokens_mean") is not None]
    gate[arm] = {"updates": len(rows), "p_step_tokens_min": min(vals), "p_step_tokens_max": max(vals),
                 "cst_gate_abs_mean_final": rows[-1].get("gate_abs_mean")}

def paired(a, b):
    groups = {}
    for key, r in preds[a].items():
        o = preds[b][key]
        g = groups.setdefault(r["video_key"], [0, 0])
        g[0] += int(r["prediction"] == r["answer"]) - int(o["prediction"] == o["answer"])
        g[1] += 1
    units = list(groups.values())
    point = 100 * sum(u[0] for u in units) / sum(u[1] for u in units)
    rng, draws = random.Random(7), []
    for _ in range(4000):
        pick = rng.choices(units, k=len(units))
        draws.append(100 * sum(u[0] for u in pick) / sum(u[1] for u in pick))
    draws.sort()
    return {"delta_pp": round(point, 3), "ci95_pp": [round(draws[100], 3), round(draws[3899], 3)],
            "videos": len(units)}

pairs = {"transition_minus_b4": paired("transition", "b4"), "after_minus_b4": paired("after", "b4"),
         "transition_minus_after": paired("transition", "after"), "transition_minus_native": paired("transition", "native"),
         "after_minus_native": paired("after", "native"), "b4_minus_native": paired("b4", "native")}
def verdict(p):
    lo, hi = p["ci95_pp"]
    return "higher" if lo > 0 else "lower" if hi < 0 else "no detectable difference"
acc = {a: scores[a]["accuracy"] for a in arms}
decision = {
    "budget_matched": spread <= 0.005 and all(g["p_step_tokens_min"] == g["p_step_tokens_max"] == 20 for g in gate.values()),
    "M2_gate (transition >= b4 + 2 pp and >= 95% native)":
        pairs["transition_minus_b4"]["delta_pp"] >= 2 and acc["transition"] >= 0.95 * acc["native"],
    "surprise selection (after vs b4)": verdict(pairs["after_minus_b4"]),
    "before/delta tokens (transition vs after)": verdict(pairs["transition_minus_after"]),
}
summary = {"accuracy": acc, "ci95": {a: scores[a]["ci95"] for a in arms}, "lm_tokens_per_step": tokens,
           "budget_spread": spread, "kv_mb_mean": {a: scores[a]["kv_mb_mean"] for a in arms},
           "paired": pairs, "training_gate_logs": gate, "decision": decision,
           "note": "Predeclared rule: docs/v3/experiments.md section 10. 100-update pilot; one seed; 0-30 s MC only."}
tmp = runs / ".summary.tmp"
tmp.write_text(json.dumps(summary, indent=2) + "\n")
tmp.replace(runs / "summary.json")
print(json.dumps(summary, indent=2))
PY
log "Completed: $TRACE_RUNS/summary.json (please post it on issue #4)"
