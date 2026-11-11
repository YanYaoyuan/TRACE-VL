#!/usr/bin/env bash
# Stage-2 transfer check on OVO-Bench (experiments.md §14.3; src/trace_vl/v3/ovo_transfer.py). EVALUATION ONLY.
#   TRACE_STAGE=inventory  CPU only: verify the §13 split, re-convert the official annotation, build the
#                          ovo-transfer-dev-v2 manifest (diagnostic hash half) and inventory, plus the v1 comparison. Submit inventory.json before any GPU work.
#   TRACE_STAGE=flow       small GPU flow check: S1/S2 FIFO on the 2 shortest + 2 longest streams (coverage,
#                          no future frames, identical schedules, wall-time estimate).
#   TRACE_STAGE=full       the four predeclared arms (S1/S2 x FIFO/full) on every stream, then the report.
# Inputs are only read: TRACE_DIAG_RUN (the 9653e2a temporal diagnostic), TRACE_PILOT_RUN (448adf1 Stage-2
# checkpoint), TRACE_S1_CKPT (pilot-005 B4). Everything is written into TRACE_OUT. The frozen OVO final half is
# never read except for the video-ID disjointness check.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_STAGE:?}" "${TRACE_DIAG_RUN:?}" "${TRACE_PILOT_RUN:?}" "${TRACE_S1_CKPT:?}" "${TRACE_OUT:?}"
export TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_MODEL="${TRACE_MODEL:-}" TRACE_ACCEPT_INVENTORY="${TRACE_ACCEPT_INVENTORY:-0}"
export TRACE_EVALREPAIR_RUN="${TRACE_EVALREPAIR_RUN:-}"
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.lock"
flock -n 9 || { echo "Another OVO transfer stage is using $TRACE_OUT" >&2; exit 2; }
log() { echo "[$(date -Is)] $*"; }
export TRACE_S2="$TRACE_PILOT_RUN/train" TRACE_S1="$TRACE_S1_CKPT"
export TRACE_CFG="$TRACE_OUT/memory_pilot_b4_exactfill.yaml" TRACE_T="$TRACE_OUT/transfer"
{ echo "extends: $root/configs/v3/memory_pilot_b4_exactfill.yaml"
  [[ -n "$TRACE_MODEL" ]] && printf 'backbone:\n  name: %s\n  local_files_only: true\n' "$TRACE_MODEL"; true; } > "$TRACE_CFG"

# ---------------------------------------------------------------------------------------------------- inventory
if [[ "$TRACE_STAGE" == inventory ]]; then
  log "[inventory 1/2] Verify the §13 split and re-convert the official OVO annotation (MC only)"
  "${py[@]}" - <<'PY'
import hashlib, json, os
from pathlib import Path
from trace_vl.v3.data import convert_ovo, read_jsonl
diag, out = Path(os.environ["TRACE_DIAG_RUN"]), Path(os.environ["TRACE_OUT"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
split = json.loads((diag / "split/manifest.json").read_text())
assert sha(diag / "split/diagnostic.jsonl") == split["diagnostic_sha256"], "diagnostic.jsonl differs from its manifest"
assert sha(diag / "split/final_frozen.jsonl") == split["final_frozen_sha256"], "final_frozen.jsonl differs from its manifest"
conv = json.loads((diag / "conversion.json").read_text())
report = convert_ovo(conv["annotation"], str(out / "ovo_mc.jsonl"), conv["video_root"], forward=False)
mc = {r["id"]: r for r in read_jsonl(out / "ovo_mc.jsonl")}
old = {r["id"]: r for r in read_jsonl(diag / "ovo_all.jsonl") if r.get("options") is not None}
assert mc == old, f"re-conversion differs from the §13 conversion ({len(mc)} vs {len(old)} MC records)"
diag_videos = {r["video"] for r in read_jsonl(diag / "split/diagnostic.jsonl")}
record = {"diag_run": str(diag), "annotation": conv["annotation"], "annotation_sha256": sha(conv["annotation"]),
          "video_root": conv["video_root"], "conversion": report, "ovo_mc_sha256": sha(out / "ovo_mc.jsonl"),
          "split_manifest": split, "diagnostic_videos": len(diag_videos)}
(out / "source.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "split_manifest"}, indent=2))
assert len(diag_videos) == 200, f"expected the 200 §13 diagnostic videos, found {len(diag_videos)}"
PY
  log "[inventory 2/2] ovo-transfer-dev-v2 (every video of the §13 diagnostic hash half, questions <= 600 s), the v1 comparison and per-half task counts"
  "${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.ovo_transfer import build_transfer, half_counts
out, diag = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_DIAG_RUN"])
cfg, _ = load_v3_config(os.environ["TRACE_CFG"])
src = json.loads((out / "source.json").read_text())
seed, fraction = src["split_manifest"]["seed"], src["split_manifest"]["diagnostic_fraction"]
common = dict(step_seconds=cfg.stream.step_seconds, gap=cfg.stream.frame_gap, max_seconds=cfg.stage2.max_seconds,
              max_query_seconds=cfg.stage2.max_seconds, seed=seed, fraction=fraction)
args = (str(out / "ovo_mc.jsonl"), str(diag / "split/diagnostic.jsonl"), str(diag / "split/final_frozen.jsonl"), src["video_root"])
inv = build_transfer(*args, os.environ["TRACE_T"], population="diagnostic-half", **common)
old = build_transfer(*args, str(out / "transfer-v1-comparison"), population="sampled-200", **common)
halves = half_counts(str(out / "ovo_mc.jsonl"), seed, fraction)
(out / "half-counts.json").write_text(json.dumps(halves, indent=2) + "\n")

def family(scope, tasks):
    return {t: halves.get(scope, {}).get(t, {"questions": 0, "videos": 0}) for t in tasks}

lines = ["OVO TRANSFER INVENTORY v2 (CPU only; no model run). Thresholds unchanged: >=150 backward q, >=30 backward videos, max share 10%",
         f"MC conversion {src['conversion']}",
         "Backward tasks per scope (questions / videos), before any eligibility filter:"]
for scope in ("all", "diagnostic", "final"):
    lines.append(f"  {scope}: {json.dumps(family(scope, ('EPM', 'ASI', 'HLD')))}")
lines.append("Real-time per scope: " + json.dumps({sc: sum(v['questions'] for t, v in halves.get(sc, {}).items() if t not in ('EPM', 'ASI', 'HLD')) for sc in ('all', 'diagnostic', 'final')}))
for name, x in (("v1 (200 sampled videos)", old), ("v2 (diagnostic hash half)", inv)):
    b, r = x["backward"], x["realtime"]
    lines += [f"== {x['protocol']} {name}",
              f"population videos {x['diagnostic_videos']} | of the 200 sampled: {x['sampled_200_videos_in_population']} | overlap with frozen final {x['overlap_with_frozen_final']} | hashing to final {x['diagnostic_videos_hashing_to_final']} | unreadable {x['videos_unreadable']}",
              f"in scope {x['in_scope_questions']} {x['in_scope_per_task']}",
              f"excluded {x['excluded']}",
              f"eligible {x['eligible_questions']} q on {x['streams']} streams; stream seconds {x['stream_seconds']} total {x['stream_seconds_total']}",
              f"backward {b['questions']} q / {b['videos']} videos, max video share {b['max_video_share']}, effective videos {b['effective_videos']}, query {b['query_time_s']}",
              *[f"  {t}: {v}" for t, v in b["per_task"].items()],
              f"realtime {r['questions']} q / {r['videos']} videos",
              *[f"  {t}: {v}" for t, v in r["per_task"].items()],
              f"ADEQUACY {json.dumps(x['adequacy'])}"]
lines.append(f"GPU stages use {inv['protocol']} only (transfer/); v1 is kept for comparison and is not evaluated.")
text = "\n".join(lines) + "\n"
(out / "inventory-digest.txt").write_text(text)
print(text)
PY
  log "Inventory done. Submit $TRACE_OUT/inventory-digest.txt, transfer/inventory.json, transfer-v1-comparison/inventory.json, half-counts.json and source.json; no GPU work yet."
  exit 0
fi

# ---------------------------------------------------------------------------------------------------- GPU stages
[[ -f "$TRACE_T/inventory.json" ]] || { echo "Run TRACE_STAGE=inventory first" >&2; exit 2; }
log "[0] Verify the inventory, the frozen checkpoints and record provenance"
"${py[@]}" - <<'PY'
import hashlib, json, os
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.memory_pilot import STAGE2_CHECKPOINT_STEPS, init_identity
from trace_vl.v3.pilot import training_complete
out, t, run = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_T"]), Path(os.environ["TRACE_PILOT_RUN"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inv = json.loads((t / "inventory.json").read_text())
assert sha(t / "streams.jsonl") == inv["streams_sha256"] and sha(t / "questions.jsonl") == inv["questions_sha256"], "manifest changed"
if not inv["adequacy"]["adequate"] and os.environ["TRACE_ACCEPT_INVENTORY"] != "1":
    raise SystemExit(f"inventory not adequate {inv['adequacy']['problems']}: report it first; "
                     "GPU work only with TRACE_ACCEPT_INVENTORY=1 after the owner accepts it")
identity = json.loads((run / "train-identity.expected.json").read_text())
old_config = json.loads((run / "expected.json").read_text())["config"]
ok, reason = training_complete(run / "train", old_config, identity["steps"], extra=identity, checkpoint_steps=STAGE2_CHECKPOINT_STEPS)
assert ok, f"Stage-2 checkpoint not verified: {reason}"
assert init_identity(os.environ["TRACE_S1"]) == identity["init"], "Stage-1 checkpoint differs from the pilot's initialization"
cfg, _ = load_v3_config(os.environ["TRACE_CFG"])
resolved = cfg.to_dict()
b = json.loads(json.dumps(resolved))
assert b["memory"].pop("fill") == "exact"
old_config["memory"].pop("fill", None)
assert old_config == b, "evaluation config differs from the pilot beyond memory.fill"
baseline = {}
if os.environ["TRACE_EVALREPAIR_RUN"]:
    er = Path(os.environ["TRACE_EVALREPAIR_RUN"])
    baseline = {"run": str(er), **{f"{a}_predictions_sha256": sha(er / "eval" / a / "predictions.jsonl")
                                   for a in ("s1_fifo", "s2_fifo", "s1_full", "s2_full") if (er / "eval" / a / "predictions.jsonl").is_file()}}
record = {"protocol": inv["protocol"], "streams_sha256": inv["streams_sha256"], "questions_sha256": inv["questions_sha256"],
          "inventory_adequate": inv["adequacy"]["adequate"], "accepted_inventory_override": os.environ["TRACE_ACCEPT_INVENTORY"] == "1",
          "config_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
          "stage2": {"dir": str(run / "train"), "identity": identity, "modules_sha256": sha(run / "train/modules.pt")},
          "stage1": {"dir": os.environ["TRACE_S1"], "identity": init_identity(os.environ["TRACE_S1"])},
          "dev_baseline_provenance": baseline}
target = out / "inputs.json"
if target.exists() and json.loads(target.read_text()) != record:
    raise SystemExit(f"{out} was started with different inputs; use a new TRACE_OUT")
target.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k not in ("stage2", "stage1")}, indent=2))
PY
video_root="$("${py[@]}" -c "import json;print(json.load(open('$TRACE_OUT/source.json'))['video_root'])")"

run_eval() {  # name, checkpoint dir, streams, output root, extra args...
  local name="$1" ckpt="$2" streams="$3" eroot="$4"; shift 4
  local pids=()
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval-stage2 --config "$TRACE_CFG" \
      --streams "$streams" --video-root "$video_root" --output "$eroot/$name" --mode v3 \
      --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev --shard "$i/$TRACE_GPUS" --device cuda:0 "$@" \
      > "$TRACE_OUT/$(basename "$eroot")-$name-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "eval $name failed; see $TRACE_OUT/*-$name-shard*.log" >&2; exit 1; }; done
  CUDA_VISIBLE_DEVICES=0 "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval-stage2 --config "$TRACE_CFG" \
    --streams "$streams" --video-root "$video_root" --output "$eroot/$name" --mode v3 \
    --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev "$@" > /dev/null
  log "  $name done"
}

if [[ "$TRACE_STAGE" == flow ]]; then
  log "[flow 1/2] S1/S2 FIFO on the 2 shortest + 2 longest transfer streams"
  "${py[@]}" -c "from trace_vl.v3.memory_pilot import precheck_streams; print(precheck_streams('$TRACE_T/streams.jsonl', '$TRACE_OUT/flow_streams.jsonl'))"
  run_eval s1_fifo "$TRACE_S1" "$TRACE_OUT/flow_streams.jsonl" "$TRACE_OUT/flow" --policy fifo
  run_eval s2_fifo "$TRACE_S2" "$TRACE_OUT/flow_streams.jsonl" "$TRACE_OUT/flow" --policy fifo
  log "[flow 2/2] Coverage, no future frames, identical schedules, wall-time estimate"
  "${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.ovo_transfer import flow_check
out, t = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_T"])
r = flow_check(out / "flow", out / "flow_streams.jsonl")
inv = json.loads((t / "inventory.json").read_text())
rate = max(v for v in r["wall_seconds_per_stream_second_per_gpu"].values() if v)
gpus = int(os.environ["TRACE_GPUS"])
r["estimate_hours_fifo_arm"] = round(rate * inv["stream_seconds_total"] / gpus / 3600, 2)
r["estimate_note"] = "per FIFO arm on all GPUs; the two full-history arms are slower (KV grows with stream length)"
(out / "flow.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=2))
if not r["passed"]:
    raise SystemExit(f"flow check failed: {r['problems']}")
PY
  log "Flow check passed. Submit flow.json and inputs.json; then TRACE_STAGE=full."
  exit 0
fi

[[ "$TRACE_STAGE" == full ]] || { echo "TRACE_STAGE must be inventory, flow or full" >&2; exit 2; }
"${py[@]}" -c "import json,sys; sys.exit(0 if json.load(open('$TRACE_OUT/flow.json'))['passed'] else 1)" 2>/dev/null \
  || { echo "Run TRACE_STAGE=flow first (it must pass)" >&2; exit 2; }
log "[full 1/2] Four predeclared arms on every transfer stream (frozen checkpoints, exact fill)"
run_eval s1_fifo "$TRACE_S1" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" --policy fifo
run_eval s2_fifo "$TRACE_S2" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" --policy fifo
run_eval s1_full "$TRACE_S1" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" --policy full
run_eval s2_full "$TRACE_S2" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" --policy full
log "[full 2/2] Report and the predeclared decision"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.ovo_transfer import transfer_report
out, t = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_T"])
inv = json.loads((t / "inventory.json").read_text())
r = transfer_report(out / "eval", t / "questions.jsonl", inventory=inv)
(out / "transfer-report.json").write_text(json.dumps(r, indent=2) + "\n")
lines = [f"OVO TRANSFER DIGEST ({r['protocol']}; external-dataset development evaluation; frozen checkpoints)"]
if r["coverage_problems"]:
    lines.append(f"COVERAGE PROBLEMS {r['coverage_problems']}")
for arm, acc in r.get("accuracy_pct", {}).items():
    lines.append(f"{arm}: backward {acc['backward']} (task-macro {acc['backward_task_macro']}) | realtime {acc['realtime']} | all {acc['all']} | KV {r['kv_tokens_mean'][arm]}")
for pair, p in r.get("paired", {}).items():
    lines.append(f"{pair}: backward {p['backward']['delta_pp']} {p['backward']['ci95_pp']} (n {p['backward']['n']}, {p['backward']['videos']} videos) | "
                 f"macro {p['backward_task_macro']['delta_pp']} {p['backward_task_macro']['ci95_pp']} | realtime {p['realtime']['delta_pp']} "
                 f"{p['realtime']['ci95_pp']} | all {p['all']['delta_pp']} {p['all']['ci95_pp']}")
    lines += [f"  {task}: {v['delta_pp']} {v['ci95_pp']} (n {v['n']})" for task, v in p["per_task"].items()]
lines.append(f"DECISION {json.dumps(r['decision'])}")
text = "\n".join(lines) + "\n"
(out / "transfer-digest.txt").write_text(text)
print(text)
PY
log "Completed: $TRACE_OUT/transfer-digest.txt"
