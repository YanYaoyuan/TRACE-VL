#!/usr/bin/env bash
# Recent-window control on the frozen OVO transfer population (experiments.md §14.4; src/trace_vl/v3/recent_window.py).
# EVALUATION ONLY: frozen S1/S2 checkpoints, the frozen 688-question ovo-transfer-dev-v2 manifest of the completed
# transfer run, whose four arms are reused (not rerun). ONE pass, advancing automatically when each check passes:
#   1. verify  manifest hashes = the transfer run's, complete reused fifo predictions, frozen checkpoints, config
#   2. flow    S1/S2 fresh16 on the transfer run's 4 flow streams with the A-B-A isolation check (stops, exit 4)
#   3. six evaluations: {S1,S2} x {stream_window, fresh16, fresh4}
#   4. report  coverage/protocol checks, diagnostic contrasts, the fifo-vs-fresh16 interaction
# Finished evaluation shards resume on re-run. The §14.3 decision is not changed by anything here.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_TRANSFER_RUN:?}" "${TRACE_PILOT_RUN:?}" "${TRACE_S1_CKPT:?}" "${TRACE_OUT:?}"
export TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_MODEL="${TRACE_MODEL:-}"
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.lock"
flock -n 9 || { echo "Another recent-window run is using $TRACE_OUT" >&2; exit 2; }
log() { echo "[$(date -Is)] $*"; }
export TRACE_S2="$TRACE_PILOT_RUN/train" TRACE_S1="$TRACE_S1_CKPT" TRACE_T="$TRACE_TRANSFER_RUN/transfer"
export TRACE_CFG="$TRACE_OUT/memory_pilot_b4_exactfill.yaml"
{ echo "extends: $root/configs/v3/memory_pilot_b4_exactfill.yaml"
  [[ -n "$TRACE_MODEL" ]] && printf 'backbone:\n  name: %s\n  local_files_only: true\n' "$TRACE_MODEL"; true; } > "$TRACE_CFG"

log "[1/4 verify] Frozen manifest, reused fifo arms, frozen checkpoints, config"
"${py[@]}" - <<'PY'
import hashlib, json, os
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.data import read_jsonl
from trace_vl.v3.memory_pilot import STAGE2_CHECKPOINT_STEPS, init_identity
from trace_vl.v3.pilot import training_complete
out, tr, run = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_TRANSFER_RUN"]), Path(os.environ["TRACE_PILOT_RUN"])
t = tr / "transfer"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inv, prior = json.loads((t / "inventory.json").read_text()), json.loads((tr / "inputs.json").read_text())
assert inv["protocol"] == "ovo-transfer-dev-v2" and inv["adequacy"]["adequate"]
assert sha(t / "streams.jsonl") == inv["streams_sha256"] == prior["streams_sha256"], "streams manifest differs from the transfer run"
assert sha(t / "questions.jsonl") == inv["questions_sha256"] == prior["questions_sha256"], "questions differ from the transfer run"
ids = {q["id"] for q in read_jsonl(t / "questions.jsonl")}
reused = {}
for arm in ("s1_fifo", "s2_fifo"):
    rows = read_jsonl(tr / "eval" / arm / "predictions.jsonl")
    assert {r["id"] for r in rows} == ids and len(rows) == len(ids), f"reused {arm} predictions do not cover the manifest"
    reused[arm] = sha(tr / "eval" / arm / "predictions.jsonl")
report = json.loads((tr / "transfer-report.json").read_text())
assert not report["coverage_problems"], "the transfer run reported coverage problems"
identity = json.loads((run / "train-identity.expected.json").read_text())
old_config = json.loads((run / "expected.json").read_text())["config"]
ok, reason = training_complete(run / "train", old_config, identity["steps"], extra=identity, checkpoint_steps=STAGE2_CHECKPOINT_STEPS)
assert ok, f"Stage-2 checkpoint not verified: {reason}"
assert init_identity(os.environ["TRACE_S1"]) == identity["init"], "Stage-1 checkpoint differs from the pilot's initialization"
cfg, _ = load_v3_config(os.environ["TRACE_CFG"])
resolved = cfg.to_dict()
config_sha = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
assert config_sha == prior["config_sha256"], "evaluation config differs from the transfer run's"
assert prior["stage1"]["identity"] == init_identity(os.environ["TRACE_S1"]) and prior["stage2"]["identity"] == identity
assert cfg.memory.window_seconds == 16.0 and cfg.stream.step_seconds == 1.0, "fresh16 assumes a 16 s window of 1 s steps"
record = {"protocol": "recent-window-control-v1 on ovo-transfer-dev-v2", "transfer_run": str(tr), "questions": len(ids),
          "streams_sha256": inv["streams_sha256"], "questions_sha256": inv["questions_sha256"], "config_sha256": config_sha,
          "stage2": prior["stage2"], "stage1": prior["stage1"], "reused_fifo_predictions_sha256": reused,
          "transfer_decision": report["decision"]["status"], "video_root": json.loads((tr / "source.json").read_text())["video_root"]}
target = out / "inputs.json"
if target.exists() and json.loads(target.read_text()) != record:
    raise SystemExit(f"{out} was started with different inputs; use a new TRACE_OUT")
target.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k not in ("stage2", "stage1")}, indent=2))
PY
video_root="$("${py[@]}" -c "import json;print(json.load(open('$TRACE_OUT/inputs.json'))['video_root'])")"
[[ -d "$video_root" ]] || { echo "media overlay of the transfer run is missing: $video_root" >&2; exit 2; }

run_cli() {  # name, checkpoint dir, streams, output root, cli action, extra args...
  local name="$1" ckpt="$2" streams="$3" eroot="$4" action="$5"; shift 5
  local pids=() mode=()
  [[ "$action" == eval-stage2 ]] && mode=(--mode v3)
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 "$action" --config "$TRACE_CFG" \
      --streams "$streams" --video-root "$video_root" --output "$eroot/$name" "${mode[@]}" \
      --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev --shard "$i/$TRACE_GPUS" --device cuda:0 "$@" \
      > "$TRACE_OUT/$(basename "$eroot")-$name-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "eval $name failed; see $TRACE_OUT/*-$name-shard*.log" >&2; exit 1; }; done
  CUDA_VISIBLE_DEVICES=0 "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 "$action" --config "$TRACE_CFG" \
    --streams "$streams" --video-root "$video_root" --output "$eroot/$name" "${mode[@]}" \
    --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev "$@" > /dev/null
  log "  $name done"
}

if ! "${py[@]}" -c "import json,sys; sys.exit(0 if json.load(open('$TRACE_OUT/flow.json'))['passed'] else 1)" 2>/dev/null; then
  log "[2/4 flow] S1/S2 fresh16 on the transfer run's flow streams, with the A-B-A isolation check"
  [[ -f "$TRACE_TRANSFER_RUN/flow_streams.jsonl" ]] || { echo "transfer run has no flow_streams.jsonl" >&2; exit 2; }
  cp "$TRACE_TRANSFER_RUN/flow_streams.jsonl" "$TRACE_OUT/flow_streams.jsonl"
  run_cli s1_fresh16 "$TRACE_S1" "$TRACE_OUT/flow_streams.jsonl" "$TRACE_OUT/flow" eval-recent-window --window 16 --isolation 2
  run_cli s2_fresh16 "$TRACE_S2" "$TRACE_OUT/flow_streams.jsonl" "$TRACE_OUT/flow" eval-recent-window --window 16 --isolation 2
  "${py[@]}" - <<'PY' || { echo "flow check failed; see $TRACE_OUT/flow.json" >&2; exit 4; }
import json, os
from pathlib import Path
from trace_vl.v3.recent_window import flow_check
out, tr = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_TRANSFER_RUN"])
r = flow_check(out / "flow", out / "flow_streams.jsonl", fifo_flow_root=tr / "flow")
(out / "flow.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=2))
raise SystemExit(0 if r["passed"] else 1)
PY
fi
log "Flow check passed; continuing automatically to the six evaluations"

log "[3/4 evaluate] {S1,S2} x {stream_window, fresh16, fresh4} on the frozen 688-question manifest"
run_cli s1_fresh16 "$TRACE_S1" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" eval-recent-window --window 16
run_cli s2_fresh16 "$TRACE_S2" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" eval-recent-window --window 16
run_cli s1_fresh4 "$TRACE_S1" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" eval-recent-window --window 4
run_cli s2_fresh4 "$TRACE_S2" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" eval-recent-window --window 4
run_cli s1_stream_window "$TRACE_S1" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" eval-stage2 --policy window
run_cli s2_stream_window "$TRACE_S2" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" eval-stage2 --policy window

log "[4/4 report] Checks, diagnostic contrasts and the interaction"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.recent_window import recent_window_report
out, tr = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_TRANSFER_RUN"])
r = recent_window_report(out / "eval", tr / "eval", tr / "transfer" / "questions.jsonl")
(out / "recent-window-report.json").write_text(json.dumps(r, indent=2) + "\n")
lines = [f"RECENT-WINDOW CONTROL DIGEST ({r['protocol']}; evaluation only; status {r['status']})", f"SCOPE {r['scope']}", r["fresh4_note"]]
if r["coverage_problems"]:
    lines.append(f"COVERAGE PROBLEMS {r['coverage_problems']}")
for arm, acc in r.get("accuracy_pct", {}).items():
    lines.append(f"{arm}: backward {acc['backward']} | realtime {acc['realtime']} | all {acc['all']} | KV {r['kv_tokens_mean'][arm]} | {acc['per_task']}")
for name, c in r.get("contrasts", {}).items():
    lines.append(f"{name}: backward {c['backward']['delta_pp']} {c['backward']['ci95_pp']} | macro {c['backward_task_macro']['delta_pp']} "
                 f"{c['backward_task_macro']['ci95_pp']} | realtime {c['realtime']['delta_pp']} {c['realtime']['ci95_pp']} | all {c['all']['delta_pp']} {c['all']['ci95_pp']}")
if "reading" in r:
    lines.append(f"READING {json.dumps(r['reading'])}")
text = "\n".join(lines) + "\n"
(out / "recent-window-digest.txt").write_text(text)
print(text)
PY
log "Completed: $TRACE_OUT/recent-window-digest.txt"
