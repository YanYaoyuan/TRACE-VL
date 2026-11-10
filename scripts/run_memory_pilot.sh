#!/usr/bin/env bash
# Stage-2 memory pilot (experiments.md §14; trace_vl.v3.memory_pilot). Modes:
#   smoke : 2 training updates, 2 dev streams per run; writes smoke-ok.json (machine-checked authorization)
#   pilot : 300 updates, 40 dev streams per run; refuses to start without a matching passed smoke-ok.json
#   TRACE_MODE=smoke|pilot TRACE_S1_RUN=<pilot-005 run> TRACE_DATA_DIR=<stage1 manifests> TRACE_VIDEO_ROOT=<media>
#   TRACE_LLAVA_ROOT=<llava root> TRACE_OUT=<NEW dir> [TRACE_SMOKE_OK=<smoke dir>/smoke-ok.json] bash scripts/run_memory_pilot.sh
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_MODE:?smoke or pilot}" "${TRACE_S1_RUN:?}" "${TRACE_DATA_DIR:?}" "${TRACE_VIDEO_ROOT:?}" "${TRACE_LLAVA_ROOT:?}" "${TRACE_OUT:?}"
export TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_LLAVA_MC_JSON="${TRACE_LLAVA_MC_JSON:-}" TRACE_MODEL="${TRACE_MODEL:-}"
export TRACE_SMOKE_OK="${TRACE_SMOKE_OK:-}"
case "$TRACE_MODE" in
  smoke) export TRACE_STEPS=2 TRACE_DEV_STREAMS=2 TRACE_CAPS="60 20 20" ;;
  pilot) export TRACE_STEPS=300 TRACE_DEV_STREAMS=40 TRACE_CAPS="3000 400 400" ;;
  *) echo "TRACE_MODE must be smoke or pilot" >&2; exit 2 ;;
esac
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.lock"
flock -n 9 || { echo "Another memory pilot is using $TRACE_OUT" >&2; exit 2; }
log() { echo "[$(date -Is)] $*"; }
export TRACE_S1="$TRACE_S1_RUN/train/b4" TRACE_CFG="$TRACE_OUT/memory_pilot_b4.yaml" TRACE_ROOT="$root"
[[ -f "$TRACE_S1/modules.pt" && -f "$TRACE_S1/lora/adapter_config.json" ]] || { echo "missing Stage-1 B4 checkpoint in $TRACE_S1" >&2; exit 2; }
if [[ -n "$TRACE_LLAVA_MC_JSON" ]]; then mc=("$TRACE_LLAVA_MC_JSON"); else mapfile -t mc < <(ls "$TRACE_LLAVA_ROOT"/0_30_s_youtube_v0_1/*_mc_*.json); fi
((${#mc[@]})) || { echo "no LLaVA MC json found; set TRACE_LLAVA_MC_JSON" >&2; exit 2; }
export TRACE_MC_JSON="${mc[*]}"

log "[0/6] Config, run expectations and (pilot) smoke authorization"
{ echo "extends: $root/configs/v3/memory_pilot_b4.yaml"
  [[ -n "$TRACE_MODEL" ]] && printf 'backbone:\n  name: %s\n  local_files_only: true\n' "$TRACE_MODEL"; true; } > "$TRACE_CFG"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.memory_pilot import data_sources, init_identity, source_identity, verify_smoke

out = Path(os.environ["TRACE_OUT"])
cfg, _ = load_v3_config(os.environ["TRACE_CFG"])
build = {"stream_seconds": 300.0, "delay_min": 45.0, "delay_max": 240.0, "immediate_fraction": 0.2,
         "distractor_fraction": 0.4, "split_fractions": [0.8, 0.1, 0.1], "split_seed": 7}
excluded = [str(Path(os.environ["TRACE_DATA_DIR"]) / f) for f in ("train.jsonl", "val.jsonl")]
expected = {"revision": source_identity(Path("src/trace_vl/v3")), "config": cfg.to_dict(),
            "init": init_identity(os.environ["TRACE_S1"]),
            "data_sources": data_sources(os.environ["TRACE_MC_JSON"].split(), excluded, build)}
target = out / "expected.json"
if target.exists() and json.loads(target.read_text()) != expected:
    raise SystemExit(f"{out} was started with different code/config/init/data inputs; use a new TRACE_OUT")
target.write_text(json.dumps(expected, indent=2) + "\n")
(out / "build-params.json").write_text(json.dumps(build) + "\n")
if os.environ["TRACE_MODE"] == "pilot":
    smoke = os.environ["TRACE_SMOKE_OK"]
    problems = verify_smoke(smoke, expected) if smoke else ["TRACE_SMOKE_OK not set (path to the smoke run's smoke-ok.json)"]
    if problems:
        raise SystemExit("pilot not authorized: " + "; ".join(problems))
    print("smoke authorization verified:", smoke)
PY

log "[1/6] Data: convert MC questions; freeze splits by source video (Stage-1 videos excluded); stitch per split"
if [[ ! -f "$TRACE_OUT/data/streams/dev.jsonl" ]]; then
  rm -rf "$TRACE_OUT/data" && mkdir -p "$TRACE_OUT/data/streams"
  "${py[@]}" -m trace_vl.cli v3 convert-llava-mc "${mc[@]}" --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_OUT/data/mc_all.jsonl" --val-fraction 0
  # shellcheck disable=SC2086
  "${py[@]}" -m trace_vl.cli v3 stage2-splits "$TRACE_OUT/data/mc_all.jsonl" --exclude "$TRACE_DATA_DIR/train.jsonl" "$TRACE_DATA_DIR/val.jsonl" \
    --output "$TRACE_OUT/data/splits" --max-videos $TRACE_CAPS
  "${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.data import build_stage2_streams
out = Path(os.environ["TRACE_OUT"]) / "data"
b = json.loads((Path(os.environ["TRACE_OUT"]) / "build-params.json").read_text())
report = {}
for split in ("train", "dev", "test"):
    report[split] = build_stage2_streams(str(out / "splits" / f"{split}.jsonl"), str(out / "streams" / f"{split}.tmp"),
                                         os.environ["TRACE_VIDEO_ROOT"], split=split, stream_seconds=b["stream_seconds"],
                                         delay_min=b["delay_min"], delay_max=b["delay_max"],
                                         immediate_fraction=b["immediate_fraction"], distractor_fraction=b["distractor_fraction"])
for split in ("train", "test", "dev"):  # dev last: its presence marks a complete build
    (out / "streams" / f"{split}.tmp").replace(out / "streams" / f"{split}.jsonl")
(out / "streams" / "build.json").write_text(json.dumps(report, indent=2) + "\n")
PY
fi

log "[2/6] Data sufficiency (before any GPU time): counts after exclusion, streams, ACTUAL beyond-window questions"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.memory_pilot import data_problems, data_report
out = Path(os.environ["TRACE_OUT"])
cfg, _ = load_v3_config(os.environ["TRACE_CFG"])
rep = data_report(out / "data" / "streams", out / "data" / "splits" / "manifest.json", int(os.environ["TRACE_DEV_STREAMS"]),
                  cfg.memory.window_seconds, cfg)
rep["excluded_stage1_videos"] = json.loads((out / "data" / "splits" / "manifest.json").read_text())["excluded_videos"]
problems = data_problems(rep, os.environ["TRACE_MODE"])
rep["problems"] = problems
(out / "data" / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(rep, indent=2))
if problems:
    raise SystemExit("insufficient data, stopping before training: " + "; ".join(problems))
PY

log "[3/6] Stage-2 training from the Stage-1 B4 checkpoint (${TRACE_STEPS} updates, ${TRACE_GPUS} GPUs)"
export TRACE_S2="$TRACE_OUT/train"
# shellcheck source=scripts/memory_pilot_lib.sh
source "$root/scripts/memory_pilot_lib.sh"
completion_check() {  # exit 0 = complete and identical (skip), 1 = train
  "${py[@]}" - <<'PY'
import json, os, sys
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.memory_pilot import STAGE2_CHECKPOINT_STEPS, init_identity, source_identity, stream_media_identity, _sha
from trace_vl.v3.pilot import training_complete
out, s2 = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_S2"])
cfg, _ = load_v3_config(os.environ["TRACE_CFG"])
streams = out / "data" / "streams" / "train.jsonl"
identity = {"revision": source_identity(Path("src/trace_vl/v3")), "streams_sha256": _sha(streams),
            "media": stream_media_identity(streams, os.environ["TRACE_VIDEO_ROOT"]), "init": init_identity(os.environ["TRACE_S1"]),
            "steps": int(os.environ["TRACE_STEPS"])}
(out / "train-identity.expected.json").write_text(json.dumps(identity, indent=2) + "\n")
ok, reason = training_complete(s2, cfg.to_dict(), int(os.environ["TRACE_STEPS"]), extra=identity,
                               checkpoint_steps=STAGE2_CHECKPOINT_STEPS)
print("completion check:", "complete and identical" if ok else f"training needed ({reason})")
sys.exit(0 if ok else 1)
PY
}
run_training() {
  cp "$TRACE_OUT/train-identity.expected.json" "$TRACE_S2/train-identity.json"
  "$uv_bin" run --locked --no-sync torchrun --standalone --nproc_per_node="$TRACE_GPUS" -m trace_vl.cli v3 train-stage2 \
    --config "$TRACE_CFG" --streams "$TRACE_OUT/data/streams/train.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_S2" \
    --init "$TRACE_S1/modules.pt" --adapter "$TRACE_S1/lora" --max-steps "$TRACE_STEPS"
  completion_check || { echo "training finished without a valid completion record" >&2; exit 1; }
}
maybe_train "$TRACE_S2"

run_eval() {  # name, checkpoint dir, extra args...
  local name="$1" ckpt="$2"; shift 2
  local pids=()
  for ((i = 0; i < TRACE_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval-stage2 --config "$TRACE_CFG" \
      --streams "$TRACE_OUT/data/streams/dev.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_OUT/eval/$name" \
      --mode v3 --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev --limit "$TRACE_DEV_STREAMS" \
      --shard "$i/$TRACE_GPUS" --device cuda:0 "$@" > "$TRACE_OUT/eval-$name-shard$i.log" 2>&1 &
    pids+=($!)
    sleep 5
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "eval $name failed; see $TRACE_OUT/eval-$name-shard*.log" >&2; exit 1; }; done
  CUDA_VISIBLE_DEVICES=0 "$uv_bin" run --locked --no-sync python -m trace_vl.cli v3 eval-stage2 --config "$TRACE_CFG" \
    --streams "$TRACE_OUT/data/streams/dev.jsonl" --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_OUT/eval/$name" \
    --mode v3 --checkpoint "$ckpt/modules.pt" --adapter "$ckpt/lora" --split dev --limit "$TRACE_DEV_STREAMS" "$@" > /dev/null
  log "  $name done"
}
log "[4/6] Dev evaluation of the Stage-2 model: only the memory policy (or eviction schedule) changes"
run_eval s2_tkm "$TRACE_S2" --policy tkm
run_eval s2_mass "$TRACE_S2" --policy mass
run_eval s2_fifo "$TRACE_S2" --policy fifo
run_eval s2_full "$TRACE_S2" --policy full
run_eval s2_tkm_trainsched "$TRACE_S2" --policy tkm --evict-schedule train   # exactly the training eviction steps
log "[5/6] Separate question: Stage-1 B4 checkpoint under the heuristic policies (does Stage-2 training help?)"
run_eval s1_mass "$TRACE_S1" --policy mass
run_eval s1_fifo "$TRACE_S1" --policy fifo
run_eval s1_full "$TRACE_S1" --policy full

log "[6/6] Report, three-state decision and (smoke) machine-checked authorization"
"${py[@]}" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.config import load_v3_config
from trace_vl.v3.memory_pilot import EXPECTED_RUNS, expected_questions, memory_report, smoke_checks
out, mode = Path(os.environ["TRACE_OUT"]), os.environ["TRACE_MODE"]
cfg, _ = load_v3_config(os.environ["TRACE_CFG"])
ids = set(expected_questions(out / "data" / "streams" / "dev.jsonl", "dev", int(os.environ["TRACE_DEV_STREAMS"]), cfg))
r = memory_report(str(out / "eval"), cfg.memory.window_seconds, expected_runs=EXPECTED_RUNS, expected_ids=ids)
(out / "memory-report.json").write_text(json.dumps(r, indent=2) + "\n")
metrics = [json.loads(s) for s in (out / "train" / "metrics.jsonl").read_text().splitlines() if s.strip()]
data_rep = json.loads((out / "data" / "report.json").read_text())
lines = [f"MEMORY PILOT DIGEST (mode {mode}; dev split; window {r['window_seconds']} s; policy over age/surprise statistics)",
         f"data: {json.dumps(data_rep['dev_selected'])}; train streams {data_rep['streams']['train']['streams']}; "
         f"excluded Stage-1 videos {data_rep['excluded_stage1_videos']}",
         f"training: {len(metrics)} updates; retention grad first/last {metrics[0].get('grad_norm_retention')}/"
         f"{metrics[-1].get('grad_norm_retention')}; ret_pos/neg total {sum(m.get('ret_pos', 0) for m in metrics)}/"
         f"{sum(m.get('ret_neg', 0) for m in metrics)}; q_acc last {metrics[-1].get('q_acc')}"]
for name, s in r["runs"].items():
    lines.append(f"{name}: all {s['all']['accuracy_pct']} {s['all']['ci95_pct']} (n {s['all']['n']}) | beyond-window "
                 f"{s['beyond_window']['accuracy_pct']} {s['beyond_window']['ci95_pct']} (n {s['beyond_window']['n']}) | "
                 f"evidence kept {s['evidence_kept_beyond_window']} | KV {s['kv_tokens_mean']} tok ({s['kv_mb_mean']} MB) | "
                 f"evicted {s['evicted_tokens_mean']} of {s['streamed_tokens_mean']} | age median {s['evidence_age_median']} s | "
                 f"{s['seconds_per_stream']} s/stream")
for pair, p in r["paired"].items():
    lines.append(f"{pair}: beyond-window {p['beyond_window']['delta_pp']} {p['beyond_window']['ci95_pp']} | all {p['all']['delta_pp']} {p['all']['ci95_pp']}")
lines.append(f"DECISION {json.dumps(r['decision'])}")
if mode == "smoke":
    checks = smoke_checks(r, metrics, data_rep)
    expected = json.loads((out / "expected.json").read_text())
    record = {"passed": all(checks.values()), "checks": checks, **expected}
    tmp = out / "smoke-ok.json.tmp"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(out / "smoke-ok.json")
    lines.append(f"SMOKE {'PASSED' if record['passed'] else 'FAILED'} {checks}")
text = "\n".join(lines) + "\n"
(out / "memory-digest.txt").write_text(text)
print(text)
if mode == "smoke" and not record["passed"]:
    raise SystemExit("smoke failed; see smoke-ok.json")
PY
log "Completed ($TRACE_MODE): $TRACE_OUT/memory-report.json, memory-digest.txt"
