#!/usr/bin/env bash
# Stage-2 transfer check on OVO-Bench (experiments.md §14.3; src/trace_vl/v3/ovo_transfer.py). EVALUATION ONLY:
# frozen checkpoints, no training. ONE pass, advancing automatically when each predeclared check passes:
#   1. prepare   (CPU) verify the §13 hash split, convert the complete official annotation, select every video of the
#                ORIGINAL diagnostic hash half, check media; only missing/unreadable videos are extracted from the
#                archives already downloaded (originals are only read); evaluation reads a symlink overlay.
#   2. inventory (CPU) ovo-transfer-dev-v2 manifest, per-half task counts, v1 comparison. STOPS (exit 3) with a
#                task/count/availability report if adequacy fails; there is no override.
#   3. flow      (GPU) S1/S2 FIFO on the 2 shortest + 2 longest streams: coverage, no future frames, identical
#                schedules, wall-time estimate. STOPS (exit 4) if it fails.
#   4. full      (GPU) S1/S2 x FIFO/full on every stream, report and the predeclared decision.
# Finished stages are kept and verified on re-run; finished evaluation shards resume. TRACE_STAGE=inventory stops
# after step 2 (CPU only). The frozen final half is never evaluated (video IDs are used only for disjointness).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
: "${TRACE_DIAG_RUN:?}" "${TRACE_PILOT_RUN:?}" "${TRACE_S1_CKPT:?}" "${TRACE_OUT:?}"
export TRACE_STAGE="${TRACE_STAGE:-all}" TRACE_GPUS="${TRACE_GPUS:-4}" TRACE_MODEL="${TRACE_MODEL:-}"
export TRACE_EVALREPAIR_RUN="${TRACE_EVALREPAIR_RUN:-}" TRACE_OVO_ROOT="${TRACE_OVO_ROOT:-}"
[[ "$TRACE_STAGE" == all || "$TRACE_STAGE" == inventory ]] || { echo "TRACE_STAGE must be all or inventory" >&2; exit 2; }
uv_bin="${TRACE_UV_BIN:-uv}"
py=("$uv_bin" run --locked --no-sync python)
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.lock"
flock -n 9 || { echo "Another OVO transfer run is using $TRACE_OUT" >&2; exit 2; }
log() { echo "[$(date -Is)] $*"; }
export TRACE_S2="$TRACE_PILOT_RUN/train" TRACE_S1="$TRACE_S1_CKPT"
export TRACE_CFG="$TRACE_OUT/memory_pilot_b4_exactfill.yaml" TRACE_T="$TRACE_OUT/transfer"
{ echo "extends: $root/configs/v3/memory_pilot_b4_exactfill.yaml"
  [[ -n "$TRACE_MODEL" ]] && printf 'backbone:\n  name: %s\n  local_files_only: true\n' "$TRACE_MODEL"; true; } > "$TRACE_CFG"

# ---------------------------------------------------------------------------------------------------- 1-2 (CPU)
if [[ ! -f "$TRACE_T/inventory.json" ]]; then
  log "[1/4 prepare] Verify the §13 split, convert the complete official annotation, check and (only if needed) extract media"
  "${py[@]}" - <<'PY'
import hashlib, json, os
from pathlib import Path
from trace_vl.v3.data import convert_ovo, read_jsonl
from trace_vl.v3.ovo_transfer import ensure_media
from trace_vl.v3.temporal import task_code, video_half
from trace_vl.v3.data import OVO_MCQ_TASKS
diag, out = Path(os.environ["TRACE_DIAG_RUN"]), Path(os.environ["TRACE_OUT"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
split = json.loads((diag / "split/manifest.json").read_text())
assert sha(diag / "split/diagnostic.jsonl") == split["diagnostic_sha256"], "diagnostic.jsonl differs from its manifest"
assert sha(diag / "split/final_frozen.jsonl") == split["final_frozen_sha256"], "final_frozen.jsonl differs from its manifest"
conv = json.loads((diag / "conversion.json").read_text())
# Complete annotation, NOT filtered by media on disk, so missing media is visible instead of silently dropped.
report = convert_ovo(conv["annotation"], str(out / "ovo_mc.jsonl"), None, forward=False)
mc = {r["id"]: r for r in read_jsonl(out / "ovo_mc.jsonl")}
old = {r["id"]: r for r in read_jsonl(diag / "ovo_all.jsonl") if r.get("options") is not None}
assert set(old) <= set(mc) and all(mc[k] == old[k] for k in old), "conversion differs from the §13 conversion on shared records"
seed, fraction = split["seed"], split["diagnostic_fraction"]
sampled = {r["video"] for r in read_jsonl(diag / "split/diagnostic.jsonl")}
assert len(sampled) == 200 and all(video_half(v, seed, fraction) == "diagnostic" for v in sampled)
population = sorted({r["video"] for r in mc.values() if task_code(r) in OVO_MCQ_TASKS and video_half(r["video"], seed, fraction) == "diagnostic"})
search = [p for p in {os.environ["TRACE_OVO_ROOT"], str(Path(conv["annotation"]).parent), conv["video_root"]} if p]
media = ensure_media(population, conv["video_root"], sorted(search), out)
(out / "media.json").write_text(json.dumps(media, indent=2) + "\n")
record = {"diag_run": str(diag), "annotation": conv["annotation"], "annotation_sha256": sha(conv["annotation"]),
          "original_video_root": conv["video_root"], "video_root": media["overlay"], "conversion": report,
          "conversion_vs_s13": {"s13_mc_records": len(old), "records": len(mc), "only_in_complete_conversion": len(set(mc) - set(old))},
          "ovo_mc_sha256": sha(out / "ovo_mc.jsonl"), "split_manifest": split, "population_videos": len(population),
          "media": {k: v for k, v in media.items() if k != "archives_scanned"} | {"archives_scanned": len(media["archives_scanned"])}}
problems = len(media["still_missing"]) + len(media["ambiguous"]) + len(media["extracted_but_unreadable"])
print(f"media: {media['available_in_root']} available, {len(media['extracted'])} extracted, {problems} unresolved "
      "(their questions are counted as video_missing_or_unreadable exclusions in the inventory)")
(out / "source.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "split_manifest"}, indent=2))
PY
  log "[2/4 inventory] ovo-transfer-dev-v2 (whole §13 diagnostic hash half, questions <= 600 s), v1 comparison, per-half task counts"
  "${py[@]}" - <<'PY'
import json, os, shutil
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
staging = out / "transfer.staging"
shutil.rmtree(staging, ignore_errors=True)
inv = build_transfer(*args, str(staging), population="diagnostic-half", **common)
old = build_transfer(*args, str(out / "transfer-v1-comparison"), population="sampled-200", **common)
halves = half_counts(str(out / "ovo_mc.jsonl"), seed, fraction)
(out / "half-counts.json").write_text(json.dumps(halves, indent=2) + "\n")
fam = lambda scope, tasks: {t: halves.get(scope, {}).get(t, {"questions": 0, "videos": 0}) for t in tasks}
m = src["media"]
lines = ["OVO TRANSFER INVENTORY v2 (CPU only). Thresholds unchanged: >=150 backward q, >=30 backward videos, max share 10%",
         f"MC conversion (complete annotation) {src['conversion']} | vs §13 {src['conversion_vs_s13']}",
         f"MEDIA population videos {m['videos']}: available {m['available_in_root']}, extracted {len(m['extracted'])}, "
         f"still missing {len(m['still_missing'])} {m['still_missing'][:20]}, ambiguous {len(m['ambiguous'])} {list(m['ambiguous'])[:20]}, "
         f"extracted but unreadable {len(m['extracted_but_unreadable'])} (archives scanned {m['archives_scanned']}, skipped {m['archives_skipped']})",
         "Backward tasks per scope (questions / videos), before eligibility:"]
lines += [f"  {sc}: {json.dumps(fam(sc, ('EPM', 'ASI', 'HLD')))}" for sc in ("all", "diagnostic", "final")]
for name, x in (("v1 (200 sampled videos; comparison only)", old), ("v2 (diagnostic hash half)", inv)):
    b, r = x["backward"], x["realtime"]
    lines += [f"== {x['protocol']} {name}",
              f"population videos {x['diagnostic_videos']} | sampled-200 included {x['sampled_200_videos_in_population']} | overlap frozen final {x['overlap_with_frozen_final']} | hashing to final {x['diagnostic_videos_hashing_to_final']} | unreadable {x['videos_unreadable']}",
              f"in scope {x['in_scope_questions']} {x['in_scope_per_task']}",
              f"excluded {x['excluded']}", f"excluded by task {x['excluded_by_task']}",
              f"eligible {x['eligible_questions']} q on {x['streams']} streams; stream seconds {x['stream_seconds']} total {x['stream_seconds_total']}",
              f"backward {b['questions']} q / {b['videos']} videos, max video share {b['max_video_share']}, effective videos {b['effective_videos']}, query {b['query_time_s']}",
              *[f"  {t}: {v}" for t, v in b["per_task"].items()],
              f"realtime {r['questions']} q / {r['videos']} videos", *[f"  {t}: {v}" for t, v in r["per_task"].items()],
              f"ADEQUACY {json.dumps(x['adequacy'])}"]
text = "\n".join(lines) + "\n"
(out / "inventory-digest.txt").write_text(text)
print(text)
if not inv["adequacy"]["adequate"]:
    (out / "STOPPED-inventory.txt").write_text(text)
    raise SystemExit(3)
shutil.rmtree(out / "transfer", ignore_errors=True)
staging.replace(out / "transfer")  # only an adequate manifest becomes the evaluation manifest
PY
fi
"${py[@]}" -c "import json;a=json.load(open('$TRACE_T/inventory.json'))['adequacy'];assert a['adequate'],a" \
  || { echo "inventory not adequate; see $TRACE_OUT/inventory-digest.txt" >&2; exit 3; }
log "Inventory adequate: $TRACE_T"
[[ "$TRACE_STAGE" == inventory ]] && { log "TRACE_STAGE=inventory: stopping before GPU work"; exit 0; }

# ---------------------------------------------------------------------------------------------------- GPU
log "[3/4 flow] Verify the manifest, the frozen checkpoints and provenance"
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
assert inv["adequacy"]["adequate"]
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

if ! "${py[@]}" -c "import json,sys; sys.exit(0 if json.load(open('$TRACE_OUT/flow.json'))['passed'] else 1)" 2>/dev/null; then
  log "[3/4 flow] S1/S2 FIFO on the 2 shortest + 2 longest transfer streams"
  "${py[@]}" -c "from trace_vl.v3.memory_pilot import precheck_streams; print(precheck_streams('$TRACE_T/streams.jsonl', '$TRACE_OUT/flow_streams.jsonl'))"
  run_eval s1_fifo "$TRACE_S1" "$TRACE_OUT/flow_streams.jsonl" "$TRACE_OUT/flow" --policy fifo
  run_eval s2_fifo "$TRACE_S2" "$TRACE_OUT/flow_streams.jsonl" "$TRACE_OUT/flow" --policy fifo
  "${py[@]}" - <<'PY' || { echo "flow check failed; see $TRACE_OUT/flow.json" >&2; exit 4; }
import json, os
from pathlib import Path
from trace_vl.v3.ovo_transfer import flow_check
out, t = Path(os.environ["TRACE_OUT"]), Path(os.environ["TRACE_T"])
r = flow_check(out / "flow", out / "flow_streams.jsonl")
inv = json.loads((t / "inventory.json").read_text())
rates = [v for v in r["wall_seconds_per_stream_second_per_gpu"].values() if v]
r["estimate_hours_fifo_arm"] = round(max(rates) * inv["stream_seconds_total"] / int(os.environ["TRACE_GPUS"]) / 3600, 2) if rates else None
r["estimate_note"] = "per FIFO arm on all GPUs; the two full-history arms are slower (KV grows with stream length)"
(out / "flow.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=2))
raise SystemExit(0 if r["passed"] else 1)
PY
fi
log "Flow check passed; continuing automatically to the full evaluation"

log "[4/4 full] Four predeclared arms on every transfer stream (frozen checkpoints, exact fill)"
run_eval s1_fifo "$TRACE_S1" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" --policy fifo
run_eval s2_fifo "$TRACE_S2" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" --policy fifo
run_eval s1_full "$TRACE_S1" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" --policy full
run_eval s2_full "$TRACE_S2" "$TRACE_T/streams.jsonl" "$TRACE_OUT/eval" --policy full
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
