#!/usr/bin/env bash
# One queue for the frozen checkpoint and three matched state-pair readers.
set -Eeuo pipefail

code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/state-attention-001}"
repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
model="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_4b}"
plain="${TRACE_PLAIN:-/nas-data/vlm/yyy/tracevl-runs/core-scale40k-54e161a-001/train/plain/checkpoint-005000}"
config="$code/configs/core_vlm/state_attention"
pairs="${TRACE_PAIRS:-}"
review="${TRACE_REVIEW:-$config/review.jsonl}"

[[ -x "$py" && -f "$review" && -s "$plain/complete.json" && -s "$model/config.json" ]]
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
mkdir -p "$TRACE_OUT/logs" "$TRACE_OUT/tmp"
TRACE_OUT="$(realpath "$TRACE_OUT")"
export TRACE_OUT PYTHONPATH="$code/src" TMPDIR="$TRACE_OUT/tmp" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS="${TRACE_THREADS:-4}" MKL_NUM_THREADS="${TRACE_THREADS:-4}"
export TRACE_WANDB=0
exec 9>"$TRACE_OUT/.launcher.lock"
flock -n 9 || { echo "This experiment queue is already running" >&2; exit 2; }
pids=()
stop() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait
  exit 130
}
trap stop INT TERM HUP
trap 'result=$?; if ((result)); then echo "STOPPED ($result). Logs and completed arms are in $TRACE_OUT." >&2; fi' EXIT
IFS=, read -r -a devices <<< "${TRACE_DEVICES:-0,1,2,3}"
declare -A seen=()
for gpu in "${devices[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ || ( "$gpu" == cpu && ${#devices[@]} == 1 ) ]]
  [[ -z "${seen[$gpu]:-}" ]]
  seen[$gpu]=1
done
(("${#devices[@]}" > 0))

echo '[1/4] Local regression checks'
CUDA_VISIBLE_DEVICES='' "$py" -m pytest -q "$code/tests/test_core_vlm_state_attention.py" \
  "$code/tests/test_core_vlm_native.py" "$code/tests/test_core_vlm_order_probe.py" \
  2>&1 | tee "$TRACE_OUT/logs/tests.log"
if [[ -z "$pairs" ]]; then
  activitynet="${TRACE_ACTIVITYNET_ROOT:-/nas-data/vlm/llava/0_30_s_activitynetqa}"
  archive="${TRACE_ACTIVITYNET_ARCHIVE:-$activitynet/0_30_s_activitynetqa_videos_1.tar.gz}"
  CUDA_VISIBLE_DEVICES='' "$py" -m trace_vl.core_vlm.state_media \
    --pairs "$config/pairs.jsonl" --inventory "$config/media.json" \
    --out "$TRACE_OUT/operator-data" --media-root "$activitynet" --archive "$archive" \
    2>&1 | tee "$TRACE_OUT/logs/media.log"
  pairs="$TRACE_OUT/operator-data/pairs.jsonl"
fi
[[ -s "$pairs" ]]
args=(--out "$TRACE_OUT" --model "$model" --plain "$plain" --pairs "$pairs" --review "$review"
      --updates "${TRACE_UPDATES:-100}" --seed "${TRACE_SEED:-17}" --batch "${TRACE_BATCH:-4}"
      --width "${TRACE_WIDTH:-256}" --capacity "${TRACE_PAIR_CAPACITY:-8}" --lr "${TRACE_LR:-0.0001}")
if [[ -n "${TRACE_NATURAL:-}" ]]; then
  args+=(--natural "$TRACE_NATURAL")
fi
runtime_device=cuda:0
if [[ "${devices[0]}" == cpu ]]; then
  devices=('')
  runtime_device=cpu
fi
args+=(--device "$runtime_device")

echo '[2/4] Verify the checkpoint, review receipts and causal inputs; cache shared features'
CUDA_VISIBLE_DEVICES="${devices[0]}" "$py" -m trace_vl.core_vlm.state_experiment prepare "${args[@]}" \
  2>&1 | tee "$TRACE_OUT/logs/prepare.log"

echo '[3/4] Frozen Plain, ordinary cross-attention, difference, and antisymmetric attention'
arms=(plain cross difference antisymmetric)
start=0
while ((start < ${#arms[@]})); do
  pids=()
  names=()
  for i in "${!devices[@]}"; do
    index=$((start+i))
    if ((index >= ${#arms[@]})); then break; fi
    arm="${arms[$index]}"
    names+=("$arm")
    echo "START $arm device=${devices[$i]:-cpu} log=$TRACE_OUT/logs/$arm.log"
    CUDA_VISIBLE_DEVICES="${devices[$i]}" "$py" -m trace_vl.core_vlm.state_experiment arm \
      --out "$TRACE_OUT" --arm "$arm" --device "$runtime_device" \
      >>"$TRACE_OUT/logs/$arm.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "DONE ${names[$i]}"
    else
      tail -n 25 "$TRACE_OUT/logs/${names[$i]}.log" >&2
      failed=1
    fi
  done
  pids=()
  ((failed == 0)) || exit 1
  start=$((start+${#devices[@]}))
done

echo '[4/4] Paired report and logs'
CUDA_VISIBLE_DEVICES='' "$py" -m trace_vl.core_vlm.state_experiment report --out "$TRACE_OUT"
"$py" - <<'PY'
import os
import tarfile
from pathlib import Path

root = Path(os.environ["TRACE_OUT"])
files = [root / name for name in ("run.json", "features.json", "report.json", "REPORT.md")]
media_audit = root / "operator-data/media.json"
if media_audit.is_file():
    files.append(media_audit)
for subdir in ("logs", "constructed", "natural"):
    files.extend(p for p in (root / subdir).rglob("*")
                 if p.is_file() and p.suffix in {".json", ".jsonl", ".log"})
partial = root / "reports.tmp"
with tarfile.open(partial, "w:gz") as archive:
    for path in sorted(files):
        archive.add(path, arcname=path.relative_to(root))
partial.replace(root / "reports.tar.gz")
PY
echo "REPORT BUNDLE: $TRACE_OUT/reports.tar.gz"

if [[ "${TRACE_SWANLAB:-0}" == 1 ]]; then
  export SWANLAB_MODE="${SWANLAB_MODE:-online}" TRACE_SWANLAB_PROJECT="${TRACE_SWANLAB_PROJECT:-trace-vl-core}"
  if ! CUDA_VISIBLE_DEVICES='' "$py" -m trace_vl.core_vlm.swanlab_upload --runs "$TRACE_OUT"; then
    echo "SwanLab upload failed. Experiment results and report bundle are complete." >&2
  fi
fi
