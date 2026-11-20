#!/usr/bin/env bash
# Core video-understanding QA: prepare -> native parity -> DDP flow -> four baselines -> report.
# The caller pins TRACE_REV to a reviewed commit. Existing checkout is not switched or edited.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the tested core-QA code commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_4b}"
export TRACE_DATA_OUT="${TRACE_DATA_OUT:-/nas-data/vlm/yyy/tracevl-data/core-llava-youtube-v1-${TRACE_REV:0:7}}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-qa-${TRACE_REV:0:7}-001}"
export TRACE_DEVICES="${TRACE_DEVICES:-0,1,2,3}"
export TRACE_DATA_WORKERS="${TRACE_DATA_WORKERS:-32}"
export TRACE_ARCHIVE_WORKERS="${TRACE_ARCHIVE_WORKERS:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRACE_WANDB=0 WANDB_MODE=offline
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
py="${TRACE_PYTHON:-$TRACE_REPO/.venv/bin/python}"
[[ -x "$py" ]] || { echo "Missing UV environment: $py" >&2; exit 2; }
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
cd "$TRACE_REPO"
revision=$(git rev-parse --verify "${TRACE_REV}^{commit}")
export TRACE_REV="$revision"
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.launcher.lock"
flock -n 9 || { echo "This experiment is already running" >&2; exit 2; }
trap 'code=$?; if (( code )); then echo "STOPPED ($code): keep logs in $TRACE_OUT; correct the reported error and rerun." >&2; fi' EXIT
if [[ ! -f "$TRACE_OUT/source/.snapshot-complete" ]]; then
  [[ ! -e "$TRACE_OUT/source" ]] || { echo "Incomplete source snapshot; use a new TRACE_OUT" >&2; exit 2; }
  staging=$(mktemp -d "$TRACE_OUT/source.tmp.XXXXXX")
  git archive "$revision" | tar -xf - -C "$staging"
  printf '%s\n' "$revision" > "$staging/.snapshot-complete"
  mv "$staging" "$TRACE_OUT/source"
fi
[[ "$(cat "$TRACE_OUT/source/.snapshot-complete")" == "$revision" ]]
cd "$TRACE_OUT/source"
export PYTHONPATH="$PWD/src"
"$py" -c 'import torch, transformers, peft, pytest; print("Environment imports OK")'
"$py" -m pytest -q tests/test_core_vlm_*.py tests/test_v3_run_io.py
"$py" - <<'PY'
import json
import os
from pathlib import Path
from trace_vl.core_vlm.prepare import write_json
from trace_vl.v3.run_io import ensure_run_manifest

out = Path(os.environ['TRACE_OUT'])
source = Path(os.environ.get('TRACE_DATA_CONFIG', 'configs/core_vlm/llava_youtube_pilot.json'))
data = json.loads(source.read_text())
data['workers'] = int(os.environ['TRACE_DATA_WORKERS'])
data['archive_workers'] = int(os.environ['TRACE_ARCHIVE_WORKERS'])
cfg = json.loads(Path('configs/core_vlm/baselines_a100.json').read_text())
cfg.update(model=os.environ['TRACE_MODEL'], data_config=str(out/'data-config.json'),
           data_out=os.environ['TRACE_DATA_OUT'],
           gpus=[int(x) for x in os.environ['TRACE_DEVICES'].split(',')])
# Freeze launch configuration before writing resolved files on repeat launches.
ensure_run_manifest(out/'launcher.json', {'id': os.environ['TRACE_REV'], 'data': data, 'experiment': cfg})
write_json(out/'data-config.json', data)
write_json(out/'experiment-config.json', cfg)
PY
args=(--config "$TRACE_OUT/experiment-config.json" --out "$TRACE_OUT")
if [[ "${TRACE_PREPARE_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
"$py" -m trace_vl.core_vlm.experiment "${args[@]}"
