#!/usr/bin/env bash
# Training-only mixture -> matched real-GPU smoke -> plain/temporal QA training -> external report.
# The caller pins TRACE_REV to a reviewed commit. Existing checkout is not switched or edited.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the tested mixed-core code commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_4b}"
export TRACE_DATA_OUT="${TRACE_DATA_OUT:-/nas-data/vlm/yyy/tracevl-data/core-mixed-temporal-${TRACE_REV:0:7}}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-mixed-temporal-${TRACE_REV:0:7}-001}"
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
export TMPDIR="$TRACE_OUT/tmp"
mkdir -p "$TMPDIR"
"$py" -m pytest -q tests/test_core_vlm_*.py tests/test_v3_run_io.py
# Only official small annotation CSVs may be downloaded; media/model reuse locally.
export TRACE_NEXTQA_TRAIN="${TRACE_NEXTQA_TRAIN:-/nas-data/vlm/data/NExTQA/trace-core-official-train.csv}"
export TRACE_NEXTQA_VAL="${TRACE_NEXTQA_VAL:-/nas-data/vlm/data/NExTQA/trace-core-official-val.csv}"
get_annotation() {
  local part="$1" dest="$2" expected="$3" url
  mkdir -p "$(dirname "$dest")"
  if [[ ! -f "$dest" ]]; then
    url="https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/$part.csv"
    curl -fL --retry 2 --connect-timeout 20 --max-time 900 "$url" -o "$dest.tmp" ||
      curl -fL --retry 2 --connect-timeout 20 --max-time 900 "https://ghfast.top/$url" -o "$dest.tmp"
    printf '%s  %s\n' "$expected" "$dest.tmp" | sha256sum --check -
    mv "$dest.tmp" "$dest"
  fi
  printf '%s  %s\n' "$expected" "$dest" | sha256sum --check -
}
get_annotation train "$TRACE_NEXTQA_TRAIN" 51ae6e611fa58835f63a0beb325fe647bb197c7e320a31f0bb7a9cf969e082b2
get_annotation val "$TRACE_NEXTQA_VAL" 43198bdef8436b8d64a9b75d846b0987c10cbf94ebf4be325c4a4e54634d66b8
"$py" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.core_vlm.prepare import write_json
from trace_vl.core_vlm.mixed_data import resolve_general_data
from trace_vl.v3.run_io import ensure_run_manifest
out=Path(os.environ['TRACE_OUT'])
cfg=json.loads(Path('configs/core_vlm/mixed_temporal_a100.json').read_text())
cfg.update(model=os.environ['TRACE_MODEL'], data_out=os.environ['TRACE_DATA_OUT'],
           general_data=os.environ.get('TRACE_GENERAL_DATA',cfg['general_data']),
           external_data=os.environ.get('TRACE_EXTERNAL_DATA',cfg['external_data']),
           nextqa_train=os.environ['TRACE_NEXTQA_TRAIN'], nextqa_val=os.environ['TRACE_NEXTQA_VAL'],
           gpus=[int(x) for x in os.environ['TRACE_DEVICES'].split(',')],
           workers=int(os.environ['TRACE_DATA_WORKERS']), archive_workers=int(os.environ['TRACE_ARCHIVE_WORKERS']))
cfg['general_data'] = str(resolve_general_data(cfg))
print('GENERAL QA DATA:', cfg['general_data'], flush=True)
ensure_run_manifest(out/'launcher.json', {'id': os.environ['TRACE_REV'], 'config': cfg})
write_json(out/'experiment-config.json', cfg)
PY
args=(--config "$TRACE_OUT/experiment-config.json" --out "$TRACE_OUT")
if [[ "${TRACE_PREPARE_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
if [[ "${TRACE_SMOKE_ONLY:-0}" == 1 ]]; then args+=(--smoke-only); fi
"$py" -m trace_vl.core_vlm.mixed_experiment "${args[@]}"
# Small reports + predictions + logs; exclude model weights, media and source snapshots.
"$py" - <<'PY'
import os, tarfile
from pathlib import Path
out=Path(os.environ['TRACE_OUT']); data=Path(os.environ['TRACE_DATA_OUT'])
archive=out/'reports.tar.gz'
with tarfile.open(archive.with_suffix('.tmp'),'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        rel=p.relative_to(out)
        if (p.is_file() and rel.parts[0] not in {'source','tmp'} and p.suffix in {'.json','.jsonl','.log','.md'}
                and not any(part.startswith('checkpoint-') for part in rel.parts)):
            tar.add(p,arcname=str(rel))
    for name in ['inventory.json','inputs.json','complete.json','media.json','manifest.jsonl']:
        p=data/name
        if p.is_file(): tar.add(p,arcname='data/'+name)
    external=Path(__import__('json').loads((out/'experiment-config.json').read_text())['external_data'])
    for name in ('ovo.jsonl','nextqa.jsonl','complete.json','inputs.json','inventory.json','exclusions.json'):
        tar.add(external/name,arcname='external/'+name)
archive.with_suffix('.tmp').replace(archive)
print('REPORT BUNDLE:',archive)
PY
