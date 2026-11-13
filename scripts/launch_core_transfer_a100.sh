#!/usr/bin/env bash
# External core-QA transfer -> gated matched continuation vs temporal baseline -> report bundle.
# The caller pins TRACE_REV to a reviewed commit. Existing checkout is not switched or edited.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the tested core-transfer code commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_4b}"
export TRACE_DATA_OUT="${TRACE_DATA_OUT:-/nas-data/vlm/yyy/tracevl-data/core-transfer-v1-${TRACE_REV:0:7}}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-transfer-${TRACE_REV:0:7}-001}"
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
# Download only a checksum-pinned, <1 MB official validation annotation if absent.
# Video archives/model/checkpoints are reused locally, never downloaded here.
export TRACE_NEXTQA_ANNOTATION="${TRACE_NEXTQA_ANNOTATION:-/nas-data/vlm/data/NExTQA/trace-core-official-val.csv}"
expected=43198bdef8436b8d64a9b75d846b0987c10cbf94ebf4be325c4a4e54634d66b8
if [[ ! -f "$TRACE_NEXTQA_ANNOTATION" ]]; then
  mkdir -p "$(dirname "$TRACE_NEXTQA_ANNOTATION")"
  url=https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/val.csv
  curl -fL --retry 3 --connect-timeout 20 --max-time 180 "$url" -o "$TRACE_NEXTQA_ANNOTATION.tmp" ||
    curl -fL --retry 3 --connect-timeout 20 --max-time 180 "https://ghfast.top/$url" -o "$TRACE_NEXTQA_ANNOTATION.tmp"
  printf '%s  %s\n' "$expected" "$TRACE_NEXTQA_ANNOTATION.tmp" | sha256sum --check -
  mv "$TRACE_NEXTQA_ANNOTATION.tmp" "$TRACE_NEXTQA_ANNOTATION"
fi
printf '%s  %s\n' "$expected" "$TRACE_NEXTQA_ANNOTATION" | sha256sum --check -
"$py" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.core_vlm.prepare import write_json
from trace_vl.v3.run_io import ensure_run_manifest
out=Path(os.environ['TRACE_OUT'])
cfg=json.loads(Path('configs/core_vlm/transfer_a100.json').read_text())
cfg.update(model=os.environ['TRACE_MODEL'], data_out=os.environ['TRACE_DATA_OUT'],
           baseline_run=os.environ.get('TRACE_BASELINE_RUN',cfg['baseline_run']),
           nextqa_annotation=os.environ['TRACE_NEXTQA_ANNOTATION'],
           gpus=[int(x) for x in os.environ['TRACE_DEVICES'].split(',')],
           workers=int(os.environ['TRACE_DATA_WORKERS']), archive_workers=int(os.environ['TRACE_ARCHIVE_WORKERS']))
ensure_run_manifest(out/'launcher.json', {'id': os.environ['TRACE_REV'], 'config': cfg})
write_json(out/'experiment-config.json', cfg)
PY
args=(--config "$TRACE_OUT/experiment-config.json" --out "$TRACE_OUT")
if [[ "${TRACE_PREPARE_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
"$py" -m trace_vl.core_vlm.transfer "${args[@]}"
# Small reports + predictions + logs; exclude model weights, media and source snapshots.
"$py" - <<'PY'
import os, tarfile
from pathlib import Path
out=Path(os.environ['TRACE_OUT']); data=Path(os.environ['TRACE_DATA_OUT'])
archive=out/'reports.tar.gz'
with tarfile.open(archive.with_suffix('.tmp'),'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        rel=p.relative_to(out)
        if (p.is_file() and rel.parts[0] != 'source' and p.suffix in {'.json','.jsonl','.log'}
                and not any(part.startswith('checkpoint-') for part in rel.parts)):
            tar.add(p,arcname=str(rel))
    for name in ['inventory.json','inputs.json','complete.json','media.json','ovo.jsonl','nextqa.jsonl']:
        p=data/name
        if p.is_file(): tar.add(p,arcname='data/'+name)
archive.with_suffix('.tmp').replace(archive)
print('REPORT BUNDLE:',archive)
PY
