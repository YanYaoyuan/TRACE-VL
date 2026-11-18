#!/usr/bin/env bash
# Reuse audited mixture -> real-GPU smoke -> CE/preserved/temporal QA -> external and refusal reports.
# The caller pins TRACE_REV to a reviewed commit. Existing checkout is not switched or edited.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the tested preservation code commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_4b}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-preservation-${TRACE_REV:0:7}-001}"
export TRACE_DEVICES="${TRACE_DEVICES:-0,1,2,3}"
export TRACE_DATA_WORKERS="${TRACE_DATA_WORKERS:-32}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRACE_WANDB="${TRACE_WANDB:-1}" WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-trace-vl-core}"
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
# Fully offline: reuse verified model, training media and external manifests.
"$py" - <<'PY'
import json, os
from pathlib import Path
from trace_vl.core_vlm.prepare import write_json
from trace_vl.v3.run_io import ensure_run_manifest
out=Path(os.environ['TRACE_OUT'])
cfg=json.loads(Path('configs/core_vlm/preservation_a100.json').read_text())
cfg.update(model=os.environ['TRACE_MODEL'],
           mixture_data=os.environ.get('TRACE_MIXTURE_DATA',cfg['mixture_data']),
           external_data=os.environ.get('TRACE_EXTERNAL_DATA',cfg['external_data']),
           reviewed_pairs=os.environ.get('TRACE_REVIEWED_PAIRS') or None,
           gpus=[int(x) for x in os.environ['TRACE_DEVICES'].split(',')],
           workers=int(os.environ['TRACE_DATA_WORKERS']))
print('FROZEN TRAINING MIXTURE:', cfg['mixture_data'], flush=True)
print('REVIEWED ANSWERABILITY SUPERVISION:', cfg['reviewed_pairs'] or 'NONE; preservation-only pilot', flush=True)
ensure_run_manifest(out/'launcher.json', {'id': os.environ['TRACE_REV'], 'config': cfg})
write_json(out/'experiment-config.json', cfg)
PY
args=(--config "$TRACE_OUT/experiment-config.json" --out "$TRACE_OUT")
if [[ "${TRACE_PREPARE_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
if [[ "${TRACE_SMOKE_ONLY:-0}" == 1 ]]; then args+=(--smoke-only); fi
"$py" -m trace_vl.core_vlm.preservation_experiment "${args[@]}"
# Small reports + predictions + logs; exclude model weights, media and source snapshots.
"$py" - <<'PY'
import os, tarfile
from pathlib import Path
out=Path(os.environ['TRACE_OUT'])
archive=out/'reports.tar.gz'
with tarfile.open(archive.with_suffix('.tmp'),'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        rel=p.relative_to(out)
        if (p.is_file() and rel.parts[0] not in {'source','tmp'} and p.suffix in {'.json','.jsonl','.log','.md'}
                and not any(part.startswith('checkpoint-') for part in rel.parts)):
            tar.add(p,arcname=str(rel))
    external=Path(__import__('json').loads((out/'experiment-config.json').read_text())['external_data'])
    for name in ('ovo.jsonl','nextqa.jsonl','complete.json','inputs.json','inventory.json','exclusions.json'):
        tar.add(external/name,arcname='external/'+name)
archive.with_suffix('.tmp').replace(archive)
print('REPORT BUNDLE:',archive)
PY
