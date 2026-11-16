#!/usr/bin/env bash
# Evaluation only: frozen mixed-core checkpoints -> replay -> ablations -> report.
set -Eeuo pipefail
: "${TRACE_REV:?Set TRACE_REV to the reviewed diagnostics commit}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_BASE="${TRACE_BASE:-/nas-data/vlm/yyy/tracevl-runs/core-mixed-temporal-97272fb-001}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/core-diagnostics-${TRACE_REV:0:7}-001}"
export TRACE_DEVICES="${TRACE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRACE_WANDB=0 WANDB_MODE=offline
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
py="${TRACE_PYTHON:-$TRACE_REPO/.venv/bin/python}"
[[ -x "$py" ]] || { echo "Missing UV environment: $py" >&2; exit 2; }
cd "$TRACE_REPO"
revision=$(git rev-parse --verify "${TRACE_REV}^{commit}")
export TRACE_REV="$revision"
mkdir -p "$TRACE_OUT"
exec 9>"$TRACE_OUT/.launcher.lock"
flock -n 9 || { echo "This diagnostics launcher is already running" >&2; exit 2; }
trap 'code=$?; if (( code )); then echo "STOPPED ($code): retain $TRACE_OUT; fix the reported issue and rerun the same command." >&2; fi' EXIT
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
export TMPDIR="$TRACE_OUT/tmp"
mkdir -p "$TMPDIR"
"$py" -c 'import torch, transformers, peft, pytest; print("Environment imports OK; evaluation only")'
"$py" -m pytest -q tests/test_core_vlm_*.py tests/test_v3_run_io.py
"$py" -m trace_vl.core_vlm.mixed_diagnostics --base "$TRACE_BASE" --out "$TRACE_OUT" --devices "$TRACE_DEVICES"
"$py" - <<'PY'
import os, tarfile
from pathlib import Path
out=Path(os.environ['TRACE_OUT'])
archive=out/'reports.tar.gz'
with tarfile.open(archive.with_suffix('.tmp'), 'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        rel=p.relative_to(out)
        if (p.is_file() and rel.parts[0] not in {'source','tmp'}
                and p.suffix in {'.json','.jsonl','.log','.md','.txt'}):
            tar.add(p, arcname=str(rel))
archive.with_suffix('.tmp').replace(archive)
print('REPORT BUNDLE:', archive, flush=True)
PY
