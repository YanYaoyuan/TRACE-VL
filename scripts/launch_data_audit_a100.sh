#!/usr/bin/env bash
# Read-only data audit on the A100 host (issue #4 dataset inventory). Code 4de3105 (exact snapshot).
# Extracts nothing, downloads nothing, trains nothing, touches no GPU. Runs at idle CPU/IO priority so it can
# share the host with a running pilot. Writes only into $TRACE_AUDIT (a NEW directory).
# Expected time: tens of minutes, dominated by listing the dataset trees and opening sampled videos on the NAS.
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_AUDIT="${TRACE_AUDIT:-/nas-data/vlm/yyy/tracevl-audit/audit-4de3105-001}"
export TRACE_DATA_DIR="${TRACE_DATA_DIR:-/nas-data/vlm/yyy/tracevl-data/stage1-30s-001}"
export TRACE_LLAVA_ROOT="${TRACE_LLAVA_ROOT:-/nas-data/vlm/llava}"
export TRACE_VIDEO_ROOT="${TRACE_VIDEO_ROOT:-/nas-data/vlm/llava/media}"
export TRACE_DATASETS_ROOT="${TRACE_DATASETS_ROOT:-/nas-data/vlm/data}"
export TRACE_PROBE_LIMIT="${TRACE_PROBE_LIMIT:-2000}" TRACE_MAX_FILES="${TRACE_MAX_FILES:-500000}"
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
rev=4de3105e68bb0d774b1378546b1baca6edc3e7cd
mkdir -p "$TRACE_AUDIT"
exec 8>"$TRACE_AUDIT/.launcher.lock"
flock -n 8 || { echo 'This audit is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing existing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
for d in "$TRACE_DATA_DIR" "$TRACE_LLAVA_ROOT" "$TRACE_VIDEO_ROOT" "$TRACE_DATASETS_ROOT"; do
  [[ -d "$d" ]] || { echo "Missing directory: $d" >&2; exit 2; }
done
if ! git -C "$TRACE_REPO" cat-file -e "$rev^{commit}" 2>/dev/null; then
  git -C "$TRACE_REPO" fetch origin design/v3-transition-streaming
fi
git -C "$TRACE_REPO" cat-file -e "$rev^{commit}"
if [[ ! -f "$TRACE_AUDIT/source/.complete" ]]; then
  rm -rf "$TRACE_AUDIT/source" && mkdir -p "$TRACE_AUDIT/source"
  git -C "$TRACE_REPO" archive "$rev" | tar -x -C "$TRACE_AUDIT/source"
  touch "$TRACE_AUDIT/source/.complete"
fi
printf '%s\n' "$rev" > "$TRACE_AUDIT/base-revision.txt"
cd "$TRACE_AUDIT/source"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
nice_cmd=(nice -n 19)
command -v ionice >/dev/null && nice_cmd=(ionice -c3 nice -n 19)

# Raw directory listings (what the operator previously supplied by hand), for the record.
{
  echo "# du was already supplied by the operator (issue #4); sizes below come from the bounded audit walk"
  echo "# ls (llava subsets, first 40 entries each)"
  for d in "$TRACE_LLAVA_ROOT"/*/; do echo "## $d"; ls -la "$d" 2>&1 | head -40 || true; done
} > "$TRACE_AUDIT/listings.txt"

"${nice_cmd[@]}" "$TRACE_UV_BIN" run --locked --no-sync python -m trace_vl.cli v3 audit-data \
  --data-dir "$TRACE_DATA_DIR" --llava-root "$TRACE_LLAVA_ROOT" --media-root "$TRACE_VIDEO_ROOT" \
  --data-root "$TRACE_DATASETS_ROOT" --report-dir "$TRACE_AUDIT" \
  --probe-limit "$TRACE_PROBE_LIMIT" --max-files "$TRACE_MAX_FILES"

"$TRACE_UV_BIN" run --locked --no-sync python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['TRACE_AUDIT'])
report = json.loads((root/'audit.json').read_text())
missing = [s for s in ('manifests', 'media_root', 'llava', 'datasets', 'converters', 'overlap') if s not in report]
print('Sections completed:', 6 - len(missing), '/ 6', '(missing: ' + ', '.join(missing) + ')' if missing else '')
if report.get('errors'):
    print('SECTION ERRORS (report them; other sections are still valid):', report['errors'])
print('AUDIT DONE. Submit:', root/'audit-digest.txt', root/'audit.json', root/'listings.txt')
PY
