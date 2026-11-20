#!/usr/bin/env bash
# Fixed held-out probe of pilot-004's FINAL checkpoints (transition / after / b4), pilot-004 review item 2:
# the same validation clips for every arm; per P-step gist and slot entropies, gist logit spread, gist/slot token
# cosine, and emitted-token cosine + effective rank measured on the NORM-MATCHED tokens actually sent to the LM
# (the training-time B4 rank was measured before norm matching). No training, no evaluation; GPU 0, ~10 min.
# Code bfd5490 (exact snapshot in its own directory). Reads the run; writes only probe-heldout-*.json into it.
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_RUNS="${TRACE_RUNS:-/nas-data/vlm/yyy/tracevl-runs/pilot004-a3f8c29-001}"
export TRACE_DATA_DIR="${TRACE_DATA_DIR:-/nas-data/vlm/yyy/tracevl-data/stage1-30s-001}"
export TRACE_VIDEO_ROOT="${TRACE_VIDEO_ROOT:-/nas-data/vlm/llava/media}"
export TRACE_PROBE_VIDEOS="${TRACE_PROBE_VIDEOS:-16}"
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rev=bfd54904141089481221a40cb4ec96e353e10e1f
src="$TRACE_RUNS/probe-src-${rev:0:7}"
out="$TRACE_RUNS/probe-heldout-${TRACE_PROBE_VIDEOS}v-${rev:0:7}.json"
[[ -f "$TRACE_RUNS/identity.json" ]] || { echo "No pilot run at $TRACE_RUNS" >&2; exit 2; }
for arm in transition after b4; do
  [[ -f "$TRACE_RUNS/train/$arm/modules.pt" && -f "$TRACE_RUNS/configs/$arm.yaml" ]] || { echo "Missing $arm checkpoint/config" >&2; exit 2; }
done
exec 8>"$TRACE_RUNS/.probe.lock"
flock -n 8 || { echo 'A probe for this run is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
if ! git -C "$TRACE_REPO" cat-file -e "$rev^{commit}" 2>/dev/null; then
  git -C "$TRACE_REPO" fetch origin design/v3-transition-streaming
fi
git -C "$TRACE_REPO" cat-file -e "$rev^{commit}"
if [[ ! -f "$src/.complete" ]]; then
  rm -rf "$src" && mkdir -p "$src"
  git -C "$TRACE_REPO" archive "$rev" | tar -x -C "$src"
  touch "$src/.complete"
fi
cd "$src"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$TRACE_UV_BIN" run --locked --no-sync python -m trace_vl.cli v3 diagnose-pilot --probe-only --runs "$TRACE_RUNS" \
  --data-dir "$TRACE_DATA_DIR" --video-root "$TRACE_VIDEO_ROOT" --probe-videos "$TRACE_PROBE_VIDEOS" \
  --device cuda:0 --output "$out" > /dev/null
"$TRACE_UV_BIN" run --locked --no-sync python - "$out" <<'PY'
import json, sys
probe = json.load(open(sys.argv[1]))["probe"]
keys = ("diag_gist_entropy_min", "diag_gist_logit_std", "diag_gist_token_cos", "diag_slot_entropy_mean",
        "diag_slot_entropy_min", "diag_part_entropy_mean", "diag_emitted_cos", "diag_emitted_rank", "cst_residual_ratio")
print(f"PROBE DIGEST ({len(probe['videos'])} fixed validation videos; final checkpoints; means [min, max])")
for arm in ("transition", "after", "b4"):
    stats = probe[arm]
    print(arm + ": " + ", ".join(f"{k}={stats[k]['mean']:.4g}[{stats[k]['min']:.3g},{stats[k]['max']:.3g}]"
                                 for k in keys if k in stats))
print("Report:", sys.argv[1])
PY
