#!/usr/bin/env bash
# Pilot-005 on 4 A100s (experiments.md §12, approved): pilot-004's three arms with ONE change, gist_mode: grid
# (transition-grid, after-grid, B4 retrained as the matched control and reproducibility check). Same data, 1,000
# fresh updates, checkpoints 100 / 300 / 1,000 (final = primary), 20-token P-steps, original M2.
# DO NOT LAUNCH before the pilot-004 held-out probe has been reviewed and local verification has passed.
# Code 6a46fb5 (exact snapshot in the NEW output directory). About 5 hours (3 x ~76 min training + 10 evaluations).
# Optional live tracking: export TRACE_WANDB=1 WANDB_PROJECT=trace-vl TRACE_WANDB_GROUP=pilot005-6a46fb5-001
# (failures never stop training; offline fallback; sync later with trace-vl v3 wandb-sync --runs <dir>).
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_RUNS="${TRACE_RUNS:-/nas-data/vlm/yyy/tracevl-runs/pilot005-6a46fb5-001}"
export TRACE_DATA_DIR="${TRACE_DATA_DIR:-/nas-data/vlm/yyy/tracevl-data/stage1-30s-001}"
export TRACE_VIDEO_ROOT="${TRACE_VIDEO_ROOT:-/nas-data/vlm/llava/media}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_2b}"
export TRACE_GPUS=4 TRACE_CLIP_SECONDS=30 CUDA_VISIBLE_DEVICES=0,1,2,3
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rev=6a46fb52fa38358089a8d6487d8205f4efc98fdf
mkdir -p "$TRACE_RUNS"
exec 8>"$TRACE_RUNS/.launcher.lock"
flock -n 8 || { echo 'This pilot-005 launcher is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing existing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
if [[ -f "$TRACE_RUNS/base-revision.txt" && "$(cat "$TRACE_RUNS/base-revision.txt")" != "$rev" ]]; then
  echo "$TRACE_RUNS belongs to another revision; use a new TRACE_RUNS" >&2; exit 2
fi
if ! git -C "$TRACE_REPO" cat-file -e "$rev^{commit}" 2>/dev/null; then
  git -C "$TRACE_REPO" fetch origin design/v3-transition-streaming
fi
git -C "$TRACE_REPO" cat-file -e "$rev^{commit}"
if [[ ! -f "$TRACE_RUNS/source/.complete" ]]; then
  rm -rf "$TRACE_RUNS/source" && mkdir -p "$TRACE_RUNS/source"
  git -C "$TRACE_REPO" archive "$rev" | tar -x -C "$TRACE_RUNS/source"
  touch "$TRACE_RUNS/source/.complete"
fi
printf '%s\n' "$rev" > "$TRACE_RUNS/base-revision.txt"
cd "$TRACE_RUNS/source"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

"$TRACE_UV_BIN" run --locked --no-sync python - <<'PY'
import os
from pathlib import Path
import torch
import trace_vl
assert Path(trace_vl.__file__).resolve().is_relative_to(Path.cwd().resolve()), f'Wrong trace_vl import: {trace_vl.__file__}'
assert torch.cuda.device_count() == 4, 'Expected four visible CUDA GPUs'
assert Path(os.environ['TRACE_MODEL'], 'config.json').is_file(), 'Cached model path is missing'
print('Code: 6a46fb5 (pilot-005); GPUs:', [torch.cuda.get_device_name(i) for i in range(4)])
print('Output:', os.environ['TRACE_RUNS'])
PY

# Gate: CPU unit tests of the pilot code must pass before any GPU time is spent.
if "$TRACE_UV_BIN" run --locked --no-sync python -c 'import pytest' 2>/dev/null; then
  CUDA_VISIBLE_DEVICES= "$TRACE_UV_BIN" run --locked --no-sync python -m pytest -q -p no:cacheprovider \
    tests/test_v3_pilot004.py tests/test_v3_pilot003.py tests/test_v3_diagnose.py tests/test_v3_tracking.py tests/test_v3_wandb_offline.py
else
  echo 'pytest not installed in this environment: unit tests skipped (they must have passed locally / in CI).'
fi

TRACE_PILOT_PREFIX=pilot005 bash scripts/run_pilot004.sh

# Strict post-checks and a short digest to paste back.
"$TRACE_UV_BIN" run --locked --no-sync python - <<'PY'
import json, os
from pathlib import Path
from trace_vl.v3.pilot import training_complete
root = Path(os.environ['TRACE_RUNS'])
identity = json.loads((root/'identity.json').read_text())
summary = json.loads((root/'summary.json').read_text())
assert identity.get('pilot') == 'pilot005', 'identity is not a pilot-005 run'
for arm in ('transition', 'after', 'b4'):
    cfg = identity['configs'][arm]
    assert cfg['transition']['gist_mode'] == 'grid', f'{arm}: gist_mode is not grid' 
    ok, reason = training_complete(root/'train'/arm, cfg, cfg['train']['steps'])
    assert ok, f'{arm}: {reason}'
    rows = [json.loads(s) for s in (root/'train'/arm/'metrics.jsonl').read_text().splitlines() if s.strip()]
    assert [r['step'] for r in rows] == list(range(1, 1001)), f'{arm}: metric steps'
    p = [r['p_step_tokens_mean'] for r in rows if r.get('p_step_tokens_mean') is not None]
    assert p and min(p) == max(p) == 20, f'{arm}: P-step tokens not fixed at 20'
    assert sum(any(k.startswith('diag_') for k in r) for r in rows) >= 100, f'{arm}: selector diagnostics missing'
diag = json.loads((root/'selector-diagnostics.json').read_text())
lines = ['PILOT-005 DIGEST (code 6a46fb5; grid gist; fresh 1,000-update training; final checkpoint primary)']
for step, s in summary['per_checkpoint'].items():
    acc = ', '.join(f"{a} {v:.4f}" for a, v in s['accuracy'].items())
    pairs = ', '.join(f"{k} {v['delta_pp']:+.2f} {v['ci95_pp']}" for k, v in s['paired'].items())
    lines.append(f"step {step}: accuracy {acc}")
    lines.append(f"step {step}: paired pp {pairs}")
lines.append(f"decision: {json.dumps(summary['decision'])}")
pick = ('diag_gist_entropy_min', 'diag_slot_entropy_mean', 'diag_slot_entropy_min', 'diag_part_entropy_mean',
        'diag_emitted_cos', 'diag_emitted_rank', 'diag_update_gist', 'diag_update_slots', 'diag_update_cst_gate')
for arm, series in diag.items():
    parts = []
    for key in pick:
        if series.get(key):
            first, last = series[key][0], series[key][-1]
            parts.append(f"{key} {first[1]:.4g}@{first[0]}->{last[1]:.4g}@{last[0]}")
    lines.append(f"selector {arm}: " + ', '.join(parts))
digest = '\n'.join(lines) + '\n'
(root/'pilot005-digest.txt').write_text(digest)
print(digest)
print('VERIFIED: completed module/LoRA pairs with checkpoints 100/300/1000, 1,000 logged updates, fixed budgets.')
print('Report:', root/'summary.json')
PY
