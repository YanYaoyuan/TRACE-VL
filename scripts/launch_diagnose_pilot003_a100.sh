#!/usr/bin/env bash
# Pilot-003 diagnostics on the A100 host: runs against the COMPLETED cc00d60 pilot-003 output, no training.
#   1. gist-only ablation evals of the trained transition and after arms (4 GPUs, 4 shards each)
#   2. provenance, per-question agreement, validation NLL, training curves (CPU)
#   3. weight changes vs the replicated initialization + activation probe on 8 validation videos (GPU 0)
# Diagnostics code: 7dc2ce9 (exact snapshot in its own directory; the run's cc00d60 source/ is not touched).
# The completed run's train/, eval/{native,transition,after,b4}/, identity.json and summary.json are only read.
# New outputs: eval/{transition,after}_gistonly/, configs/*_gistonly.yaml, diagnostics.json, diagnostics-digest.txt.
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_RUNS="${TRACE_RUNS:-/nas-data/vlm/yyy/tracevl-runs/pilot003-cc00d60-001}"
export TRACE_DATA_DIR="${TRACE_DATA_DIR:-/nas-data/vlm/yyy/tracevl-data/stage1-30s-001}"
export TRACE_VIDEO_ROOT="${TRACE_VIDEO_ROOT:-/nas-data/vlm/llava/media}"
export TRACE_MODEL="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_2b}"
export TRACE_GPUS=4 TRACE_PROBE_VIDEOS="${TRACE_PROBE_VIDEOS:-8}" CUDA_VISIBLE_DEVICES=0,1,2,3
export TRACE_UV_BIN="${TRACE_UV_BIN:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rev=7dc2ce9cb78ca20d9077c2195617c70a5c1bf527
trained_rev=cc00d604efd3ba175a822c88f5dc7a4f727483d7
src="$TRACE_RUNS/diagnostics-src-${rev:0:7}"

[[ -f "$TRACE_RUNS/summary.json" && -f "$TRACE_RUNS/identity.json" ]] || { echo "No completed pilot-003 run at $TRACE_RUNS" >&2; exit 2; }
[[ "$(cat "$TRACE_RUNS/base-revision.txt" 2>/dev/null)" == "$trained_rev" ]] || { echo "Run was not produced by $trained_rev" >&2; exit 2; }
exec 8>"$TRACE_RUNS/.diagnostics-launcher.lock"
flock -n 8 || { echo 'This diagnostics launcher is already running.' >&2; exit 2; }
[[ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]] || { echo "Missing existing UV environment: $UV_PROJECT_ENVIRONMENT" >&2; exit 2; }
if ! git -C "$TRACE_REPO" cat-file -e "$rev^{commit}" 2>/dev/null; then
  git -C "$TRACE_REPO" fetch origin design/v3-transition-streaming
fi
git -C "$TRACE_REPO" cat-file -e "$rev^{commit}"
if [[ ! -f "$src/.complete" ]]; then
  rm -rf "$src" && mkdir -p "$src"
  git -C "$TRACE_REPO" archive "$rev" | tar -x -C "$src"
  touch "$src/.complete"
fi
printf '%s\n' "$rev" > "$TRACE_RUNS/diagnostics-revision.txt"
cd "$src"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

"$TRACE_UV_BIN" run --locked --no-sync python - <<'PY'
import json, os
from pathlib import Path
import torch
import trace_vl
from trace_vl.v3.pilot import training_complete
assert Path(trace_vl.__file__).resolve().is_relative_to(Path.cwd().resolve()), f'Wrong trace_vl import: {trace_vl.__file__}'
assert torch.cuda.device_count() == 4, 'Expected four visible CUDA GPUs'
assert Path(os.environ['TRACE_MODEL'], 'config.json').is_file(), 'Cached model path is missing'
root = Path(os.environ['TRACE_RUNS'])
identity = json.loads((root/'identity.json').read_text())
for arm in ('transition', 'after', 'b4'):
    cfg = identity['configs'][arm]
    ok, reason = training_complete(root/'train'/arm, cfg, cfg['train']['steps'])
    assert ok, f'{arm}: {reason}'
for arm in ('native', 'transition', 'after', 'b4'):
    assert list((root/'eval'/arm).glob('predictions.*.jsonl')), f'{arm}: missing pilot predictions'
print('Diagnostics code: 7dc2ce9; trained run: cc00d60; GPUs:', [torch.cuda.get_device_name(i) for i in range(4)])
print('Run:', root)
PY

# Fast CPU unit tests of the diagnostics code (skipped if pytest is not installed in this environment).
if "$TRACE_UV_BIN" run --locked --no-sync python -c 'import pytest' 2>/dev/null; then
  CUDA_VISIBLE_DEVICES= "$TRACE_UV_BIN" run --locked --no-sync python -m pytest -q -p no:cacheprovider \
    tests/test_v3_diagnose.py tests/test_v3_pilot003.py
else
  echo 'pytest not installed in this environment: unit tests skipped (they passed in CI).'
fi

bash scripts/diagnose_pilot003.sh

# Strict post-checks and a short digest to paste back.
"$TRACE_UV_BIN" run --locked --no-sync python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['TRACE_RUNS'])
d = json.loads((root/'diagnostics.json').read_text())
prov = d['provenance']
assert prov['module_checkpoints_distinct'], 'Arm checkpoints are not distinct'
for arm, p in prov['arms'].items():
    assert p['training_complete'] and p['evaluated_checkpoint_is_trained_checkpoint'], f'{arm}: provenance {p}'
assert set(d['ablation']) == {'transition', 'after'}, 'Gist-only ablation missing'
for arm, a in d['ablation'].items():
    assert a['agreement']['n'] == 488, f'{arm}: ablation covered {a["agreement"]["n"]} questions'
lines = ['PILOT-003 DIAGNOSTICS DIGEST (code 7dc2ce9, run cc00d60)']
for arm, a in d['ablation'].items():
    lines.append(f"ablation {arm}: full {a['accuracy_full']:.4f} | gist-only {a['accuracy_gist_only']:.4f} | "
                 f"full-gistonly {a['full_minus_gist_only']} | same answers {a['agreement']['same_prediction_fraction']:.3f} | "
                 f"gistonly-b4 {a.get('gist_only_minus_b4')}")
for pair, a in d['agreement'].items():
    lines.append(f"agreement {pair}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
for arm, n in d['validation_nll'].items():
    lines.append(f"val NLL {arm}: full-vocab {n['nll_full_vocab']:.4f}, options {n['nll_options']:.4f} (n={n['n']})")
for arm, t in d['training'].items():
    f, l = t['first'], t['last']
    lines.append(f"curves {arm}: ce {f.get('ce'):.4f}->{l.get('ce'):.4f}, kd {f.get('kd'):.4f}->{l.get('kd'):.4f}, "
                 f"grad_modules {f.get('grad_norm_modules'):.3f}->{l.get('grad_norm_modules'):.3f}, "
                 f"grad_lora {f.get('grad_norm_lora'):.3f}->{l.get('grad_norm_lora'):.3f}, gate {l.get('gate_abs_mean'):.2e}")
for arm, w in d.get('weights', {}).items():
    # Zero-initialized groups (e.g. cst.gate, type_embed) have no relative change: print N/A and the absolute change.
    rel = sorted(((g['rel_change'], name) for name, g in w['groups'].items() if g['rel_change'] is not None), reverse=True)
    zero = [(name, g['abs_change']) for name, g in w['groups'].items() if g['rel_change'] is None]
    lines.append(f"weights {arm}: init_replication_valid={w['init_replication_valid']}, lora_B_norm={w['lora_B_norm']:.4f}, "
                 f"cst_gate={w['cst_gate']}, rel_change " + ", ".join(f"{n}={v:.3e}" for v, n in rel)
                 + "; zero-init groups (rel N/A, abs change) " + ", ".join(f"{n}={a:.3e}" for n, a in zero))
for arm, p in d.get('probe', {}).items():
    if arm == 'videos':
        continue
    lines.append(f"probe {arm}: " + ", ".join(f"{k}={v['mean']:.4g}[{v['min']:.3g},{v['max']:.3g}]" for k, v in p.items()))
digest = "\n".join(lines) + "\n"
(root/'diagnostics-digest.txt').write_text(digest)
print(digest)
if not all(w['init_replication_valid'] for w in d.get('weights', {}).values()):
    print('WARNING: initialization replica not exact; rel_change values are not interpretable (report it, do not rerun).')
print('VERIFIED: provenance, complete 488-question gist-only ablations. Report:', root/'diagnostics.json')
PY
