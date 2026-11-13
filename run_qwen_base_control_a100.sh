#!/usr/bin/env bash
# Original Qwen3.5-2B vs frozen Stage-1, matched B4/FIFO. No training, no final-half evaluation.
# Standalone file; run on the A100 host. No git switch/pull or environment changes.
set -Eeuo pipefail
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export TRACE_PREVIOUS="${TRACE_PREVIOUS:-/nas-data/vlm/yyy/tracevl-runs/ovo-transfer-cdbac0c-001}"
export TRACE_OUT="${TRACE_OUT:-/nas-data/vlm/yyy/tracevl-runs/qwen-base-control-e262092-001}"
export TRACE_SOURCE_REV=e2620921f6f1f196b47f23a6ee08e8b14a4b15d6
export TRACE_GPUS="${TRACE_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRACE_WANDB=0 WANDB_MODE=offline
py="${TRACE_PYTHON:-$TRACE_REPO/.venv/bin/python}"
[[ -x "$py" ]] || { echo "Missing Python environment: $py" >&2; exit 2; }
mkdir -p "$TRACE_OUT"
printf 'Source: %s\nOutput: %s\n' "$TRACE_SOURCE_REV" "$TRACE_OUT"
exec 9>"$TRACE_OUT/.launcher.lock"
flock -n 9 || { echo "This experiment is already running" >&2; exit 2; }
cd "$TRACE_REPO"
if ! git cat-file -e "$TRACE_SOURCE_REV^{commit}"; then
  git fetch origin design/v3-transition-streaming
fi
if [[ ! -f "$TRACE_OUT/source/.snapshot-complete" ]]; then
  [[ ! -e "$TRACE_OUT/source" ]] || { echo "Incomplete source directory; use a fresh TRACE_OUT" >&2; exit 2; }
  staging=$(mktemp -d "$TRACE_OUT/source.tmp.XXXXXX")
  git archive "$TRACE_SOURCE_REV" | tar -xf - -C "$staging"
  printf '%s\n' "$TRACE_SOURCE_REV" > "$staging/.snapshot-complete"
  mv "$staging" "$TRACE_OUT/source"
fi
[[ "$(cat "$TRACE_OUT/source/.snapshot-complete")" == "$TRACE_SOURCE_REV" ]]
export PYTHONPATH="$TRACE_OUT/source/src"
cat > "$TRACE_OUT/qwen_base_control.py" <<'TRACE_BASE_HELPER'
"""Original Qwen weights vs Stage-1, in the identical B4/FIFO evaluator. No training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from trace_vl.v3.config import V3Config
from trace_vl.v3.data import read_jsonl, write_jsonl
from trace_vl.v3.harness import _dir_identity, _video_identity
from trace_vl.v3.model import TraceV3Modules, stream_step
from trace_vl.v3.modules import CSTCache, TECache, exact_pool, match_norm
from trace_vl.v3.recent_window import check_arms, contrast


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + '\n')


def contract(cfg):
    t, m = cfg.transition, cfg.memory
    assert t.p_step_mode == 'pooled' and t.pooled_match and t.fixed_active == 4
    assert t.gist == 8 and t.emit_slots and t.scene_cut_surprise > 1
    assert t.i_step_every == 8
    assert m.policy == 'fifo' and m.fill == 'exact' and m.evict_every == 1
    assert m.window_seconds == 16 and m.budget_tokens == 1024


@torch.no_grad()
def identity_test(cfg, width=32):
    """Zero-gate wrappers emit exactly raw I-step tokens or deterministic B4 pools.

    Two unrelated random initializations, multiple I/P steps and a resolution change.
    Auxiliary random selectors must not change emitted values, positions or groups.
    """
    contract(cfg)
    d = cfg.to_dict()
    d['cst'].update(dim=16, heads=4, layers=1, window=2)
    d['alert'].update(hidden=32, heads=4)
    small = V3Config.from_dict(d)
    torch.manual_seed(77)
    grids = [(6, 6)] * 10 + [(4, 6)] * 3
    xs = [torch.randn(h * w, width) for h, w in grids]
    for seed in (7, 91):
        torch.manual_seed(seed)
        model = TraceV3Modules(small, width).eval()
        assert torch.count_nonzero(model.cst.gate) == 0
        cc, tc, since, previous = CSTCache(), TECache(), 10**9, None
        for i, (x, grid) in enumerate(zip(xs, grids)):
            adapted, _, info = stream_step(model, x, grid, float(i), cc, tc)
            assert torch.equal(adapted, x), 'original baseline changed visual embeddings'
            is_i = previous != grid or since + 1 >= 8
            assert info.is_i_step == is_i
            if is_i:
                expected = x
                rr = torch.arange(grid[0]).repeat_interleave(grid[1])
                cols = torch.arange(grid[1]).repeat(grid[0])
                since = 0
            else:
                pooled, rr, cols = exact_pool(x, *grid, 20)
                expected = match_norm(pooled, x.norm(dim=-1).mean())
                since += 1
            assert torch.equal(info.tokens, expected)
            assert torch.equal(info.rows, rr.long()) and torch.equal(info.cols, cols.long())
            assert info.groups == [-1] * len(expected)
            assert info.kinds == [('frame' if is_i else 'gist')] * len(expected)
            previous = grid
    return {'zero_gate_identity': True, 'deterministic_B4': True,
            'seeds': [7, 91], 'steps_per_seed': len(xs), 'resolution_change': True}


def check_pair(preds, questions):
    arms = ('s1_fifo', 'base_fifo')
    problems = check_arms(preds, questions, arms)
    meta = {q['id']: q for q in questions}
    for arm in arms:
        for r in preds[arm]:
            q = meta.get(r['id'])
            if q is None:
                continue
            if not 0 <= r['choice'] < len(q['options']) or r['correct'] != int(r['choice'] == q['answer']):
                problems.append(f'{arm}: invalid choice/correct for {r["id"]}')
    maps = [{r['id']: r for r in preds[a]} for a in arms]
    for key in ('kv_tokens', 'visual_tokens_streamed', 'kv_appended_tokens', 'evicted_tokens', 'stream_steps'):
        if {k: v.get(key) for k, v in maps[0].items()} != {k: v.get(key) for k, v in maps[1].items()}:
            problems.append(f'unequal {key}')
    if problems:
        raise ValueError('; '.join(problems))


def prepare(out, previous):
    run = json.loads((previous / 'eval/s1_fifo/run.json').read_text())
    inp = json.loads((previous / 'inputs.json').read_text())
    source = json.loads((previous / 'source.json').read_text())
    cfg = V3Config.from_dict(run['config'])
    contract(cfg)
    test = identity_test(cfg)
    streams_path = previous / 'transfer/streams.jsonl'
    qs_path = previous / 'transfer/questions.jsonl'
    assert sha(streams_path) == inp['streams_sha256'] == run['records_sha256']
    assert sha(qs_path) == inp['questions_sha256']
    streams, qs = read_jsonl(streams_path), read_jsonl(qs_path)
    assert len(streams) == 260 and len(qs) == 688
    assert len({q['id'] for q in qs}) == len(qs)
    assert _video_identity(streams, source['video_root']) == run['videos'], 'media identity changed'
    s1 = Path(inp['stage1']['dir'])
    assert sha(s1 / 'modules.pt') == run['checkpoint']
    assert _dir_identity(str(s1 / 'lora')) == run['adapter']
    base = Path(cfg.backbone.name)
    model_config = json.loads((base / 'config.json').read_text())
    width = model_config['text_config']['hidden_size']
    assert width == 2048, 'unexpected backbone hidden size'
    # Hash actual local weights once; metadata/path alone is not a model identity.
    base_files = sorted(p for p in base.rglob('*') if p.is_file() and p.suffix in {'.json', '.safetensors'})
    assert any(p.suffix == '.safetensors' for p in base_files)
    identity = {'protocol': 'qwen-base-vs-s1-b4-fifo-v1',
                'source_revision': os.environ.get('TRACE_SOURCE_REV'),
                'helper_sha256': sha(__file__), 'config': cfg.to_dict(),
                'streams_sha256': sha(streams_path), 'questions_sha256': sha(qs_path),
                's1_modules_sha256': run['checkpoint'], 's1_adapter': run['adapter'],
                'base_files': {str(p.relative_to(base)): sha(p) for p in base_files},
                'video_identity': run['videos'], 'video_root': source['video_root'], 's1_dir': str(s1)}
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'inputs.json').exists():
        assert json.loads((out / 'inputs.json').read_text()) == identity, 'inputs changed: use a new TRACE_OUT'
        assert sha(out / 'base-modules.pt') == json.loads((out / 'base-identity.json').read_text())['sha256']
    else:
        torch.manual_seed(7)
        modules = TraceV3Modules(cfg, width).eval()
        assert torch.count_nonzero(modules.cst.gate) == 0
        modules.save(out / 'base-modules.pt', {'baseline': 'fresh zero-gate wrapper; no trained weights'})
        write_json(out / 'base-identity.json', {'sha256': sha(out / 'base-modules.pt'), 'seed': 7, 'adapter': None, 'test': test})
        write_json(out / 'inputs.json', identity)
    write_json(out / 'config.json', cfg.to_dict())
    write_jsonl(out / 'streams.jsonl', streams)
    write_jsonl(out / 'questions.jsonl', qs)
    ordered = sorted(streams, key=lambda r: max(q['time'] for q in r['questions']))
    write_jsonl(out / 'flow-streams.jsonl', ordered[:2] + ordered[-2:])
    print('PREPARED: original weights + deterministic B4/FIFO vs frozen S1; no training')


def evaluate(out, arm, stage, shard, count):
    from trace_vl.v3.train_stage2 import evaluate_streams
    cfg = V3Config.from_dict(json.loads((out / 'config.json').read_text()))
    identity = json.loads((out / 'inputs.json').read_text())
    s1 = Path(identity['s1_dir'])
    is_base = arm == 'base_fifo'
    torch.manual_seed(7)
    evaluate_streams(cfg, str(out / ('flow-streams.jsonl' if stage == 'flow' else 'streams.jsonl')),
                     identity['video_root'], str(out / stage / arm), mode='v3',
                     checkpoint=str(out / 'base-modules.pt') if is_base else str(s1 / 'modules.pt'),
                     adapter=None if is_base else str(s1 / 'lora'), policy='fifo',
                     device='cuda:0', split='dev', shard=(shard, count))


def report(out, stage):
    qs = read_jsonl(out / 'questions.jsonl')
    if stage == 'flow':
        ids = {q['id'] for r in read_jsonl(out / 'flow-streams.jsonl') for q in r['questions']}
        qs = [q for q in qs if q['id'] in ids]
    preds = {a: read_jsonl(out / stage / a / 'predictions.jsonl') for a in ('base_fifo', 's1_fifo')}
    runs = {a: json.loads((out / stage / a / 'run.json').read_text()) for a in preds}
    assert runs['base_fifo']['adapter'] is None
    for k in ('config', 'records_sha256', 'mode', 'videos'):
        assert runs['base_fifo'][k] == runs['s1_fifo'][k], f'run mismatch: {k}'
    for a in preds:
        assert all(r['run'] == runs[a]['id'] for r in preds[a])
    check_pair(preds, qs)
    if stage == 'flow':
        write_json(out / 'flow.json', {'passed': True, 'questions': len(qs), 'exact_per_question_budget': True})
        print('FLOW PASSED: equal coverage, frames/schedule and exact token/KV budget')
        return
    meta = {q['id']: q for q in qs}
    correct = {a: {r['id']: r['correct'] for r in rows} for a, rows in preds.items()}
    scores = {a: contrast(correct, meta, {a: 1}) for a in preds}
    delta = contrast(correct, meta, {'s1_fifo': 1, 'base_fifo': -1})
    result = {'status': 'complete', 'scope': 'reused external-dataset development set; one trained seed',
              'scores_percent': scores, 's1_minus_base_pp': delta,
              'note': 'Original Qwen weights under TRACE B4/FIFO wrapper, not official native Qwen benchmark. '
                      'Difference tests the complete Stage-1 package, not LoRA or CST separately. '
                      'These bootstrap intervals do not cover training-seed variance.'}
    write_json(out / 'base-control-report.json', result)
    lines = ['ORIGINAL QWEN vs STAGE-1: identical B4/FIFO, 688 development questions; no training']
    for a in scores:
        lines.append(a + ': ' + ', '.join(f'{k}={scores[a][k]["delta_pp"]:.3f}%' for k in ('all', 'backward', 'realtime')))
    for k in ('all', 'backward', 'realtime'):
        lines.append(f'S1 - original {k}: {delta[k]}')
    lines.append('Primary endpoint: all-question accuracy. Positive/negative CI excludes zero; otherwise inconclusive. '
                 'This is development evidence, not proof of general superiority.')
    text = '\n'.join(lines) + '\n'
    (out / 'base-control-digest.txt').write_text(text)
    print(text)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'eval', 'report'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--previous', type=Path)
    p.add_argument('--arm', choices=['base_fifo', 's1_fifo'])
    p.add_argument('--stage', choices=['flow', 'eval'], default='flow')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--count', type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.action == 'prepare':
        prepare(a.out, a.previous)
    elif a.action == 'eval':
        evaluate(a.out, a.arm, a.stage, a.shard, a.count)
    else:
        report(a.out, a.stage)

TRACE_BASE_HELPER
cat > "$TRACE_OUT/test_qwen_base_control.py" <<'TRACE_BASE_TESTS'
import copy

import pytest

from qwen_base_control import check_pair, contract, identity_test
from trace_vl.v3.config import V3Config


@pytest.fixture
def cfg():
    d = V3Config().to_dict()
    d['transition'].update(p_step_mode='pooled', pooled_match=True, fixed_active=4,
                           gist_mode='grid', scene_cut_surprise=1.01)
    d['memory'].update(policy='fifo', fill='exact', window_seconds=16, budget_tokens=1024)
    return V3Config.from_dict(d)


def test_original_wrapper_is_exact_pooling_independent_of_random_weights(cfg):
    assert identity_test(cfg)['deterministic_B4']


@pytest.mark.parametrize('section,key,value', [
    ('memory', 'policy', 'tkm'), ('memory', 'fill', 'groups'),
    ('transition', 'scene_cut_surprise', .6), ('transition', 'fixed_active', 0),
    ('transition', 'pooled_match', False), ('transition', 'p_step_mode', 'transition')])
def test_reject_uncontrolled_baseline(cfg, section, key, value):
    d = cfg.to_dict()
    d[section][key] = value
    with pytest.raises(AssertionError):
        contract(V3Config.from_dict(d))


@pytest.fixture
def pair():
    q = {'id': 'q1', 'stream': 'v1', 'time': 2., 'answer': 1, 'options': ['a', 'b']}
    row = {'id': 'q1', 'stream': 'v1', 't': 2., 't_step': 1.25, 'step_window_end': 1.5,
           'choice': 1, 'correct': 1, 'kv_tokens': 20, 'visual_tokens_streamed': 30,
           'kv_appended_tokens': 40, 'evicted_tokens': 20, 'stream_steps': 2}
    return {'s1_fifo': [row], 'base_fifo': [copy.deepcopy(row)]}, [q]


def test_valid_pair(pair):
    check_pair(*pair)


@pytest.mark.parametrize('field,value', [('kv_tokens', 21), ('visual_tokens_streamed', 31),
    ('t_step', 1.5), ('step_window_end', 3), ('correct', 0), ('choice', 5)])
def test_reject_budget_schedule_leak_and_score_changes(pair, field, value):
    preds, qs = pair
    preds['base_fifo'][0][field] = value
    with pytest.raises(ValueError):
        check_pair(preds, qs)


def test_reject_missing_question(pair):
    preds, qs = pair
    preds['base_fifo'] = []
    with pytest.raises(ValueError):
        check_pair(preds, qs)

TRACE_BASE_TESTS
cd "$TRACE_OUT/source"
"$py" -m pytest -q "$TRACE_OUT/test_qwen_base_control.py" tests/test_v3_recent_window.py tests/test_v3_stage2.py
"$py" "$TRACE_OUT/qwen_base_control.py" prepare --out "$TRACE_OUT" --previous "$TRACE_PREVIOUS"
# Check devices only after CPU validation. No process killing or automatic memory reclamation.
"$py" - <<'GPU'
import os, torch
n=int(os.environ['TRACE_GPUS'])
assert n>0 and torch.cuda.device_count()==n, 'TRACE_GPUS must match CUDA_VISIBLE_DEVICES'
for i in range(n):
    free,total=torch.cuda.mem_get_info(i)
    print(f'GPU {i}: free {free/2**30:.1f} GiB / {total/2**30:.1f} GiB')
    assert free >= 30*2**30, 'Need 30 GiB free per selected GPU; free other jobs first'
GPU
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for stage in flow eval; do
  for arm in base_fifo s1_fifo; do
    printf '[%s] %s %s: starting/resuming\n' "$(date -Is)" "$stage" "$arm"
    pids=()
    for ((i=0; i<TRACE_GPUS; i++)); do
      CUDA_VISIBLE_DEVICES="${devices[$i]}" "$py" "$TRACE_OUT/qwen_base_control.py" eval \
        --out "$TRACE_OUT" --stage "$stage" --arm "$arm" --shard "$i" --count "$TRACE_GPUS" \
        > "$TRACE_OUT/$stage-$arm-shard$i.log" 2>&1 &
      pids+=("$!")
    done
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        :
      else
        status=$?
        printf 'FAILED: %s %s shard %s (exit %s). Shard log tails follow.\n' "$stage" "$arm" "$i" "$status" >&2
        for log in "$TRACE_OUT/$stage-$arm-"shard*.log; do
          printf '\n%s\n' "$log" >&2
          tail -n 30 "$log" >&2 || true
        done
        exit "$status"
      fi
    done
    pids=()
    # Assembly only: all streams must already be complete; the evaluator verifies its unit records.
    "$py" "$TRACE_OUT/qwen_base_control.py" eval --out "$TRACE_OUT" --stage "$stage" --arm "$arm"
    printf '[%s] %s %s: done\n' "$(date -Is)" "$stage" "$arm"
  done
  "$py" "$TRACE_OUT/qwen_base_control.py" report --out "$TRACE_OUT" --stage "$stage"
done
printf 'VERIFIED: %s/base-control-digest.txt\n' "$TRACE_OUT"
echo 'Submit: base-control-digest.txt, base-control-report.json, inputs.json, base-identity.json, flow.json, eval/*/predictions.jsonl, eval/*/run.json and the launcher log.'
