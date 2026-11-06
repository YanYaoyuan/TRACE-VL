# v3 runbook (M0 → M2)

Code: `src/trace_vl/v3/`. Every command runs on your machine; none were run on the dev server.
Post each result as a comment on issue #4 using the template in `experiments.md` §7.

| File | Role |
|---|---|
| `backbone.py` | Qwen3.5-2B wrapper: per-step vision encoding, incremental LM calls, hybrid-cache eviction, LoRA |
| `positions.py` | Qwen3.5 MRoPE positions for streaming (matches `get_rope_index`) |
| `modules.py` | CST adapter, transition encoder, alert head, event (JEPA) head, retention scorer |
| `kv.py` | full-attention KV policies: full / window / fifo / drift / tkm |
| `session.py` | streaming session (native or v3), forked question answering, selective decoding |
| `data.py` | timed multiple-choice records, OVO-Bench / Video-MME / MEVA converters, causal step reader |
| `harness.py` | evaluation, scoring (video/camera bootstrap), profiling, the M1 equality check |
| `train.py` | Stage 1 distillation (teacher = backbone with LoRA disabled), torchrun DDP |

## 0. Environment

```bash
git fetch origin && git checkout design/v3-transition-streaming
uv sync --extra dev --extra vlm --extra data      # transformers 5.20+ (has Qwen3.5), peft, torchvision
uv pip install flash-linear-attention             # optional: fast Gated DeltaNet kernels (else torch fallback)
uv run --locked pytest -q tests/test_v3.py        # CPU, no weights
uv run --locked trace-vl v3 params                # new-module parameter report
```

If `transformers` warns that the fast GDN path is unavailable, record it in the M0 comment, because it
changes the latency numbers.

## M0: backbone harness and baselines

```bash
# 1. Equality gate first (also part of M1), in float32: incremental streaming == HF forward (< 1e-3),
#    zero-gated v3 == native (< 1e-5). bf16 differs by ~0.1 from rounding alone (informational: --dtype bfloat16).
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl v3 check --config configs/v3/native_full.yaml --video path/to/short.mp4
# 1b. Gradient regression on a static clip (real backbone): total unclipped norm must stay < 1e4
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl v3 gradcheck --config configs/v3/stage1.yaml

# 2. Cost profile on synthetic frames (10 min = 600 steps) for each KV policy
for c in native_full native_window native_drift native_fifo; do
  CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl v3 profile --config configs/v3/$c.yaml --steps 600 > outputs/v3/profile_$c.json
done

# 3. Benchmarks -> timed multiple-choice records
uv run --locked trace-vl v3 convert-ovo /data/ovo/ovo_bench_new.json --output data/v3/ovo.jsonl
uv run --locked trace-vl v3 convert-videomme /data/videomme/test-00000-of-00001.parquet --output data/v3/videomme.jsonl
uv run --locked trace-vl v3 convert-meva --bench data/meva/bench --split test --output data/v3/meva_test.jsonl

# 4. Zero-shot baselines, one shard per GPU, then score
for c in native_window native_drift native_fifo; do
  CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl v3 eval --config configs/v3/$c.yaml --records data/v3/ovo.jsonl \
    --video-root /data/ovo --output outputs/v3/ovo_$c --shard 0/2 &
  CUDA_VISIBLE_DEVICES=1 uv run --locked trace-vl v3 eval --config configs/v3/$c.yaml --records data/v3/ovo.jsonl \
    --video-root /data/ovo --output outputs/v3/ovo_$c --shard 1/2 &
  wait
  uv run --locked trace-vl v3 score outputs/v3/ovo_$c --output outputs/v3/ovo_$c/score.json
done
# Video-MME offline at a 16K visual budget:
uv run --locked trace-vl v3 eval --config configs/v3/native_budget16k.yaml --records data/v3/videomme.jsonl \
  --video-root /data/videomme/videos --output outputs/v3/videomme_b16k
```

Gate: the Video-MME result is within about 2 points of the model card. The card reports 69.0 in
thinking mode; we run non-thinking, so record our number as the reference. Post per-task accuracy,
tokens/step, KV MB and answer latency.

## M1: modules and one training step

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl v3 check --config configs/v3/native_full.yaml   # "pass": true
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl v3 profile --config configs/v3/v3_tkm.yaml --mode v3 --steps 600
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl v3 train --config configs/v3/stage1.yaml \
  --records data/v3/train.jsonl --video-root /data/videos --output outputs/v3/m1 --max-steps 1 --clip-seconds 180
```

`data/v3/train.jsonl` uses the same record format with `"split": "train"`. MEVA train questions
(`convert-meva --split train`) work immediately. Optional fields: `rules` + `alert_spans` train the
alert head; `captions` train the event head. Check `outputs/v3/m1/metrics.jsonl`: finite losses,
`peak_gb` under 30, `gate_abs_mean` moving off 0.

## M2: Stage 1 (2 × 5090)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked torchrun --standalone --nproc_per_node=2 -m trace_vl v3 train \
  --config configs/v3/stage1.yaml --records data/v3/train.jsonl --video-root /data/videos \
  --output outputs/v3/stage1 --clip-seconds 120
uv run --locked trace-vl v3 eval --config configs/v3/v3_tkm.yaml --mode v3 --checkpoint outputs/v3/stage1/modules.pt \
  --adapter outputs/v3/stage1/lora --records data/v3/ovo.jsonl --video-root /data/ovo --output outputs/v3/ovo_v3_stage1
```
