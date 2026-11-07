# v3 runbook (M0 → M2)

Code: `src/trace_vl/v3/`. Every command runs on your machine; none were run on the dev server.
Post each result as a comment on issue #4 using the template in `experiments.md` §7.

| File | Role |
|---|---|
| `backbone.py` | Qwen3.5-2B wrapper: per-step vision encoding, incremental LM calls, hybrid-cache eviction, LoRA |
| `positions.py` | Qwen3.5 MRoPE positions for streaming (matches `get_rope_index`) |
| `modules.py` | CST adapter, transition encoder, transition tracker, retention scorer |
| `kv.py` | full-attention KV policies: full / window / fifo / drift / tkm |
| `session.py` | streaming session (native or v3), forked question answering, per-step alert answers |
| `alert.py` | Jev-style alert head (noul / choice), latent connections (layer taps, GDN state reads) |
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
(`convert-meva --split train`) work immediately. Optional `alerts` field trains the alert head:
`[{"id": "abnormal", "kind": "noul", "text": "Is anything abnormal happening?", "spans": [[start, end]]},
{"id": "incident", "kind": "choice", "text": "Which incident?", "options": ["none", "fall"], "label_spans": [[1, s, e]]}]`. Check `outputs/v3/m1/metrics.jsonl`: finite losses,
`peak_gb` under 30, `gate_abs_mean` moving off 0.

## M2: Stage 1 (2 × 5090)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked torchrun --standalone --nproc_per_node=2 -m trace_vl v3 train \
  --config configs/v3/stage1.yaml --records data/v3/train.jsonl --video-root /data/videos \
  --output outputs/v3/stage1 --clip-seconds 120
uv run --locked trace-vl v3 eval --config configs/v3/v3_tkm.yaml --mode v3 --checkpoint outputs/v3/stage1/modules.pt \
  --adapter outputs/v3/stage1/lora --records data/v3/ovo.jsonl --video-root /data/ovo --output outputs/v3/ovo_v3_stage1
```

## Phase 2: Stage 1 on 4 × A100-40GB (after M0/M1 passed on 2 × 5090)

Code functionality and protocol fidelity were verified on 2 × 5090 at `ac3fec5` (experiments.md §8). The
A100 machine has the storage for training data at scale.

```bash
git fetch origin && git checkout design/v3-transition-streaming && git pull   # PR #3 head
uv sync --extra dev --extra vlm --extra data && uv pip install flash-linear-attention
uv run --locked pytest -q tests/test_v3.py tests/test_v3_gpu.py
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl v3 check --config configs/v3/native_full.yaml --video <short.mp4>  # sm_80 sanity

# Data: LLaVA-Video-178K multiple-choice subsets (the trainer needs option-letter answers; open-ended QA is
# not used by Stage 1 yet). Download the *_mc_* json files plus their videos for the 0-30 s ... 2-3 min
# buckets; record dataset versions/licences in configs/datasets/downloads.json.
uv run --locked trace-vl v3 convert-llava-mc /data/llava-video/*/*_mc_*.json --video-root /data/llava-video \
  --output data/v3/llava_mc.jsonl
uv run --locked trace-vl v3 convert-meva --bench data/meva/bench --split train --output data/v3/meva_train.jsonl   # optional
cat data/v3/llava_mc.jsonl data/v3/meva_train.jsonl > data/v3/stage1_train.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --locked torchrun --standalone --nproc_per_node=4 -m trace_vl v3 train \
  --config configs/v3/stage1_a100.yaml --records data/v3/stage1_train.jsonl --video-root /data \
  --output outputs/v3/stage1_a100 --clip-seconds 180
```

Evaluate checkpoints (`modules.pt` + `lora/`) with `v3 eval --mode v3 --checkpoint ... --adapter ...` on
the held-out `val` split, OVO-Bench and Video-MME, against B0/B1/B2 at equal visual tokens
(experiments.md §2), and post every run to issue #4.

Alert-head data and evaluation: section "Alert head (anomaly sets)" below; Stage 2: next section.

## Stage 2: long streams through the streaming cache (after an M2 Stage 1 checkpoint)

Stage 1 never sees the KV policy or the Gated DeltaNet state over minutes. Stage 2 trains the inference path
itself (`src/trace_vl/v3/train_stage2.py`): steps -> CST + transition encoder -> chunk LM forward on the hybrid
cache -> TKM eviction -> delayed questions on a forked cache (shared KV, cloned GDN states) -> CE. Chunks of
`stage2.chunk_steps` (also cut at each question) are backpropagated and then every cache is detached
(truncated BPTT). At every chunk end all older KV groups (transition slots and whole steps, so the pooled B4
arm is scored the same way) are rescored with their current age; the retention scorer gets BCE against
"a question still to come is about this group's time". Distractor clips that are never asked about, and
evidence whose question was already answered, are the negatives (`ret_pos` / `ret_neg` in the metrics).
The same scores drive eviction during training and at inference (`kv.RetentionBook`). Gradients are all-reduced by hand (streams have
different chunk counts, so DDP hooks would deadlock).

```bash
# 1. Streams: LLaVA-MC clips of the same frame grid concatenated to ~5 min; each question is asked right after
#    its clip (25%) or 30-240 s later while other clips stream, and names its interval ("from 37 to 61 seconds").
#    40% of the clips are distractors (never asked about): retention negatives. Check `queried_seconds_fraction`
#    and `streams_reaching_target` in the report; the last stream of each frame-grid bucket can be short.
#    Records with query_time (e.g. MEVA train) become one stream per video with questions at their own times.
uv run --locked trace-vl v3 build-stage2 data/v3/llava_mc.jsonl --video-root /data/llava_video \
    --output data/v3/stage2_train.jsonl --stream-seconds 300
uv run --locked trace-vl v3 build-stage2 data/v3/llava_mc.jsonl --video-root /data/llava_video \
    --split val --output data/v3/stage2_val.jsonl --stream-seconds 300
# 2. Smoke (1 x 5090): two updates; check finite grad norms, q_ce, evidence_kept, kv_tokens_end, peak_gb.
uv run --locked trace-vl v3 train-stage2 --config configs/v3/stage2.yaml --streams data/v3/stage2_train.jsonl \
    --video-root /data/llava_video --init outputs/v3/stage1/modules.pt --adapter outputs/v3/stage1/lora \
    --output outputs/v3/stage2-smoke --max-steps 2
# 3. Run: 2 x 5090 (configs/v3/stage2.yaml) or 4 x A100 (stage2_a100.yaml); B4 arm with stage2_b4.yaml from
#    the Stage 1 B4 checkpoint, identical streams and seed.
uv run --locked torchrun --nproc_per_node=4 -m trace_vl.cli v3 train-stage2 --config configs/v3/stage2_a100.yaml \
    --streams data/v3/stage2_train.jsonl --video-root /data/llava_video \
    --init outputs/v3/stage1-a100/modules.pt --adapter outputs/v3/stage1-a100/lora --output outputs/v3/stage2-a100
```

Held-out evaluation, representation x retention at the same KV budget (window 32 s + 4,096 older tokens):

```bash
for arm in transition b4; do for policy in tkm mass fifo; do
  uv run --locked trace-vl v3 eval-stage2 --config configs/v3/stage2$([ $arm = b4 ] && echo _b4).yaml \
    --streams data/v3/stage2_val.jsonl --video-root /data/llava_video --mode v3 --policy $policy \
    --checkpoint outputs/v3/stage2-$arm/modules.pt --adapter outputs/v3/stage2-$arm/lora \
    --output outputs/v3/s2eval-$arm-$policy
done; done
```

`tkm` = learned retention, `mass` = surprise mass (non-learned), `fifo` = recency. Both arms use the same group
bookkeeping, so learned-vs-non-learned retention and transition-vs-pooled representation are separate
factors. `summary.json`: accuracy (whole-stream bootstrap CI), accuracy by delay bucket, `evidence_kept`.

Metrics per update (`metrics.jsonl`): `q_ce` / `q_acc` (delayed questions), `evidence_kept` (fraction of
visual tokens streamed inside the asked-about interval still in the KV cache at question time; the retention
scorer should raise it above the `fifo` policy at the same budget), `mean_delay`, `retention_bce`,
`p_steps_with_active`, `expected_p_tokens`, `kv_tokens_end`, `ret_pos` / `ret_neg`, `peak_gb`. The alert head is
frozen in Stage 2 (no optimizer state, no weight decay); multi-GPU gradient sync keeps `grad=None` for
parameters no rank used.

Known limits (by design, documented for the review): question CE reaches only the chunk it is asked in, not
evidence steps of earlier chunks (truncated BPTT); the retention scorer carries that signal from annotations.
Eviction happens at chunk ends during training but after every step at inference (same policy, same budget).
Retention features (both paths): age, surprise mass, hits, slot-vs-step; the drift feature is 0 (chunked
forwards have no per-step drift). Labels come from clip-level evidence windows (coarse). No KD
teacher in Stage 2 (a full-token teacher over 5-10 min streams is too costly); the alert loss is not used yet.

## Transition-sensitive evaluation (added for M2)

```bash
# OVO-Bench incl. forward-active tasks (official prompts; REC count digits / SSR, CRR Yes-No as in the official scorer)
uv run --locked trace-vl v3 convert-ovo /data/ovo/ovo_bench_new.json --forward --video-root /data/ovo --output data/v3/ovo_all.jsonl
# TOC-Bench (object state / temporal consistency): answer_final.json from github.com/cjzcjz666/toc_bench,
# videos (15.2 GB) from huggingface.co/datasets/anonymous-video-benchmark/toc_bench (check its licence first)
uv run --locked trace-vl v3 convert-tocbench /data/toc/answer_final.json --video-root /data/toc --output data/v3/toc.jsonl
# MEVA long-gap (our benchmark)
uv run --locked trace-vl v3 convert-meva --bench data/meva/bench --split test --output data/v3/meva_test.jsonl
```

Evaluate each with `v3 eval` for native / transition / B4 exactly as for Video-MME; report per-task accuracy
(`score` groups by `source/task`). TOC-Bench here is a local option-likelihood protocol (ordering items by
generation), not the official TOC-Bench extractor; OVO forward follows the official prompts and scoring rules
with first-token Yes/No likelihood for SSR/CRR.

## Alert head (anomaly sets)

ShanghaiTech Campus (13 fixed cameras, 130 abnormal events) and CUHK Avenue (16 train / 21 test videos) carry
frame-level anomaly labels. **Licence:** neither project page states one (they ask for a citation), so use
them for academic research only: download from the official pages
(svip-lab.github.io/dataset/campus_dataset.html, cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html),
never commit or redistribute frames or labels, and cite both. Both are small; please report the downloaded sizes.

```bash
# ShanghaiTech: testing/frames/<video>/*.jpg + testing/test_frame_mask/<video>.npy + training/videos/*.avi.
# The standard protocol trains only on normal videos; the head needs positives, so labelled test videos are
# split video-disjointly. Prefer the Zhong et al. (CVPR 2019) weak split list if you have it (--train-list,
# one video name per line); otherwise a seeded 50/50 hash split.
uv run --locked trace-vl v3 convert-anomaly shanghaitech --root /data/shanghaitech --output data/v3/sht.jsonl \
    --prefix shanghaitech/ [--train-list /data/shanghaitech/SH_Train.txt] --normal-train 100
# Avenue: official .mat masks need scipy (not a project dependency):
uv run --locked --with scipy trace-vl v3 convert-anomaly avenue --root /data/avenue --prefix avenue/ \
    --output data/v3/avenue.jsonl
# Stage 1 with alert supervision: QA records + alert-only records, 25% alert samples (no CE/KD on those).
uv run --locked torchrun --nproc_per_node=2 -m trace_vl.cli v3 train --config configs/v3/stage1_alert.yaml \
    --records data/v3/llava_mc.jsonl data/v3/sht.jsonl --video-root /data --output outputs/v3/stage1-alert
# Evaluate on the held-out videos: alert head (no decoding) vs the LM asked Yes/No every step.
uv run --locked trace-vl v3 eval-alerts --config configs/v3/stage1_alert.yaml --records data/v3/sht.jsonl \
    --video-root /data --checkpoint outputs/v3/stage1-alert/modules.pt \
    --adapter outputs/v3/stage1-alert/lora --method head --output outputs/v3/alerts-sht-head
uv run --locked trace-vl v3 eval-alerts --config configs/v3/stage1_alert.yaml --records data/v3/sht.jsonl \
    --video-root /data --method lm --output outputs/v3/alerts-sht-lm-zeroshot
```

`--video-root` is the folder record paths are relative to. For the mixed training run, keep both datasets under
one parent (e.g. /data/llava_video and /data/shanghaitech under /data), convert with `--prefix shanghaitech/`
(and LLaVA records with their own prefix if needed), and train with `--video-root /data`.
Alert-only training crops are sampled (half of them placed to contain the start of a random abnormal span,
half uniform over the video) with absolute times, and `alert_positive_fraction` in the metrics reports the
share of positive decisions actually presented. `eval-alerts` and `eval-stage2` write one atomically
published file per completed video/stream (`videos/`, `streams/`; an interrupted unit is recomputed), and
frame folders enter the run id by per-frame content hash. `eval-alerts` writes per-step rows (`label` = frame label at the step centre, `p`, `ms`) and `summary.json`:
micro step-level AUROC and AP, whole-video bootstrap CI, and ms per decision. Step-level AUROC at 1 step/s is not
directly comparable with published frame-level AUROC at 24-25 fps; compare methods within this protocol.

