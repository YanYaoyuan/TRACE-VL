# TRACE-VL v2 on fixed-camera inspection: benchmark and run guide

TRACE-VL is a streaming video understanding model with transition evidence memory (see the
[model doc](architecture.md)). Fixed-camera property inspection is its first application and
evaluation setting. This page covers that setting: the claim, the MEVA benchmark, the label audit and
how to run it.

Status: implemented, not yet run on GPU or real video. Nothing below is a result.

## Claim to test

In fixed-camera property surveillance, the questions that matter are about **history**: was
something left and later removed, how long has it been there, how often did it happen. The current
frame cannot answer these questions and uniform frame sampling misses short events. Our hypothesis:

> Under the same visual-token budget, a memory of **transitions** (where the scene changed, with
> before/after crops and onset/end times) answers long-gap history questions better than uniformly
> sampled frames, a recent-frame window, or a text caption log written by the same VLM. On
> questions the current frame can answer it does no worse.

The earlier TRACE-VL design (frozen LLM + randomly initialized bridge) never learned to use the
video. This design keeps the pretrained VLM's own image pathway and changes only *what* evidence it
sees, so visual grounding is inherited rather than relearned.

## Method

```text
video (2 fps) -> dual-background change detector -> region transitions (before crop, after crop,
                 t_start, t_known, t_end, t_end_known) -> bounded store (active changes kept last)
question/rule -> rank transitions (region overlap, window, recency) -> pack under token budget
              -> Qwen3-VL (current frame + evidence) -> option log-probabilities
```

- **Detector** (`inspection/change.py`). Fast and slow backgrounds. Static change means "different
  from the slow background but stable", confirmed after `confirm_seconds`. The slow background is
  frozen inside active regions, so a bag left for 20 minutes is not absorbed. A transition resolves
  when its region matches the slow background again. Onset is the start of the continuous
  difference (when the object or person arrived), not the confirmation time. Scene-wide changes
  (lights) reset the detector instead of flooding memory. Strictly causal: a transition exists only
  from `t_known`, and its end is hidden until `t_end_known`.
- **Store** (`memory.py`). Capacity-bounded. Evicts resolved changes first, active ones last.
- **Packing** (`evidence.py`). Every method gets the same current frame (`frame_tokens`) and the same
  extra budget (`budget_tokens`). Images are resized to multiples of the VLM token unit, so the
  planned count is exact, and evaluation logs the token count the processor actually produced.
- **VLM** (`vlm.py`). Qwen3-VL-2B-Instruct by default. Answers by scoring the option letters
  (one forward pass, no parsing). LoRA SFT on the answer letter only.

Methods compared at equal budget: `none` (language prior), `current` (memory-free), `uniform`,
`recent`, `textlog` (captions every 10 s by the same VLM; caption compute reported separately),
`transition` (ours), and the ablations `transition_after` (no before crops) and `transition_meta`
(times and locations as text only).

## Benchmark (public data only): MEVA long-gap questions

MEVA (CC BY 4.0) has fixed facility cameras (school, admin, bus station, hospital, parking) with
activity annotations. Consecutive 5-minute clips of a camera are chained into streams of up to an
hour. Two facts shaped the builder:

1. `person_closes_facility_door` is essentially never annotated (53 opens and 0 closes in a
   45-clip sample), so door state is **not** used.
2. Object tracks cover only their activity, and the put-down and the later pick-up of the same object
   have different IDs. Persistence is therefore reconstructed by linking each put-down to the first
   later pick-up at the same place. This linking is heuristic and must be audited (see below).

Question types (all answers are option letters):

| type | memory needed | example |
|---|---|---|
| `placed_and_removed` | yes | In the last 10 minutes, did someone put down an object inside the red box and later pick it up again? |
| `duration` | yes | ... How long has it been there? (<1 min / 1-5 / 5-15 / >15 min) |
| `count` | yes | How many times did someone put down an object inside the red box in the last 5 minutes? |
| `still_there` | no (control) | Earlier, someone put down an object inside the red box. Is it still there now? |

The negatives are hard: the same region queried before anything happened there, and objects that
were placed and are still present. Answers are balanced per type. Splits are camera-disjoint
(70/10/20) and every question records the gap since the decisive event. `censored` marks puts with
no linked pick-up (the object may have been removed without annotation); report results with and
without them (`score --exclude-censored`).

**Label audit (required before any claim):** fill in `label_correct(y/n)` in `audit.csv` (40 test
questions per type, with clip names and in-clip times) by watching the clips, and report the
per-type label accuracy next to every result.

Alerts: the rule "object left for at least N minutes" is replayed causally from the transition
logs and matched to linked put-downs. Precision is a lower bound, because the detector also fires
on unannotated changes. `--verify` adds VLM before/after verification.

## Running on 2x RTX 5090

```bash
uv lock                                  # new `vlm` extra (transformers, peft, torchvision cu128)
uv sync --extra dev --extra vlm
uv run --locked pytest -q tests/test_inspection.py          # CPU, seconds

# 1. Annotations only (~15 GB streamed, a few hundred MB kept); then questions
uv run --locked trace-vl inspection meva-fetch --output data/meva/annotations --workers 8
uv run --locked trace-vl inspection meva-build --annotations data/meva/annotations --output data/meva/bench
cat data/meva/bench/report.json          # streams, questions per type/split, gap buckets, censored share

# 2. Videos for the test split first (~140 MB per 5-min clip); aria2c reads the generated list
uv run --locked trace-vl inspection meva-videos --bench data/meva/bench --split test --output data/meva/urls_test.txt
aria2c -i data/meva/urls_test.txt -d data/raw/meva -j 8 -c

# 3. One CPU pass per stream (detector + snapshots + caption frames)
uv run --locked trace-vl inspection prepare --bench data/meva/bench --video-root data/raw/meva \
  --output data/meva/prepared --split test --workers 8
cat data/meva/prepared/prepare_report.json   # check "errors" and "missed"

# 4. Zero-shot evaluation, one shard per GPU (resumable), then caption log and scoring
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl inspection evaluate --shard 0/2 \
  --methods none current uniform recent transition transition_after transition_meta &
CUDA_VISIBLE_DEVICES=1 uv run --locked trace-vl inspection evaluate --shard 1/2 \
  --methods none current uniform recent transition transition_after transition_meta &
wait
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl inspection caption --shard 0/2 &
CUDA_VISIBLE_DEVICES=1 uv run --locked trace-vl inspection caption --shard 1/2 &
wait
CUDA_VISIBLE_DEVICES=0 uv run --locked trace-vl inspection evaluate --methods textlog --shard 0/1
uv run --locked trace-vl inspection score --predictions outputs/inspection/eval --output outputs/inspection/eval/score.json
uv run --locked trace-vl inspection alerts --min-duration 120 --output outputs/inspection/alerts_120.json

# 5. LoRA SFT on the train split (prepare --split train first), both GPUs; train each method
#    separately for a fair fine-tuned comparison
CUDA_VISIBLE_DEVICES=0,1 uv run --locked torchrun --standalone --nproc_per_node=2 -m trace_vl \
  inspection train --method transition --budget 1024 --output outputs/inspection/lora-transition
```

Use `configs/inspection/qwen3vl_4b.yaml` for 4B on a 5090 and `qwen3vl_8b.yaml` on the A100s
(a scale check, not the main model). If a model ID does not resolve, check the exact Qwen3-VL
release name on the Hub.

## What would support or refute the claim

- `transition` > `uniform`, `recent` and `textlog` in macro balanced accuracy at matched mean visual
  tokens on `placed_and_removed`, `duration` and `count`, with camera-bootstrap CIs that do not
  overlap, and the gap widening with longer gaps (`by_gap`).
- `transition` ≈ `current` on `still_there` (the control).
- `none` near chance everywhere; otherwise the questions leak through their wording.
- `transition` > `transition_after` > `transition_meta` would support the before/after
  (transition) structure itself, not just "more relevant crops".
- Results must hold with `--exclude-censored` and come with the audited label accuracy.
