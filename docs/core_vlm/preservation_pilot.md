# Core video VLM: capability-preserving recipe pilot

This is core video-to-language QA training on Qwen3.5-4B, not an alert head or a harness. It is a controlled pilot, not large-scale training, and the temporal adapter is an architectural baseline whose novelty/benefit is not established.

## Evidence and question

The previous matched-mixture pilot improved NExT temporal QA vs the original grid backbone by 7.75 pp (plain) and 8.29 pp (temporal), but temporal vs plain was only +0.54 pp with an interval crossing zero. OVO HLD fell from 28/69 correct to 16/69 (plain) and 14/69 (temporal). Recent-window and order diagnostics do not establish a unique temporal mechanism. These are one-seed development results; they do not justify scaling the old recipe.

Test whether retaining original answer distributions on general video-QA training examples prevents loss of backbone capability while preserving temporal QA gains. Then test whether the temporal module adds value under that same recipe. A KL penalty is a standard recipe control, not a proposed paper contribution.

## Arms and objective fixed before the run

| Run name | Trainable model | Objective |
|---|---|---|
| `ce` | ordinary LoRA | answer CE only |
| `plain` | ordinary LoRA | answer CE + 0.5 × preservation KL |
| `temporal` | causal temporal adapter + LoRA | same preservation objective |
| `original_grid` | original frozen Qwen3.5-4B + deterministic grid pooling | evaluation only |

All three training arms start afresh from the same original backbone and identical LoRA initialization. Same frozen mixture, sample order, rank-8 LoRA, learning-rate schedule (2e-5 initial), 1,000 updates, 4 GPUs, four microbatches per GPU per update, 16 observed frames, 20 spatial tokens per two-frame step. Final checkpoint is primary; no best-checkpoint selection. Added temporal parameters and FLOPs are reported and are not claimed equal.

KL is `T² KL(p_original || p_student)`, T=2, over full-vocabulary distributions at gold-answer and EOS positions, averaged per question. It applies only to the `general` training family. The original teacher shares frozen weights with the student: LoRA disabled, temporal module omitted, eval mode, no gradient/cache. Adapter state and individual module training flags are restored before student forward/backward. No second 4B model is resident. The `ce` control executes the identical teacher forward and KL computation with weight zero, to control the extra teacher computation. This is answer-position anchoring, not a guarantee of general capability retention or full generative distillation.

Primary contrasts: `plain − ce` isolates this recipe change; `temporal − plain` tests architecture under it. Both are also compared to `original_grid`.

## Data and answerability labels

Reuse `/nas-data/vlm/yyy/tracevl-data/core-mixed-temporal-97272fb`: 2,000 general and 2,000 temporal training questions. Source exclusion audit, manifests and video bytes are verified; archives are not rescanned and no video/model is downloaded. Fixed external data remain `/nas-data/vlm/yyy/tracevl-data/core-transfer-v1-f1c19a7`. NExT validation tests in-dataset generalization after official NExT training; OVO is external development. OVO final half stays unused. Existing populations and earlier results are never overwritten.

**Default: preservation only. No verified answerability pair file is currently available.** The launcher prints this and records `answerability_supervision_active=false`. Do not call this answerability-supervised training. Arbitrary crops, shuffled video, missing frames, or mismatched video do not automatically establish that a question is unanswerable. OVO HLD is never converted into training supervision.

Optional `TRACE_REVIEWED_PAIRS=/absolute/file.jsonl` activates a separately identified paired-supervision variant for all three arms. Each source must be an existing frozen training question. Review the question, all candidate answers, and the exact sampled frames of both views. The positive view must support the original answer; the insufficient view must not reveal it through remaining frames or question wording. Document all necessary evidence intervals and a human reviewer/reason. Program checks cannot certify a semantic judgment.

Each JSONL record has this schema (placeholder hashes are not executable labels):

```json
{
  "source_id": "existing-frozen-training-id",
  "reviewer": "reviewer-id",
  "reason": "Describe supporting evidence and why the other view cannot answer.",
  "question_and_sampled_frames_reviewed": true,
  "video_sha256": "exact-video-bytes-sha256",
  "evidence_intervals": [[20.0, 22.0]],
  "views": [
    {"answerable": true, "observation": {"start": 0.0, "end": 24.0}, "query_time": 24.0, "decoded_rgb_sha256": "hash-of-reviewed-sampled-RGB"},
    {"answerable": false, "observation": {"start": 0.0, "end": 10.0}, "query_time": 10.0, "decoded_rgb_sha256": "hash-of-other-reviewed-sampled-RGB"}
  ]
}
```

Decode with `core_vlm.data.decode_observation` at the configured steps/pixels to obtain the exact frames/hashes for review. Intake redecodes those views and rejects changed hashes, held-out IDs, identical views, or disagreement between declared evidence intervals and sampled timestamps. Each pair adds two examples, with identical option position within the pair and explicit insufficient-evidence choice. Option position is deterministic per pair. Use a new run directory if adding/changing the review file. These examples change sample exposure vs the default pilot; never conflate the two runs.

## Evaluation and continuation

Identical frozen OVO and NExT observations for every arm. Report restricted-choice accuracy separately from generated-answer strict accuracy/format validity; pair by source video (4,000 bootstrap resamples, seed 7). Main architecture gate retained: temporal must gain at least 2 pp on NExT temporal vs preserved plain AND original-grid, with CI lower >0. OVO backward/realtime and NExT descriptive must have CI lower >−2 pp vs both.

Additional capability gate: both preserved arms' HLD and non-HLD accuracy must separately have CI lower >−2 pp vs original-grid. Small HLD size can make this inconclusive. Report all subgroups regardless of pass/fail.

Before any training, freeze up to 200 non-HLD/non-unknown-option questions per external dataset (hash order, seed 17). Append an insufficient-evidence option with balanced positions, keeping the source gold answer. Evaluate all four arms and report abstention choice rates, choice/strict accuracy and paired rate differences. These are **source-answerable probes**, not human-verified evidence sufficiency at sampled frames, official benchmark scores, or camera false-alarm rates. An all-abstaining model fails ordinary QA; HLD alone cannot qualify a result. Review these diagnostics separately even if the numeric gates pass.

Intervals are exploratory, unadjusted, one-seed development evidence. No automatic larger run is authorized by this script. If the recipe helps but temporal does not, report recipe gain without attributing it to architecture.

## A100 operation

Pin the reviewed commit in `TRACE_REV` and extract `scripts/launch_core_preservation_a100.sh` with `git show`. The existing checkout need not switch branches. The launcher uses the existing UV environment, offline model files, and a separate source snapshot and output directory.

Stages: CPU tests → frozen data/media hash checks → three two-update DDP smoke runs plus checkpoint evaluations → three full 1,000-update runs → original/trained evaluations and refusal probes → reports. A failure stops the queue. Repeating the same command resumes validated checkpoints/results; changed configuration/code is rejected in the existing directory. Optional `TRACE_PREPARE_ONLY=1` or `TRACE_SMOKE_ONLY=1` stops at that stage.

Default output: `/nas-data/vlm/yyy/tracevl-runs/core-preservation-<commit7>-001/`. Temp files go there, not the nearly-full system `/tmp`. Large media and weights remain on NAS. Required: 30 GiB free on each selected GPU. This needs no extra model, dataset, or annotations downloaded.

Return `reports.tar.gz` and the top-level operator log. The archive contains `REPORT.md`, `report.json`, `training-audit.json`, `smoke-ok.json`, data reuse/supervision identities, frozen external/probe manifests, prediction files, metrics and subprocess logs, excluding model weights/source/temp files. Completion marker: `VERIFIED CORE PRESERVATION PILOT` followed by `REPORT BUNDLE`. CPU tests do not establish that the real 4×A100 run succeeds; the mandatory smoke checks it before full training.
