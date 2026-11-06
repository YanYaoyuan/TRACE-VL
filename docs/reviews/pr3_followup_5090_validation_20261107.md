# PR #3: independent 2×5090 follow-up validation, 2026-11-07

Latest fully exercised source: `74eb90838b3dc33fc451a57a972c3567f1e8c6a6`.
Base: `5acfbe77c27d01c9a96dada66d8a0840c71086b5`.
Each revision was materialized in a separate source directory; the dirty primary worktree was preserved. No merge was performed.

Training/runtime regressions are fixed through this revision. **M0 official-processor input alignment remains open:** native and training serialization omit the outer visual start/end tokens retained by the public processor. See the reference control below.

## Environment

- Two RTX 5090 cards, 32 GiB each; single-card capacity tests used GPU 0 without display load.
- UV environment, Torch 2.8.0+cu128, torchvision 0.23.0+cu128, Transformers 5.20.2, PEFT 0.21.2.
- Real local snapshot of Qwen/Qwen3.5-2B, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- No flash-linear-attention / causal-conv1d: GDN uses the reference PyTorch implementation. Do not extrapolate these timings to optimized deployment.
- `peak_gb` is allocated CUDA bytes / 1e9, not total board usage. Timings below identify synchronization and scope.

## Regression results

| Check | Revision | Result |
|---|---|---|
| Complete suite, real CUDA available | 74eb908 | 86 passed |
| Unmodified gradient checker, state read forced on, dropout off | 74eb908 | total unclipped norm 13.9983, finite |
| Reject adapter removal before prediction reuse | 0171f07 | pass |
| Expire track before matching after 100 s gap | 0171f07 | new onset 100, age 0 |
| 180 s single-GPU training, two updates | 0171f07 | 14.242 GB; previously OOM |
| 120 s single-GPU training, two updates | 0171f07 | 11.108 GB |
| 180 s training, noul + choice, GDN read on every sample | 74eb908 | 14.436 GB; two updates; all finite |
| Two-GPU DDP, noul + choice, 6 s, two updates | 74eb908 | pass; rank-0 peak 6.338 GB |
| Default alert config, 6 s, one update | 74eb908 | pass; peak 5.948 GB |

180 s tests use batch=1, accumulation=1, 448² pixels/frame, CST dim 768 / 4 layers / 12 heads / window 16, no TBPTT. Synthetic capacity data is not an alert-quality benchmark. At 74eb908, the 180 s mixed-alert run had module norms 22.933 → 4.425 and LoRA norms 26.958 → 23.900; cumulative trainer elapsed 22.0 / 35.6 s includes teacher, preprocessing, forward/backward and optimizer.

The 0171f07 alert smoke ran 60 synthetic steps with real GDN state shape `[3,16,128,128]`. CUDA-synchronized medians: LM 16.83 ms, alert-state read/head 1.19 ms, entire step 38.49 ms. Batch vs individual-question logits agree within 1.2e-7; ad-hoc classification preserves stream cache length. The head is untrained; probability values do not establish accuracy.

Two failures newly found at 0171f07 were reproduced, reported, then verified fixed at 74eb908:

1. Choice confidence used BCE inside BF16 autocast, which raises. Fixed with pre-sigmoid confidence and BCE-with-logits.
2. `gradient_check` encoded text under no-grad inside the student autocast scope, triggering checkpoint metadata mismatch. Moving encoding outside autocast fixes it. A separate control forcing state reads verifies the corrected path.

## Real training data and held-out QA

Existing LLaVA-Video / ActivityNet source videos, gold yes/no answers converted to two choices, no fabricated distractors. Train: 40 QA / 16 videos; held out: 13 QA / 6 disjoint videos. Video-MME test examples never enter training.

| Method | Updates | Correct | Visual tokens/step | Mean KV MiB |
|---|---:|---:|---:|---:|
| Native | 0 | 11/13 | 196.62 | 55.81 |
| v3 + LoRA, a761f09 | 12 | 11/13 | 35.35 | 12.00 |
| v3 + LoRA, 74eb908 | 12 | 12/13 | 35.35 | 12.00 |

Latest real-data train peak 6.615 GB, elapsed 37.2 s. This is a plumbing diagnostic, not official ActivityNet-QA. One changed answer on 13 questions and 12 training updates cannot establish a reliable quality improvement. Saved modules and LoRA were reloaded for evaluation.

## 600-step resource profiles (frozen a761f09, synthetic, untrained modules)

| Policy | Visual tokens/step | Final KV MiB | Peak GB | Synchronized median whole-step ms |
|---|---:|---:|---:|---:|
| native full | 196 | 1447.18 | 7.117 | 65.56 |
| native window | 196 | 79.70 | 4.614 | 51.22 |
| native drift | 196 | 127.68 | 4.704 | 56.15 |
| v3 TKM | 52.5 | 71.95 | 5.030 | 58.32 |

The entire v3 step was slower than the simple window baseline in this run. Reduced LM tokens and KV do not by themselves prove end-to-end acceleration or retained quality. Raw curves and unsynchronized phase timings are included in the JSON; use the synchronized values for latency comparisons.

## Video-MME reference control and open M0 issue

Official HF first archive shard, 30/900 videos, 90/2700 QA across short/medium/long; no subtitles. All 30 decoded successfully. Native at 448² / max 80 two-frame steps scored 50/90, zero errors. This selected pilot cannot be compared directly to model-card scores.

Public HF processor control uses its default frame sampling/resizing, non-thinking, HF OpenCV decoding because PyAV/torchcodec are absent, public `apply_chat_template`, and full model forward with `logits_to_keep=1`. Native receives the identical preprocessed temporal-patch tensors and timestamps. Scores are constrained A-D likelihoods; unconstrained greedy first-token validity is logged separately.

At 74eb908, this 90-question comparison gives:

- Public full forward: 50/90.
- Current native with matched pixels/sampling: 47/90.
- Prediction agreement: 73/90; greedy first token is an option only 69/90 times.
- Same 50/90 aggregate as the earlier low-budget run is coincidental, not evidence of protocol equivalence.

The public processor retains an outer `<|vision_start|>` before the first timestamp and an outer `<|vision_end|>` after the last patch. Current streaming omits both. On `0ay2Qy3wBe8.mp4`, sequence length is 12369 vs 12371, first mismatch at index 3. Question tails match.

FP32 on that real video's three questions: current native vs processor max log-prob difference **0.6305**. A diagnostic session that only adds the outer start to prompt prefill and the outer end on a query-cache fork reduces the max to **2.37e-4**; all three predictions agree. This isolates a serialization issue, not BF16 drift. The existing equality gate assembles its reference by hand using the same layout and therefore misses it.

Requested fix: align streaming, student, and teacher with the public processor; add a processor-originated equality reference. Close the outer video only on question forks so the live stream can continue. The diagnostic subclass is not a patch to the author branch.

## Artifacts

- `docs/reviews/pr3_followup_results_20261107.json`: commit-labelled raw metrics, profiles, errors, predictions, data provenance and source hashes.
- `scripts/validation/pr3_alert_backward_probe.py`: independent CUDA checkpoint/autocast probe; use fresh output, local model config and `--sha`.
- `scripts/validation/pr3_official_processor_control.py`: public-processor control; `--dtype float32 --limit-videos 1`; optional `--align-outer-markers` is the diagnostic only.
- Branch: `review/pr3-5090-validation-20261107`. Model/video payloads remain local.

No full surveillance FPR/FNR, alert delay, cross-camera benchmark, multi-seed ablation, or publication-level quality claim has been established.
