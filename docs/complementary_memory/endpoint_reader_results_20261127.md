# Endpoint reader training and development evaluation: 27 November 2026

The matched 64-source/317-QA endpoint training pilot and the complete 77-source development evaluation are finished. **Neither prespecified 300-second co-primary NLL contrast excludes zero.** Historical retention still gives a positive point-estimate trend, but this weak native endpoint training recipe has not demonstrated additional memory-specific capability. Do not scale it to a multi-day run on the strength of these results.

Training runtime: `798eee2d09b323b987252fbf40d99af3ea825a6a`. Evaluation runtime: `b0a2fad2af8937388136b18543862556a26f91ce`. The evaluation input ID is `b84f06c7cb1e850a37b8519358ec7830945ce17c1bb52392a7c904439cb3925c`. All weights use the same original Plain initialization; training uses one seed, 64 source-mean AdamW updates and detached complete-source prefixes. Gradients adapt the question reader's LM LoRA, not the visual prefix or a learned writer. The weak labels have no verified historical-evidence requirement.

## Primary 300-second results

Each row contains 77 questions, clustered into 73 exact-video-byte groups for uncertainty. Memory uses source-hashed seed17 throughout this evaluation. The earlier three-seed Reservoir headline must not be compared as if it were this single-seed denominator.

| Checkpoint | Retention/read policy | Correct | Accuracy | Gold option NLL |
|---|---|---:|---:|---:|
| Starting Plain | Reservoir8+32 | 54/77 | 70.13% | 0.696484 |
| Endpoint-trained Reservoir | Reservoir8+32 | 56/77 | 72.73% | 0.684881 |
| Starting Plain | Continuous window40 | 52/77 | 67.53% | 0.792924 |
| Endpoint-trained window40 | Continuous window40 | 51/77 | 66.23% | 0.778103 |
| Endpoint-trained window40 | Reservoir8+32 | 56/77 | 72.73% | 0.685976 |

Prespecified co-primary comparisons, negative NLL favoring the first named system:

- Trained Reservoir/Reservoir minus trained window/window40: **-0.093222 NLL**, 97.5% CI **[-0.224008, +0.033574]**. Accuracy difference +6.49 percentage points; 95% exploratory CI [-2.76, +16.25] pp.
- Trained Reservoir/Reservoir minus starting Plain/Reservoir: **-0.011603 NLL**, 97.5% CI **[-0.030516, +0.005676]**. Accuracy improves by two questions, +2.60 pp; its 95% exploratory interval touches zero.

Under the same Reservoir policy, both trained checkpoints have identical per-question correctness at 300s; their NLL difference is only -0.001095, 95% CI [-0.006555, +0.004100]. Thus the +6.49 pp trained-system difference is not evidence that Reservoir-specific training caused the gain. It combines checkpoint and retention-policy differences. The matched-policy controls do not establish a training specialization benefit.

The window-trained checkpoint's within-window NLL improvement is -0.014821, 95% exploratory CI [-0.027466, -0.003172], while accuracy drops from 52 to 51 correct. Probability calibration and classification accuracy need separate reporting. This secondary result does not rescue either co-primary comparison.

## Other delays and controls

| Delay | Starting Plain / Reservoir | Trained Reservoir / Reservoir | Trained window / window40 |
|---|---:|---:|---:|
| 0s | 75.32% | 75.32% | 77.92% |
| 15s | 75.32% | 75.32% | 81.82% |
| 60s | 72.73% | 74.03% | 75.32% |
| 180s | 74.03% | 75.32% | 68.83% |
| 300s | 70.13% | 72.73% | 66.23% |

At 15s the trained-system NLL comparison goes the other way: +0.128871, 95% exploratory CI [+0.052704, +0.212543], favoring window40. Do not generalize the 300s point estimate to all horizons. Zero-delay questions are anchored-state perception proxies; later questions target the old state and do not measure current-scene perception.

Text-only accuracy also rises: 54.55% starting Plain, 57.14% trained Reservoir and 58.44% trained window. A **post hoc exploratory**, text-adjusted 300s Reservoir training comparison gives +0.002747 NLL, 95% CI [-0.017338, +0.021758], and 0.00 pp accuracy, 95% CI [-5.19, +5.26] pp. This is not a prespecified endpoint; it supplies no demonstrated visual-memory benefit beyond generic adaptation.

All 0/15/60/180/300s arms and contrasts are in [the aggregate report](results/endpoint_reader_development_20261127.json). The cohort was repeatedly used for development, has question-provided past cutoffs, and has only 73 certified exact-byte clusters; broader origin/reencoding independence remains unproven. This is neither an unseen cohort nor an official benchmark score. Null intervals are inconclusive, not proof of equivalence.

## Completion and independent verification

The four-GPU recovery completed at **19:16:58 Asia/Shanghai**, after 3h27m of that phase. All four workers exited0 on their first four-GPU attempt, with no further OOM retries. Each checkpoint completed77 sources/385 queries:231 source receipts,1,155 query records and4,620 score vectors. Peak owned CUDA allocation was9.0368GiB. The original Plain reader reproduced all four saved controls on every query with maximum logit gap0; exact native packing passed throughout, with maximum short cached/uncached gap0.175117 against the frozen0.5 bound. These short parity checks do not bound long-stream numerical drift.

An independent CPU script using no project scoring/analysis functions recomputed every arm mean and every95%/97.5% interval from saved logits. Maximum numerical discrepancy was0. All36 files preserved from smoke are byte-unchanged. [Verification summary](results/endpoint_reader_development_verification_20261127.json).

- Aggregate report SHA256: `e83e083d0bb2938ce828ef403c24f7f573a90ea848a6769666890b15c627fece`.
- Verification report SHA256: `385251ffcb374f63ca39b77551a75ee55c244597ecc3a029f22be285f497d7b2`.
- A100 artifact root: `/data/yyy/tracevl-runs/endpoint-reader-eval-b0a2fad-003`.

Earlier failures remain in separate state/log files: lint launcher setup, a28GiB free-memory reservation hold, and a shared-workload model-loading OOM with two workers per GPU. The successful recovery used one process per GPU, deterministic source-length balancing, and the same frozen evaluator/inputs/statistical protocol. No unrelated GPU task was terminated. Authoring-PC work was limited to static parsing and saved-record handling; no local tests, lint or model execution.

The next decision should concern training supervision and measurable history dependence. Preserve this pilot as a complete baseline; establish old-evidence-dependent training/reader headroom before expanding this exact native endpoint recipe. Complementary utility/writer claims and official OVO/StreamingBench evaluation remain separate, unfinished work.
