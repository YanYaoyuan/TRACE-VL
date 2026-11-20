# Pilot-004: 1,000-update A100 experiment (2026-11-09)

The operator supplied the completed report bundle. The original M2 gate and both exploratory continuation checks **failed**. Longer training improved accuracy, but did not establish an advantage over B4.

## Download and provenance

- [Full original report bundle](pilot004-reports-20261109-150430.tar.gz) — 700,963 bytes, 12 files.
- SHA-256: `277ea7574007e391b532a758be759c261fea36aacd5cad5629d1a5c68ee90024`
- [Full scoring summary](summary.json); [operator digest](pilot004-digest.txt).
- Training source: `a3f8c290d00f61db24d7cbedf973765945e92f65`.
- Fresh training: transition, after and B4, each 1,000 updates on four A100s; 30-second clip limit; one seed.
- Checkpoints: 100 / 300 / 1,000; **1,000 is the primary comparison**. The first 100 updates have a different cosine-learning-rate trajectory from pilot-003.
- Evaluation: 488 questions, 29 videos. Primary uncertainty: paired video-cluster bootstrap.

The bundle includes the digest, summary, selector-diagnostics.json, complete pilot-004.log, all three metrics.jsonl files, all three configs, identity.json, and base-revision.txt. It does **not** include checkpoint tensors, completion manifests, per-question predictions or evaluation shard logs.

The archive and its extracted files were compared byte for byte. Local report inspection confirmed 1,000 sequential metric records per arm, P-step tokens fixed at 20 in all logged P-step records, and 101 token-diagnostic records per arm. The operator log ends with VERIFIED. No independent A100 replay or checkpoint/prediction audit was performed from this bundle.

## Accuracy

| Checkpoint | Native | Transition | After | B4 |
|---|---:|---:|---:|---:|
| 100 | 88.32% | 85.66% | 85.45% | 88.32% |
| 300 | 88.32% | 87.09% | 88.93% | 88.52% |
| **1,000 (primary)** | **88.32%** | **87.91%** | **89.14%** | **89.96%** |

Final paired differences (percentage points):

| Comparison | Difference | 95% CI |
|---|---:|---:|
| Transition - B4 | -2.049 | [-4.321, 0.000] |
| After - B4 | -0.820 | [-3.049, 1.434] |
| Transition - After | -1.230 | [-3.279, 0.617] |
| Transition - Native | -0.410 | [-3.067, 2.259] |
| After - Native | +0.820 | [-1.856, 3.512] |
| B4 - Native | +1.639 | [-0.611, 3.901] |

All three trained arms have 46.912743 mean LM tokens/step at every checkpoint. B4 has the highest final point estimate, but its improvement over native is not statistically established by this interval. The transition/B4 interval touches zero; do not describe this final comparison as a clear significant loss. Neither slot arm establishes superiority or equivalence to B4.

Within this run, 100-to-1,000-step accuracy increased by 2.25 pp (transition), 3.69 pp (after), and 1.64 pp (B4). This supports improvement under extended training, not a proof of architectural benefit. After's small advantage at step 300 is not the predeclared final outcome.

## What the diagnostics say

Means over the final ten logged token-diagnostic samples (rank-0 training samples, not a fixed held-out probe):

| Metric | Transition | After | B4 |
|---|---:|---:|---:|
| Minimum gist spatial entropy | 0.999992 | 0.999995 | ~1.000000 |
| Mean active-slot spatial entropy | 0.9963 | 0.8861 | ~1.0000 |
| Mean emitted-token cosine similarity | 0.9790 | 0.9152 | 0.7296 |
| Emitted-token effective rank | 1.88 | 2.47 | 12.92 |

- Both learned-gist arms remain close to spatially uniform in the logged samples. B4 does not emit the learned gist, so its unused-gist entropy is not evidence of a B4 failure.
- After's slots become more selective (final-step mean entropy 0.913; final-ten mean 0.886). Transition's remain close to uniform. It is no longer appropriate to characterize both arms as equally unchanged.
- Transition shows highly similar emitted tokens and low effective rank. This supports a redundancy hypothesis, but does not establish the sole cause of the accuracy gap.
- After's part-pool entropy is also measured (final-ten mean 0.680). Its emitted effective rank remains low despite more selective attention.
- Important measurement limitation: in the a3f8c29 pooled branch, B4's emitted-token statistics are computed **before** match_norm, whereas the slot-arm statistics use normalized emitted tokens. Cosine similarity is theoretically invariant to positive per-token rescaling, but effective rank is not. Recompute rank after normalization on identical held-out inputs before treating the rank gap as a fully controlled mechanistic comparison.
- Metrics across steps use different training samples; use fixed held-out clips for a direct representation comparison.

Final-100-update average losses:

| Arm | CE | KD |
|---|---:|---:|
| Transition | 0.3235 | 0.0956 |
| After | 0.3202 | 0.0878 |
| B4 | 0.3011 | 0.0646 |

Training duration from each arm's elapsed counter: transition 74.3 min, after 78.3 min, B4 75.9 min; total training ~3.81 hours, excluding evaluation and setup. Reported rank-0 peak PyTorch allocation: 7.23 / 7.31 / 7.18 GB. These are not total nvidia-smi device usage or all-rank peaks.

## Proposed next decision

Do not launch another longer run or mix all available datasets solely because this pilot improved.

1. Preserve this as a negative result against the original architecture gate.
2. Use the existing checkpoints on fixed held-out clips to inspect attention and actual emitted tokens; align normalization for the rank diagnostic.
3. Propose **one** next controlled intervention. A grid-gist control for the after arm is a plausible candidate: replace only its eight learned gist tokens with eight spatial pools, retaining its twelve slot/part tokens, the P-step budget and matched B4 baseline. This is a hypothesis to discuss, not an implemented or approved new run.
4. Audit the already extracted data and prepare independent OVO-Bench / Video-MME evaluation. Keep dataset expansion separate from the architecture intervention, and verify source-video overlap before using broader LLaVA/NExT-QA data.
5. Keep the original M2 gate unchanged and distinguish exploratory continuation from a paper-level superiority claim.

Full raw curves and diagnostic trajectories are included in the archive for the developer to inspect.
