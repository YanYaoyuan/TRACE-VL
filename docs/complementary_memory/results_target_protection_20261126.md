# Phase 1c target protection — 26 November 2026

**The frozen 300 s NLL primary is inconclusive.** Privileged protection improves the point estimate over Reservoir by 0.0708 nats, with a 95% interval of [-0.1588, +0.0222] for protected minus Reservoir. Accuracy is 74.89% versus 73.16%, a +1.73 pp difference [-5.19, +8.66]. Neither interval establishes superiority at the primary horizon. The interval also does **not** exclude an improvement of at least 0.05 nats: this result is not evidence that useful selection headroom is absent.

All **77 sources / 385 questions** completed at inference revision `9ee2d8f4634e776aa939d7ec9fa64a9ff20977cb`, producing **1,155 protected prediction vectors** and ten auxiliary window8 vectors. The archive and both saved reference sets were independently verified; **56 aggregate statistics** were recomputed from records with saved-JSON/NumPy math. No training was performed.

## Frozen comparison and all horizons

The policy has disjoint recent8, protected departed target steps P, and a Reservoir of other departed steps R. A causal privileged capture announces the eight current target steps without gold/question content. Each admitted protected step shrinks R by one; the final allocation is recent8 + protected8 + Reservoir24. A protected step does not increment the ordinary candidate count. Uniform thinning and ordinary draws use the same per-stream `random.Random`, so post-admission draw consumption determines subsequent samples. This is a privileged selection diagnostic, not a deployable writer.

Gold option-normalized NLL is computed separately for seeds 17/29/43, then metrics are averaged within source. All 77 sources are paired and bootstrapped with 10,000 draws, seed 7. Five delays repeat the same 77 anchored facts; they are not 385 independent facts. Negative NLL favors protection; positive accuracy favors protection.

| Delay, s | Protected accuracy | Reservoir accuracy | NLL difference, nats [95% CI] | Accuracy difference, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| 0 | 75.76% | 75.76% | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| 15 | 78.79% | 76.19% | -0.0855 [-0.1534, -0.0247] | +2.5974 [-0.4329, +6.4935] |
| 60 | 78.79% | 72.73% | -0.0912 [-0.1730, -0.0149] | +6.0606 [+1.7316, +10.8225] |
| 180 | 75.76% | 77.06% | -0.0756 [-0.1659, +0.0193] | -1.2987 [-6.4935, +3.4632] |
| 300 | 74.89% | 73.16% | -0.0708 [-0.1588, +0.0222] | +1.7316 [-5.1948, +8.6580] |

Only **300 s** is the frozen primary. All shorter horizons and accuracy intervals are secondary, pointwise, and unadjusted for multiple comparisons. At 60 s both reported intervals favor protection, which motivates follow-up but does not replace the inconclusive 300 s endpoint. At 180 s the accuracy point estimate goes in the other direction. Fifteen seconds is descriptive: no noninferiority margin or claim was specified.

The 300 s primary paired SE is **0.04649 nats** and approximate MDE80 is **0.13019 nats**. These planning statistics do not create another gate. The predeclared relevant-effect interval flag is `False`: the lower primary bound is below -0.05, so a 0.05-nat gain remains compatible with this cohort. No automatic training decision follows.

## Secondary comparison with newest-40

This contrast uses **B0's separately executed, unreproduced window40 reference**. Its cross-run numerical drift is unmeasured. Each policy evolves its own recurrent state; these are policy comparisons rather than a fixed-state causal intervention.

| Delay, s | Protected minus window40 NLL, nats [95% CI] | Accuracy difference, pp [95% CI] |
| --- | ---: | ---: |
| 0 | +0.0279 [-0.0148, +0.0711] | +0.4329 [-3.8961, +4.7619] |
| 15 | +0.0265 [-0.0203, +0.0740] | -3.0303 [-8.2251, +2.1645] |
| 60 | -0.0599 [-0.1552, +0.0392] | +3.4632 [-4.7619, +11.2554] |
| 180 | -0.1564 [-0.2915, -0.0191] | +1.7316 [-8.2251, +11.6883] |
| 300 | -0.1952 [-0.3304, -0.0547] | +7.3593 [-3.4632, +18.6147] |

At 300 s the protected versus window40 accuracies are **74.89% versus 67.53%**. The NLL interval favors protection; the accuracy interval crosses zero. This does not establish an advantage over the stronger Reservoir baseline at the frozen primary horizon. At 15 s newest-40 still contains the target; a zero-crossing interval does not establish a tie or noninferiority.

## Integrity and limits

- **231 zero-delay reproductions** (77 sources × three seeds) and all ten smoke-only window8 checks have **zero option-logit gap, zero choice changes, zero mean/max-absolute NLL drift**. These match saved query-time retained indices, token/byte budgets, state bytes and native next positions. Later-delay Reservoir and B0 window40 logits were not rerun; their drift remains unbounded. Old packing digests are absent.
- All protected query records retain all eight snapshot indices in recent or protected ownership. Three sets remain disjoint, steady retained steps equal min(prefix, 40), and logical append-before-eviction steps stay at most 41. Each step has 16 pooled visual tokens. Timestamp/delimiter tokens mean equal steps do not imply exactly equal KV bytes.
- At 300 s protected/Reservoir/window40 mean KV tokens are **1037.342 / 1036.558 / 1044.792**, respectively; mean recurrent-state bytes are **51,904,512** for each. Forks leave live cache/state and logical/RNG snapshots unchanged; repeated protected answers are deterministic.
- Native eight-step parity has exact packing and maximum option-logit gap **0.1736221**, below the unchanged 0.5 gate. This short check does not bound long-stream numerical drift. Reference/model/checkpoint/media/runtime/code hashes were checked; no package/kernel/decode changes were made.
- Target timing is privileged, questions expose a cutoff, and the frozen Plain LoRA2500 was trained on short clips. This is reused development data with possible creator/source dependence, not an official OVO/StreamingBench evaluation or a new trained method. The result does not establish question-blind predictability, cross-question utility-label transfer, event memory, or recurrent-state-conditioned writing.

## Execution and review audit

Smoke took **1007.6 s (16.79 min)**. Its component estimate for four-way full execution was **1.288 h**, with a conservative estimate of **1.776 h**. Immediately before full launch, free memory was **39,892 MiB on every GPU**. Four-way full execution ran from **09:38:24 to 11:13:00 Asia/Shanghai**, taking **5675.7 s (1.577 h)** including startup and the completed-smoke reload. Worker wall times are observed START/DONE times with up to one polling interval error:

| GPU / full shard | Observed wall seconds |
| --- | ---: |
| 0 | 5634.6 |
| 1 | 5226.7 |
| 2 | 5107.5 |
| 3 | 5433.1 |

These are execution observations, not an algorithmic speedup claim. Shared-host contention varied; other jobs later occupied all four GPUs, and the final experiment check found all four released. There were **39,028** encoded steps and peak allocated CUDA memory was **9.2152 GiB**. Timing includes budget/packing/fork audits. Reference PyTorch convolution/gated-delta fallbacks remain.

**Cost-review limitation:** the automatic continuation verified finite positive estimates, exact remaining work, integrity, and current free memory, but enforced **no hard duration ceiling**. A reviewer requested such a ceiling; it was not implemented in this completed run. We do not retrospectively claim that gate existed or passed. Future continuation protocols should specify an enforceable ceiling before launch. This workflow limitation does not change the frozen numerical comparisons.

Both exact inference-head hosted checks passed **560 tests / 5 skipped**, Ruff and repository smoke. A100 startup passed 25 tests and Ruff. No tests, lint, or model code ran on the authoring PC; independent verification used saved records only.

## Provenance

- Frozen input: `9e393a4c3ae549364baa6d7227bf93e24a69d82725c560c3e6413fe910501d0e`
- Archive: **910,663 bytes**, SHA-256 `62529c213ac5242dcad063d456dd5ae3dcb04bebb882bb36024aa2aaff06aaed`
- Report SHA-256: `9cf46c5bf8968e596f6f0d5241d91c514906568cee15950193bd1e886b213a28`
- Run: `/data/yyy/tracevl-runs/target-protected32-9ee2d8f-001`
- [Frozen protocol](target_protection_protocol.md)
- [Frozen aggregate report](results_target_protection_20261126/report.json)
- [Independent verification](results_target_protection_20261126/verification.json)
- [Execution, measured cost and per-GPU timing](results_target_protection_20261126/execution.json)
