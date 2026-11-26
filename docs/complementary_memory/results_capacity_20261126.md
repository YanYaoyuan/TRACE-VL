# Fresh R128 versus R32 capacity diagnostic — 26 November 2026

**The frozen 300 s NLL primary is inconclusive, with a favorable point estimate for R128.** Fresh R128-minus-R32 NLL is **-0.06837 [-0.13835, +0.00968]**. Accuracy is **77.06% versus 73.16%**, a **+3.90 pp [-0.87, +9.09]** difference. Neither primary-horizon interval establishes superiority. This does not establish equivalence or absence of useful capacity headroom.

The budget rises from **640 to 2,176 visual tokens (3.4x)**, so the result is a capacity-policy tradeoff, not a matched-budget writer result or a new trained model. R64, writer training and the later equal-token coarsening design were not run.

All **77 sources / 385 questions**, **2,310 fresh option vectors**, and **1,155 saved-R32 reproduction checks** completed at inference revision `aeea1d9bf7950f0c1977f19f3b05ee1fdc661610`. The archive, three saved reference sets, records, source budgets and phase receipts were independently checked; **48 aggregate statistics** were recomputed. Every R32 reproduction has zero option-logit gap, choice changes and mean/max-absolute NLL drift. The smoke-cost formula and exact remaining four-shard schedule were independently recalculated too.

## Frozen same-run primary and all horizons

Both capacities use recent8 and the unchanged NativeStream Reservoir law, with memory seeds 17/29/43. Six live caches share the same frozen visual encoding, source prefix, native coordinates and weights; each cache evolves its own state. Metrics are computed per seed, averaged within source, and paired across all 77 sources (10,000 bootstrap draws, seed 7). These are memory-sampling seeds, not training seeds. Five delays repeat 77 anchored facts; 385 questions are not independent facts.

Negative NLL favors R128, positive accuracy favors R128. Only the **300 s NLL** row is the frozen primary. Other intervals are secondary, pointwise, unadjusted for multiple comparisons and do not replace it.

| Delay, s | R32 NLL | R128 NLL | R32 accuracy | R128 accuracy | NLL difference [95% CI], nats | Accuracy difference [95% CI], pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6004 | 0.6114 | 75.76% | 74.89% | +0.0111 [-0.0531, +0.0703] | -0.8658 [-3.8961, +2.5974] |
| 15 | 0.6067 | 0.5345 | 76.19% | 79.65% | -0.0721 [-0.1297, -0.0119] | +3.4632 [+0.0000, +7.3593] |
| 60 | 0.6293 | 0.5577 | 72.73% | 77.06% | -0.0716 [-0.1385, -0.0012] | +4.3290 [+0.0000, +9.0909] |
| 180 | 0.6554 | 0.5664 | 77.06% | 78.35% | -0.0890 [-0.1536, -0.0216] | +1.2987 [-3.0303, +5.6277] |
| 300 | 0.6685 | 0.6002 | 73.16% | 77.06% | -0.0684 [-0.1384, +0.0097] | +3.8961 [-0.8658, +9.0909] |

The 15/60 s accuracy lower bounds are machine-level positive residues (~2.9e-18), reported as **zero**, not strict evidence of accuracy superiority. The secondary NLL intervals at 15/60/180 s favor R128, while the primary remains inconclusive. Zero-crossing short-horizon intervals do not establish noninferiority. No monotonicity of capacity or horizon effects is assumed.

Primary paired SE: **0.03739 nats**, approximate MDE80 **0.10470 nats**. The 300 s interval accommodates both a substantial improvement and a small worsening. These planning statistics do not create another post hoc gate. No automatic training decision follows.

## Co-equal capacity and timing costs

| Policy | Retained steps at 300 s | Visual tokens | Mean KV tokens | Mean KV MiB | Recurrent/conv bytes | First option-score seconds/query/seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R32 | 40 | 640 | 1036.56 | 32.3925 | 51904512 | 0.1556 |
| R128 | 136 | 2176 | 3510.37 | 109.6990 | 51904512 | 0.1534 |

These visual-token totals omit timestamp/delimiter tokens; actual KV tokens/bytes are reported separately. Steady steps equal min(prefix, 40/136) and logical append-before-eviction maxima are 41/137. Recurrent/conv bytes remain constant. Peak allocated CUDA memory across all six caches and shared model/encoding is **9.8432 GiB**.

The first-answer time includes a disposable cache fork, question-tail LM forward and option-letter scoring; deterministic re-answer/audit overhead is separate. R32 is always executed before R128. Fixed order, warming, shared-host load and reference PyTorch convolution/gated-delta kernels make timings descriptive, **not a speedup claim or deployment latency benchmark**. No kernel/package optimization was introduced.

## Cross-run secondary references

Saved B0 window40 and 1c protected32 were not rerun here, so their numerical drift remains unmeasured. They do not supply the primary baseline. All policies evolve their own states; none of these contrasts is a fixed-state causal intervention.

| Delay, s | Saved comparison arm (R128 minus arm) | NLL difference [95% CI], nats | Accuracy difference [95% CI], pp |
| --- | --- | ---: | ---: |
| 15 | saved_window40 | +0.0398 [-0.0442, +0.1230] | -2.1645 [-8.2251, +3.8961] |
| 15 | saved_protected | +0.0133 [-0.0476, +0.0719] | +0.8658 [-1.2987, +3.8961] |
| 300 | saved_window40 | -0.1927 [-0.3237, -0.0622] | +9.5238 [-0.4329, +19.9134] |
| 300 | saved_protected | +0.0024 [-0.0884, +0.0831] | +2.1645 [-3.0303, +7.7922] |

R128 versus protected32 at 300 s has similar NLL point estimates (0.6002 versus 0.5978), but its interval crosses zero; that does not prove a tie, equivalence or a memory-efficiency advantage. Privileged protection of a specific target set is not an optimal-set bound, and full-KV gaps do not decompose additively into capacity and selection effects.

## Integrity, execution and limits

- All R32 seeds on all five delays reproduce saved Test A: **385 vectors per seed**, max/quantile option-logit gaps 0, changed choices 0, mean/max-absolute gold NLL drift 0. Retained recent/history indices, KV token/byte budgets, state bytes and absolute next positions match. This is direct observed provenance for these R32 runs; it does not retrospectively bound every other arm/run.
- Native eight-step parity has exact packing and maximum option-logit gap **0.1736221**, within the unchanged 0.5 gate; it is not a long-stream numerical error bound. All source/media/runtime/model/checkpoint/helper/reference hashes were checked. New question records contain actual packing-prefix/head/tail digests; old per-question packing digests were absent.
- Six deterministic re-answers and live cache/state/ownership/RNG checks passed. The representation, model, LoRA and temporal schedule stayed frozen. No training was performed.
- The frozen Plain LoRA2500 reader saw short/eight-step reads in training. Larger KV changes context distribution. Questions expose a cutoff, recent KV can encode older context, source/creator dependence remains unmeasured, and this is reused development data rather than official OVO/StreamingBench or a fresh held-out benchmark.
- Smoke ran **12:29:36–12:51:42 Asia/Shanghai**, **22.09 min**. The conservative full estimate was **2.853 h** (component estimate 2.405 h). The estimate and remaining source/shard counts were independently recomputed and were below 4 h20 before full launch.
- Full ran **12:51:42–15:33:38**, **2.699 h**, below the enforced **4 h20** ceiling; smoke was below its **60 min** ceiling. Phase alarms/monotonic checks, signal-shielded owned-group cleanup and non-resettable failed-phase receipts were in the frozen source. No timeout occurred. All worker exit codes are zero, all GPUs were released, and the smoke bundle remains preserved.

| Full GPU/shard | Worker wall seconds, including startup |
| --- | ---: |
| 0 | 9709.7 |
| 1 | 9193.4 |
| 2 | 9093.3 |
| 3 | 9408.7 |

Both inference-head hosted checks passed **568 tests / 5 skipped**, Ruff and repository smoke. A100 startup passed **26 tests** and Ruff; the launcher repeated the same required CPU checks before inference. The reviewer conditionally cleared smoke after the watchdog-cleanup fix; the recorded resolution is the tested `aeea1d9` patch, without claiming a separate explicit review of that patch. No tests/lint/model code ran on the authoring PC; independent verification used saved JSON and NumPy only.

## Next research boundary

The result supports investigating the coverage/detail tradeoff, but does not yet validate a trained method. An exact-640-visual-token target would keep recent8x16 fixed and compare historical32x16 against historical128x4. It requires a real causal visual-feature encoding/retention design: averaging old contextualized KV is not equivalent. R64 is deferred as a resource choice, not ruled out. Fresh-source acquisition and final sample sizing require a frozen effect target and paired variance; an acquisition floor of 150 sources is not a guarantee of power for small effects.

## Provenance

- Input: `de153d253208ee07bacb21d4feb1b7435c4df3f717f4ef0ca2164a735d5a0707`
- Archive: **1,151,353 bytes**, SHA-256 `7a01e86b6b580e6197d5d5b157c979f46d9a7074f331c21998488be118ce7c5f`
- Report SHA-256: `b8e938b73d97387c445a8c4f25e47996af5b51a48149f7c2ed99880177abf857`
- Run: `/data/yyy/tracevl-runs/reservoir-capacity-aeea1d9-001`
- [Frozen protocol](reservoir_capacity_protocol.md)
- [Aggregate report](results_capacity_20261126/report.json)
- [Independent verification](results_capacity_20261126/verification.json)
- [Execution, cost and phase/worker receipts](results_capacity_20261126/execution.json)
