# B0 newest-40 comparison — run started 25 November 2026

**The frozen 300 s NLL primary favors Reservoir over a newest-40 window at the same retained visual-step budget.** This supports useful temporal allocation beyond simply enlarging a recent window in this diagnostic. It does not validate a learned writer or establish state-conditioned utility; coverage and cross-question transfer remain separate tests.

All **77 sources / 385 questions** completed at inference revision `5e6252476849799867219dbe1653d8b00a9f27a6`. The full archive was hash-verified and independently checked against the completed Test A records, with **770 new window prediction vectors**, ten auxiliary Reservoir17 reproduction vectors and **84 recomputed aggregate statistics**. No training was performed.

## Frozen primary and policy comparison

The primary is **Reservoir minus window40** gold option-normalized NLL at 300 s. Negative NLL favors Reservoir. The three original Reservoir seeds (17/29/43) are averaged as metrics within source; 77 sources are paired and bootstrapped with 10,000 draws, seed 7. Accuracy is secondary and positive accuracy differences favor Reservoir. Five delays repeat the same anchored fact rather than providing 385 independent facts.

| 300 s comparison | NLL difference, nats [95% CI] | Accuracy difference, pp [95% CI] |
| --- | ---: | ---: |
| reservoir_minus_window40 | -0.1244 [-0.2326, -0.0220] | +5.6277 [-3.4632, +14.7186] |
| window40_minus_window8 | -0.0249 [-0.0855, +0.0342] | +0.0000 [-7.7922, +7.7922] |
| reservoir_minus_window8 | -0.1493 [-0.2644, -0.0418] | +5.6277 [-3.4632, +15.1515] |

Primary paired SE: **0.0535**; approximate MDE80 (2.8 × SE): **0.1498 nats**. MDE80 is a planning statistic, not an additional decision gate. The secondary accuracy interval crosses zero, so the positive accuracy point estimate does not establish accuracy superiority.

| 300 s policy | Mean gold NLL | Accuracy | Mean retained steps | Mean KV tokens | Mean KV MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| reservoir | 0.6685 | 73.16% | 40.00 | 1036.56 | 32.3925 |
| window40 | 0.7929 | 67.53% | 40.00 | 1044.79 | 32.6498 |
| window8 | 0.8178 | 67.53% | 8.00 | 212.21 | 6.6315 |

Both 40-step policies use 16 pooled visual tokens per step (640 at 40 steps). Timestamp/delimiter token counts and KV bytes can differ. Each policy evolves its own recurrent state under its own KV history: this is a **policy comparison, not a fixed-state causal effect**.

## Reproduction, integrity and retained target indices

| Reproduced arm | Queries | Max option-logit gap | Changed choices | Mean gold NLL drift | Max absolute NLL drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| reservoir17 | 10 | 0 | 0 | 0 | 0 |
| window8 | 385 | 0 | 0 | 0 | 0 |

Window40 minus window8 at 0 s: NLL **-0.0320 [-0.0933, +0.0260]**; accuracy **-5.1948 [-12.9870, +1.2987] pp**. Frozen damage flag (lower NLL interval bound > +0.10): **False**. Absence of the flag is not noninferiority.

| Delay, s | Window40 retains target last step, sources | Window40 retains any snapshot step, sources | Reservoir mean target-last-step rate |
| --- | ---: | ---: | ---: |
| 0 | 77/77 | 77/77 | 1.0000 |
| 15 | 77/77 | 77/77 | 0.1991 |
| 60 | 0/77 | 0/77 | 0.1472 |
| 180 | 0/77 | 0/77 | 0.0823 |
| 300 | 0/77 | 0/77 | 0.0606 |

These are timestamp/index audits, not semantic evidence certification. No source was filtered by retention or model outcome. All five delay analyses remain in the aggregate report; none replaces the frozen 300 s primary.

## Validation, computation and limitations

- Native eight-step parity: exact packing; maximum option-logit gap **0.1736221**, within the unchanged 0.5 gate. Short parity does not bound long-stream drift.
- Every source reproduced window8 under the unchanged 0.25 gate; both fixed smoke sources also reproduced Reservoir17. Original helper/model/checkpoint/runtime/record hashes were checked. Old per-question head/tail packing digests were not recorded, so drift in reused Reservoir seeds not rerun remains unbounded; the window8 check does not prove them equivalent.
- New queries record packing-prefix/head/tail digests and are answered on isolated forks. Both window re-answers were deterministic and live cache/state snapshots were verified. Steady retained steps are bounded at 8/40; logical append-before-eviction maxima are 9/41. Recurrent-state bytes stay constant. These logical bounds differ from CUDA peak memory.
- **39,028** encoded steps; peak allocated CUDA memory **9.0885 GiB**. Timing is LM-plus-budget-audit, not Test A LM-only latency. Reference PyTorch causal-convolution/gated-delta fallbacks remain; no optimized throughput or deployment claim is made.
- Frozen Plain LoRA2500 was trained on short clips/eight-step reads. Continuous streams and longer KV introduce distribution shift. Questions reveal their historical cutoff; the deployable policies do not use it to filter reads. Recent KV can still encode old context.
- This is reused development data, with possible unmeasured creator/source dependence, not an official/unseen benchmark score or evidence of a new trained method. No automatic training decision follows.
- Both exact inference-head hosted checks passed: 553 tests / 5 skipped, Ruff and repository smoke. A100 startup passed 21 tests and Ruff; the full reference preflight passed. No tests, lint or model code ran on the authoring PC; independent verification used saved JSON and NumPy.

## Provenance

- Frozen input: `e40fc6f3290603cf2086e0a24955d828346406a1f64cc86611d27a65c380c543`
- Archive: **800,402 bytes**, SHA-256 `12366edbffa781c3b3c8a3e8568c79d7ac885654b34bda9c190bd191ef7e004b`
- Report SHA-256: `c3c0c324f058f8ae12d5fb95f12dff2944cb477a5e547496badc9ccbc3ed8654`
- Run: `/data/yyy/tracevl-runs/b0-window40-5e62524-001`
- [Frozen aggregate report](results_window40_20261125/report.json)
- [Independent verification](results_window40_20261125/verification.json)
- [Frozen protocol](window40_protocol.md)
