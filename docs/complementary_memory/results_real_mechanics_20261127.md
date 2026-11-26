# Real-processor delayed-history mechanics: passed short audit

The fixed real-processor engineering audit passed on both original Test A smoke
sources, each using its first16 encoder steps. Both delayed fine32 and coarse128
variants passed. This removes the short real-processor packing/readout blocker;
it is **not a new accuracy result, long-stream guarantee or trained method**.

## Frozen audit and results

Inference revision: `a2a21e826847453721769697977e45fe08489303`.
Input: `103a329d1054403a3f6c3795820533f8f33643d960b2389bd5fabd95eff43c5e`.
The [protocol](real_processor_mechanics.md) used one fixed synthetic placeholder
prompt, the unchanged frozen Plain LoRA, real HF processor layout, and the same
decoded observations as the per-step encoder. No benchmark question, options or
gold were used as model input; no accuracy, gold NLL or real-question prediction
was produced. No new-source extraction or training occurred.

| Original source | Fine32 maximum option-logit gap | Coarse128 maximum option-logit gap |
| --- | ---: | ---: |
| streamingbench-real:0024 | 0.1255512 | 0.0715828 |
| streamingbench-real:0199 | 0.0390701 | 0.1614208 |

Every case had bit-identical per-step/full-processor pixels, matching grids,
bit-identical mixed-resolution embeddings (maximum gap0.0), and exact native
positions against the independently constructed full-HF-layout reference.
Cached versus uncached synthetic option logits stayed below the frozen0.5
maximum-gap screen, and repeated questions were deterministic. The live history
ended at8 committed steps; recent indices8–15 were consumed exactly once on the
question fork. The runtime's state/RNG/feature isolation checks passed.

At this short boundary, fine/coarse readouts had394/298 actual KV tokens and
256/160 visual tokens. Recurrent/conv state was51,904,512 bytes in both. This
does **not** compare the intended saturated640-visual-token budgets: only eight
historical steps exist, neither capacity saturates and no retention eviction is
tested here. It also does not establish equal total KV memory.

## Execution and verification

The parent ran2026-11-27 00:43:32–00:44:15 Asia/Shanghai (**42.3868 seconds**),
within the hard900-second ceiling. GPU0 worker time was31.3208 seconds; peak
CUDA allocation was8.8040 GiB. GPU check and worker exited0. Parent and worker
were confirmed exited, with all four GPUs empty at the final read. No runtime
packages/kernels, metric thresholds or samples changed during the attempt.

Before execution, A100 CPU24 tests/Ruff and both exact-head hosted jobs passed
(574 tests/5 skipped, Ruff and repository smoke). CPU metadata preflight verified
model/LoRA/helper identities and both complete frozen schedules. Execution used
an immutable archive; it did not change the shared A100 checkout or completed
experiment source.

The developer's review was still pending. The
[execution receipt](results_real_mechanics_20261127/execution.json) explicitly
records `peer_review_received:false`: after the earlier deferral, the owner’s
standing A100/parallel-experiment authorization was applied to this bounded
engineering check, and the coordination change was posted before GPU output.
This is not represented as external review or approval for a quality run.

The complete archive was transferred with byte-count/SHA256 checks. A separate
saved-artifact verifier checked the sealed input, protocol, all148 Python source
hashes, report identity, all four numeric gates, causal index/budget arithmetic,
phase time and terminal process receipts. This is independent artifact/accounting
verification, not independent re-inference.

- Archive SHA256: `0638f242b89413ccb51b34bbdb168eb74e2383540038046865830d3fb28bc044`.
- Report SHA256: `1e7b4e5cff7c67d7f95b4f0c1522990df721a800f4cba661d0a8ecd8d5e1eb81`.
- Immutable source archive SHA256: `c6367eb2812008a13b0ed4023afcaf1af6b46ce79ac999f57df322965ee5a8bf`.
- [Report](results_real_mechanics_20261127/report.json),
  [source audits](results_real_mechanics_20261127/source_audits.json),
  [verification](results_real_mechanics_20261127/verification.json).

## What remains

The0.5 screen and16-step prefix do not bound drift over long streams or establish
quality. The proposed delayed history also changes historical contextualization.
A future quality protocol still needs matched delayed fine/coarse readouts,
original fully carried R32 as a strong separate control, actual total-KV/latency
accounting, fresh-data/power decisions and explicit interpretation. Passing this
audit launches no larger quality evaluation or training automatically.
