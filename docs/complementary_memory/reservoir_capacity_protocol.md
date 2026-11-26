# B0 phase 2: fresh R128 versus fresh R32, v1

Freeze code/input/protocol identities before outcomes. Agreement: PR #14,
comments 7181816919 / 7181831577. This is evaluation only; no new training.

## Question and estimand

Do 128 historical Reservoir steps outperform 32 when both keep recent8?
Use unchanged Plain LoRA2500, media/PTS schedule, 4x4 pooled visual tokens,
native prompt/positions, Torch/Transformers/runtime, and all 77 reused development
sources / 385 questions. Five delays repeat 77 anchored facts, not 385 independent
facts. Run six independently carried NativeStream caches in one execution:
R32 and R128, each at memory seeds 17/29/43 via the original memory_seed law.
No target protection, query-conditioned selection or retrospective retention.

Primary: **fresh R128 minus fresh R32 at 300 s, gold option-normalized NLL**.
Compute metrics separately per seed, then average within source; pair 77 sources
and bootstrap 10,000 draws, seed 7. Negative favors R128. A zero-crossing interval
is inconclusive, not equivalence. Report paired SE/MDE80. Accuracy and
0/15/60/180 s are secondary pointwise intervals; no noninferiority or monotonicity
claim and no automatic training decision. R64 is deferred as a resource priority;
a null R128 result does not rule it out. Reader context shift remains a limitation.

Saved window40 and protected32 at 15/300 s are unreproduced cross-run secondary
references with unmeasured numerical drift. They never supply the primary baseline.

## Budgets and costs

R32 retains at most 40 steps (640 visual tokens); R128 at most 136 (2,176), a
**3.4x visual-token budget** at capacity. At earlier prefixes keep min(prefix, cap+8),
with transient maxima 41/137 during append. Timestamp/delimiter tokens make actual
KV bytes differ further. Record actual steps/tokens/bytes, constant recurrent/conv
state bytes, peak allocated CUDA memory, per-GPU wall time, LM/budget-audit time,
and first-answer time per query/seed. Re-answer/audit overhead is reported separately.
Capacity gains are a cost/quality tradeoff, not a free improvement or a writer claim.

## Provenance and integrity

Seal Test A/B0/1c full input/report identities, all per-question hashes, original
helper hashes, runtime versions, model/LoRA identities, media and native schedules.
Every R32 seed on every source/delay reproduces saved reservoir{seed}:S1K1:
**1,155 option vectors**. Stop above max logit gap 0.25 without loosening the gate;
report actual max/quantiles, choices and full-report mean/max-absolute NLL drift.
Compare query-time retained indices, KV tokens/bytes, state bytes and positions too.
No free 0 s identity exists between R128 and R32 after history first exceeds 32.

Use the original eight-step exact-packing native parity check, max logit gap 0.5;
it is not a long-stream error bound. All six re-answers are deterministic;
disposable forks leave live cache/state, ownership and RNG unchanged. Capture
actual new packing-prefix/head/tail digests; do not manufacture old packing proofs.
Same-run R32 reproduction quantifies this run's historical drift, not every previous
run or another policy. Recent KV can encode older context. Reused development data
and separately evolved states do not establish a fixed-state causal effect, unseen
benchmark performance, event-memory ability or learned utility transfer.

## Validation and enforced execution budget

Authoring PC runs no tests, lint or model code. Use hosted CI and A100 CPU checks;
peer reviews protocol/code before GPU smoke. Preserve all historical files.

Smoke: the original two fixed sources, six live streams, first two selected GPUs.
Expose integrity/memory/cost only; exported bundle excludes question predictions
and outcome summaries. Full uses four GPU shards, skipping only the two completed
sources with identical sealed input. Check >24 GiB free on selected GPUs at phase
start. No runtime/kernel/package optimization is made within this experiment.

Hard **smoke ceiling: 3,600 s**. Hard **full ceiling: 15,600 s (4 h20)**, including
GPU check, worker/model startup, reporting and bundling; CPU preflight/identity
sealing precedes phase clocks. The main-thread alarm and monotonic worker deadline
are enforced, not just recorded. Cleanup sends TERM only to explicitly launched
process groups, allows a shared 10 s cleanup grace, then KILLs surviving owned groups.
Other user jobs are never discovered or terminated. Interrupted/expired/failed
phases are incomplete and cannot reset their deadline in the same run directory.

After smoke, estimate full cost before launch from exact remaining source/shard
steps and query counts. Component estimate uses pooled smoke per-step/per-query
rates. Conservative estimate uses the **maximum source rate for each component**,
plus maximum uninstrumented source residual scaled by steps, maximum *matched*
worker startup residual, and a fixed 120 s reporting/bundle allowance. No prediction
is read for this decision. Hold if conservative estimate exceeds 15,600 s. The
estimate is not a guarantee; the actual watchdog enforces the ceiling. No automatic
seed/capacity/protocol substitution follows a budget hold. Record smoke/full costs
separately and retain the smoke bundle.

Complete results require **77 sources / 385 questions**, **2,310 new option
vectors**, and **1,155 baseline reproductions**. Publish only after independent
archive/hash/record/bootstrap verification. No partial outcome peeking.

## Deferred method work

A later exact-640-visual-token target is recent8x16 + history32x16 versus
recent8x16 + history128x4. Coarse visual features must be encoded by a valid causal
policy before entering the LM; pooling contextualized KV is not equivalent.
That design, coarsening audit, fresh-source acquisition and power-based sizing
are separate work, not implemented by this capacity protocol.
