# Writer controls on the complete fixed77-source cohort

The audited reference3dc8c83 completed385queries/77sources: text42/77,
protected64/77, reservoir57/77 at300s. This supports investment in a writer
ablation; the reservoir/protected gap interval crosses0, so storage is not an
established unique bottleneck. Keep every target and all original data unchanged.

## Fixed design before new writer predictions

One continuous stream/source, same frozen Plain LoRA2500, original fullPTS2fps
schedule, all77targets x delays0/15/60/180/300s. One pass feeds all policies.
The six previous arms must exactly replay full predictions, including fp32/bf16
logits, packed frame indices and RGB identities, for all385queries. Smoke tests
shortest/longest2sources x5delays first. Full execution gates integrity only,
never whether new policies outperform. No changed labels/questions/selection,
no new reserved sources, training, target-aware normal writes or query-time encoder.

Add four arms to the old six,3850total arm predictions:

- time_coverage8: fixed temporal density compaction retaining original values;
- spatial_change8: fixed spatial feature-path triangle-excess compaction;
- reservoir_seed29_time8 and reservoir_seed43_time8: two extra memory PRNG streams.

The extra labels29/43 become source-specific seeds via the frozen digest namespace
`writer-controls-extra-reservoir-v1`. Original reservoir keeps its original seed.
These are3memory seeds for one frozen model/training seed. Deterministic new
writers run once, not duplicated as fake independent seeds.

Every normal policy independently stores32historical+4recent steps and reads
latest at most8pre-target steps/128tokens. New policies own detached cloned values,
retain exact times/indices/RGB identity and never replace old values by the current
frame. No query, answer or future target time goes to append. Prefix4+4, current8
and privileged snapshot8 remain the original explicitly separate controls.

## Analysis

Primary exploratory endpoint: each new deterministic writer at300s minus the mean
of the3reservoir accuracies within each source. Bootstrap77source units4000draws,
seed7; seeds are averaged within source, never treated as231independent videos.
Two pointwise exploratory contrasts, no architecture promotion or confirmatory
claim. Report all5delay curves, seed-wise outcomes, all target cases, precision
sensitivity and predefined Action/Object/annotation-frame strata descriptively.
No post-hoc seed, source or task selection. Preserve original single-seed result.

## Audit, payload and computation

Extend the shared evaluator through optional observation-only writer hooks. Default
six-arm calls remain unchanged. Tests exercise the actual evaluator with and without
hooks, identical base predictions and encoder counts; new reservoir exactly matches
existing AlgorithmR at identical seeds. FullGPU shared-baseline replay is mandatory.

Report per-policy actual tensor payload plus96logical metadata bytes/slot:
36 x (16 x2560 x2+96) =2,952,576B maximum. Global counters/PRNG/Python overhead
separate, not RSS. Controls can share base payload; new policies clone values, so
do not report union-of-frame IDs as total physical storage of joint experiments.
The joint experiment is multiple separate methods, not a deployable single bank.

Spatial descriptors are recomputed from retained values in capacity-bounded transient
CPUFP32 workspace. Report conservative tensor-expression workspace bound and append
wall time (including device synchronization), not an isolated GPU latency benchmark.
Do not hide this extra work behind a token-only comparison.

Write-only per-source eviction traces contain candidate indices/times and computed
scalar removal costs. No pixel or feature archive is stored; inference never reads
these traces. Temporal decisions and ranking can be independently reconstructed;
full verification that semantic costs originated from the stored feature tensors
requires independent re-encoding, not merely trusting scalar logs. Keep this audit
boundary explicit. Source trace hashes bind final receipts. Close streams and traces
on failure. A successful public report is written only after extended checks and
exact baseline replay pass; failures preserve pending files and stop full execution.

## Public benchmark route

This77-source run is development diagnosis, not official scores. Final evaluation
will preserve official prompts/questions/timestamps/scoring, explicit modalities,
frame/token budget, state size and runtime. OVO-Bench backward tracing and original
StreamingBench are candidate primary suites; Video-MME long is complementary
long-video generalization, not by itself streaming proof. Keep the previously used
250source development boundary explicit and reserved sources untouched until the
method is frozen. Visual-only subsets cannot be called whole omni-modal totals.

Official references checked2026-11-25:
- https://github.com/JoeLeelyf/OVO-Bench
- https://streamingbench.github.io/
- https://github.com/MME-Benchmarks/Video-MME
