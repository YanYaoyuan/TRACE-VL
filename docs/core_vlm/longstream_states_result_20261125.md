# Fixed77-source delayed state diagnostic: audited result, 2026-11-25

Run `core-longstream-states-3dc8c83-001` finished at11:17:22 Asia/Shanghai.
All385 queries/77 sources and2310 arm predictions completed. The full packet was
retrieved and independently audited: code/model/data identity, all250-source
selection, fullPTS schedules, seed/bank states, capacities, actual reads, option
logits and all scores.39,028 stream encoder calls plus6 native-parity calls;
24/2310 bf16/fp32 candidate-choice differences, fixed fp32 letter-head scoring.

Same77 one-target-per-source development questions at every delay; no success filter.
Frozen Plain LoRA2500,32 history+4 recent slots,128 read tokens. Original video
continues naturally; the original cutoff remains explicit in every question.

|Arm (correct/77)|0s|15s|60s|180s|300s|
|---|---:|---:|---:|---:|---:|
|Text|42|42|42|42|42|
|Latest8 at arrival|64|56|49|49|43|
|FIFO time read|64|64|42|42|42|
|Reservoir time read|63|55|55|54|57|
|Prefix time read|65|40|40|40|40|
|Protected snapshot8 (privileged)|64|64|64|64|64|

Primary paired snapshot-text: +28.57pp, exploratory source bootstrap95% interval
[15.58,41.56]pp,4000 draws/seed7;26 wins/4 losses. Passes the predeclared diagnostic
investment heuristic of>=10pp with exploratory lower bound>0. It is not a paper
acceptance or confirmatory significance gate.

Secondary reservoir300-text: +19.48pp, pointwise exploratory interval[6.49,32.47]pp.
This supports benefit from bounded persistent old visual values in this development
setting. It does not validate CausalPatchPool or the new writer prototypes.

Reservoir300-minus0: -7.79pp[-18.18,2.60]. Reservoir300-minus-protected:
-9.09pp[-19.48,1.30]. Both intervals cross0: do not call reservoir decay or a uniquely
writer-caused bottleneck established. Its curve is not monotonic. FIFO after60s
has77/77 empty pre-target reads and falls back to text; this is not retained memory.

At300s reservoir retains/reads35 of616 snapshot reference steps,47/77 targets have
none. Snapshot reference is not annotated semantic evidence; other stored frames
can still help. All shared model inputs and values, not captions of history, drive
these controls. Protected inputs are identical across delays by design and know
future target times; they are not a deployable writer or perfect semantic oracle.

All sources were previously used development videos; not official StreamingBench
or unseen validation. One frozen model and one memory seed per source; sources
can share creators/content. Question tags are not manual evidence certification.
Action39 and Object38 strata remain descriptive, without replacing the full cohort.

## Next experiment

Subsequent update: the writer controls below have now completed. Both failed to
demonstrate superiority; see [writer results](writer_controls_result_20261125.md)
and the [current consolidated status](research_results_20261125.md). The following
paragraph preserves the proposal as recorded after this baseline run.

Keep all77 sources/all5 delays, same frozen reader/sampling and32/4/128 budget.
Integrate time_coverage and spatial_change with exact shared-baseline replay;
prespecify additional memory seeds. Record actual state, auxiliary workspace,
latency and causality. New writers are CPU tested only; no GPUQA gain is claimed.
A metadata-only time_coverage preflight on all77 actual schedules retains35
reference steps/42 targets with none; it computes no QA, and no dummy features
are used to simulate spatial_change. These counts cannot predict an accuracy win.
No training scale-up or architecture promotion follows from this baseline result.

## Identity

- input:`090db3b555dd368f8d7478a2de7195408a4aa6788e451e02636fc4d91cc59e4b`
- report SHA256:`f2cd9ec0c4c401cca7af35c1093dfa453dfb1700c2b7688dd668ece50da6dd88`
- packet SHA256:`d775d83af6df3dabe901421bcf0eb2af468ae7662278ad9ce0283dedbd29a5a0`
- Workspace evidence:`DOC/experiment-reviews/longstream-states-3dc8c83-20261125/`
