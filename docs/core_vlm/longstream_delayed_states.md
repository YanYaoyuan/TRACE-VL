# Anchored past-state delayed development diagnostic

The previous all26 past-event diagnostic had text19/26 and privileged snapshot21/26.
Inspection of every actual-input sheet found recent actions, persistent scenes,
small game icons and causal questions whose relevant episode precedes snapshot8.
Neither target timestamps nor privileged8-step snapshots certify semantic evidence.
Keep that cohort and all its results unchanged.

This protocol asks whether a larger, outcome-independent cohort of original
Concurrent Action/Object questions makes visual benefit and retention observable.
It is a controlled derived task, **not official StreamingBench** and not unseen
validation: every source belongs to the same previously used250-source development
boundary. No benchmark question is used for training.

## Selection, before missing PTS resolution or new predictions

`scripts/prepare_delayed_states_data.py` seals `selection-rule.json` first.
Resolve all23 missing fullPTS entries rather than excluding sources by cache status.
From all1250 questions on all250 original sources, select Concurrent Action Perception
or Object Perception questions with original cutoff>=60s and original media continuing
at least300s beyond cutoff. Per source choose the earliest eligible cutoff, breaking
ties by question ID. Keep every selected source; no answer/oracle correctness filter,
no model-output access in preparation. Cumulative Counting is a separate task and is
excluded by this task definition, before outcomes. Do not change original text,
options or gold. Add the same explicit original-cutoff instruction at every delay.

These annotation types are not manual evidence certification. For example, a question
about what was "mentioned" may require audio; repeated clothing/scenes can remain
inferable from later pixels. Keep such limitations visible, without post-hoc deletion.
The complete metadata-selected cohort is diagnostic, not a claim that every row
requires memory. The23 repaired metadata files and all previous cache provenance remain.

Hash source and /data copies, verify fullPTS eligibility, and reject exact media-byte
training overlap. Missing/corrupt inputs abort preparation; never silently drop rows.
Record all target/source IDs, hashes, code, weights, fullPTS schedule and read slots.
Full sequential selected-frame decode remains mandatory in the worker.

## Fixed computation

Original video streams once per source through original cutoff+300s. Query each
unchanged target at0/15/60/180/300s. Frozen Plain LoRA2500 and same input resolution,
sampling and grid4x4. Normal policies store32 historical+4 recent steps and read
at most8 steps/128 visual tokens. All normal historical readers are target-relative
and causal. No query-time encoder or persistent LM KV; each question starts fresh.
One deterministic memory seed per source, not multiple training seeds.

Six arms: text_only, recent8_now (later-frame control), fifo_time8,
reservoir_time8, prefix_time8, protected_snapshot8. Prefix uses only4 prefix+4 recent
slots, a deliberately smaller simple baseline. Protected snapshot8 knows the future
target time: a privileged reference, not an algorithm/semantic oracle. One target per
source means protected storage is at most8 saved+8 current steps; ordinary budget
remains32+4. Report per-policy actual state and logical bytes rather than joint control
payload or processRSS. Writer receives no questions, options, answers or target times.
The protected reference is explicitly the exception.

Smoke covers shortest and longest distinct sources, all5 delays. Gate only integrity,
native parity, counters, capacities and causal timestamps. Full run is not conditioned
on smoke accuracy. Reuse completed source records only with matching sealed identity.

## Analysis and next decision (fixed before new predictions)

Primary descriptive comparison: paired protected_snapshot8 minus text_only accuracy
on the complete cohort (same predictions by design across delays). Report the raw
paired win/loss counts, all denominators and an exploratory source bootstrap95% CI
(4000 draws,seed7). A practically useful diagnostic gap is predefined as at least10pp,
with its exploratory CI lower bound>0. This is a decision heuristic, not a publication
significance or architecture promotion gate. Failure does not mean vision is impossible;
it means this frozen reader/input/task combination has not demonstrated the needed gap.

Secondary descriptive comparisons: each ordinary arm versus text and protected at300s;
same-arm300s minus0s; all5 delay curves. Preserve all rows and fp32/bf16 sensitivity.
Report Action/Object and annotation single/multiple separately as descriptive strata;
never redefine the primary cohort based on a better stratum. Source-level uncertainty
is exploratory, single model/training seed and one memory seed; shared creators/content
may correlate even distinct videos. Do not call these multiple training-seed results.

If the original-cutoff visual reference has useful advantage and ordinary memory
loses it after continuation, next compare capacity-matched persistent change/event
writers and reader controls. If the reference has little advantage, investigate
readout resolution/temporal support and question dependence first. No automatic
training scale-up, no CausalPatchPool promotion and no method-novelty claim here.

## Launch

```bash
TRACE_REV=<tested-revision> TRACE_STATE_DATA_ID=<verified-ready-id> \
  bash scripts/launch_core_longstream_states_a100.sh
```

The immutable source archive, startup CPU regressions, two-source smoke, full4GPU
shards, report and archive are saved in an isolated run. Existing GPU jobs are not killed.
