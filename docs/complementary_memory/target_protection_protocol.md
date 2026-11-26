# Phase 1c: privileged target-protected32, v1

Freeze source, protocol, input identities and runtime before any predictions.
This is evaluation only on the reused Test A/B0 77-source development cohort:
385 questions repeat 77 anchored facts at 0/15/60/180/300 s. No training, reserved
data, official benchmark claim or new question-blind writer is involved.

## Policy and exact candidate law

Keep three disjoint owner sets: newest 8, protected departed target steps P, and
uniformly sampled other departed steps R. Protect the eight saved snapshot indices
at the causal target event, after checking that they match the current recent set.
Snapshot length, uniqueness, timing and absolute prefix coordinates are hard checks;
do not infer validity from cutoff seconds alone. No gold/question content selects P.
The timing itself is privileged and must be named as such.

Until the first protected departure, call the original NativeStream Reservoir
append method unchanged. Reuse Test A memory_seed(group, 17/29/43). Once a protected
step leaves recent, admit it only to P and reduce R capacity to 32 - |P|. If R exceeds
that capacity, remove a uniformly random R member. A protected arrival never
increments the R candidate counter or performs an ordinary Reservoir insertion
draw. Other departed steps use the usual Reservoir law over their own seen count.
Uniform thinning preserves a uniform sample of the other departed candidates.
Before R fills, preserve all eligible entries up to its current capacity.

P contains no duplicate of recent; it eventually contains all eight target steps,
with R capacity 24. Steady visual steps equal min(steps_seen, 40); the append-before-
eviction transient is at most 41. Recurrent/conv bytes stay bounded. Each visual
step has 16 pooled tokens. Report actual KV tokens/bytes, which also include
timestamp/delimiter tokens and are not an exact cross-policy byte match.

## Frozen references, reproduction and forks

Use the unchanged original media, measured metadata/PTS schedule, pixel budget,
4x4 pooling, native head/tail and Plain LoRA2500 from completed Test A. Seal its
full report/input, all 385 record hashes, original helper hashes, model/checkpoint
identities and Torch/Transformers versions. Likewise seal completed B0's input,
report and all question-record hashes for the newest-40 comparison. No old source
or artifact is edited. No kernel/package/decode change is made during this run.

At 0 s, every source and all three protected seeds must reproduce their saved
reservoir{seed}:S1K1 option logits within maximum gap 0.25; compare recent/history
query-time indices, KV tokens/bytes, recurrent bytes and native next position too.
Protected admissions must still be zero. Report actual gaps/quantiles, choice
changes and full-report mean/max-absolute gold-NLL drift per seed. These are saved
query-time indices, not old per-append trajectory proofs.

Rerun window8 only on the same two fixed smoke sources, at all five queries each.
This reduced later-delay reproduction coverage is explicit. No inference of zero
drift is made from B0's earlier run. All three zero-delay checks strengthen reference
provenance up to that event; they do not bound drift afterwards. Old per-question
packing digests were absent; later unreproduced-reference drift remains unbounded.

Every query scores disposable forks; all three protected re-answers must be exact,
and cache/recurrent state, logical ownership and RNG snapshots must stay unchanged.
Keep original eight-step native parity at max logit gap 0.5 with exact packing;
short parity is not a long-stream numerical error bound. New records retain actual
embedding/position prefix-chain and head/tail digests, without retrospective claims
about Test A. Verify target coverage and causal reference indices at every query.

## Primary and interpretation

Primary: **300 s gold option-normalized NLL, protected minus saved Reservoir**.
Compute each seed's metric, average seeds 17/29/43 within source, then pair/bootstrap
all 77 sources (10,000 draws, seed 7). Negative favors privileged protection. A
wholly negative 95% interval is evidence of benefit; a zero-crossing interval is
inconclusive, not no headroom. Report paired SE/MDE80 and accuracy with intervals.

Freeze **0.05 nats** as the relevant-improvement magnitude for an additional interval
reading: only when the primary lower bound is greater than -0.05 is an improvement
of at least 0.05 excluded by that interval on this cohort. This is separate from
the primary sign reading and not an automatic training or global rejection rule.

Co-report 15/60/180 s, and protected-minus-B0-window40 at every delay. Zero delay is
integrity first; report its descriptive scores only with the complete result. All
secondary intervals are pointwise. Fifteen seconds stays descriptive: there is no
noninferiority margin/claim. Never replace the primary with a better horizon,
seed, source subset, checkpoint, or the 0.05 reading after observing results.

Positive headroom shows value of these eight privileged steps within a 40-step
budget. It does not show that a question-blind writer can discover them, that
single-step answer-loss labels transfer between questions, or that recurrent-state
conditioning is useful. Policies evolve their own states; this is not a fixed-state
causal effect. Full KV is unbounded and is not equal-budget selection headroom.

## Validation, smoke, cost and full execution

Use an isolated source archive/run directory, hosted CI and A100 CPU tests/lint.
No tests, lint or model execution on the authoring PC. Unit checks include exact
pre-admission path/seed parity, disjoint 40/41 ownership, early-prefix fill, target
validation, fork isolation and exhaustive finite RNG paths for uniform thinning.

Live-check GPU availability (>24 GiB free per selected GPU). Prefer all four A100s
0/1/2/3 if idle. Smoke uses the original two fixed sources on two GPUs, three live
protected streams plus window8. Smoke exposes integrity/cost/memory only, not
partial accuracy, NLL summaries or question logits in its exported bundle.

Review measured smoke time before full launch. Full non-smoke sources use three
streams; smoke has four. Estimate with the exact remaining source/shard schedules,
separate vision/packing/query costs and overhead uncertainty. Four-way full sharding
does not promise a linear speedup. B0's 1.28 wall/component factor is descriptive,
not a guarantee. Preserve the smoke bundle before the full report refreshes it.
The owner explicitly authorized proceeding after validation and cost review.

Full reporting requires all **77 sources / 385 questions**, **1,155 protected option
vectors**, **10 auxiliary window8 vectors**, and **231 zero-delay seed reproductions**
plus ten smoke-window checks. Do not peek at partial outcomes. Resume only the same
sealed code/input; never loosen a failed gate. Final publication requires independent
archive/hash/record/bootstrap checks. Keep reused-development/source-dependence,
explicit-cutoff, short-clip-trained-reader and cross-run drift limits with the result.
