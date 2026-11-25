# B0 phase 1a: carried window40 versus Reservoir, v1

Freeze this protocol, code, reference hashes and runtime versions before scoring.
This is evaluation only on the already-used 77-source Test A development boundary.
No writer/reader training, new labels, reserved sources or official benchmark claim.
The earlier Test A result and source archives remain read-only.

## Missing control and primary

Test A compared recent8+Reservoir32 with window8. This phase supplies a **40 versus
40 retained visual-step** comparison. It tests a policy, not a fixed-state causal
intervention: each live policy carries its own recurrent/conv state, evolving under
its own full-attention KV. It does not establish a state-conditional writer.

Primary: at 300 s, mean over three Test A Reservoir seeds (17/29/43) minus the new
window40 **gold option-normalized NLL**, averaged within source before source-paired
bootstrap (10,000 draws, seed 7). All 77 sources contribute. A wholly negative 95%
interval favors this Reservoir policy; a wholly positive one favors window40; an
interval crossing zero is inconclusive, not equivalence. Report paired SE/MDE80
(2.8 × SE) and accuracy with its interval. No automatic training decision follows.

Secondary: all five delays (0/15/60/180/300 s); window40 minus the freshly reproduced
window8; Reservoir minus that window8; and fresh-versus-saved window8 drift. Report
0 s window40-minus-window8 integrity separately; lower NLL interval bound above
+0.10 flags demonstrated damage but not meeting that trigger is not noninferiority.
No selection of better delays, sources, seeds or checkpoints may replace the primary.

## Fixed streams and computation

Same original media, hash-checked metadata, full PTS schedule, resolution, two-frame
steps, 4x4 visual pooling (16 tokens per step), frozen Plain LoRA2500, native prompt,
and question events as Test A. Decode/encode sequentially through each source's
last query time. New steps are question-blind. No new query-time vision encoding.

- **window40:** original `NativeStream(policy="window", recent=40)`, newest 40 steps.
- **window8:** original class with recent=8, rerun on every source for reproduction.
- **auxiliary Reservoir17:** original class and source seed on the two fixed smoke
  sources only; checks saved `reservoir17:S1K1`, not an extra analysis replicate.
- **Reference Reservoir:** reuse all three separate S1K1 seed logits from the completed
  Test A run at `fa8b205`. No averaging logits; average NLL/accuracy within source.

Original helper files must exactly match Test A code hashes and torch/Transformers
versions. Freeze the full completed reference report hash, input identity, every
question-record hash, model and checkpoint identities. Check media/metadata hashes,
causal schedule, query prefix length, recent/snapshot indices and absolute native
coordinates. Never infer old per-question packing proof: it was **not recorded**.

Reproduce saved window8 at every query of every source, and seed17 at all smoke
queries. Maximum absolute option-logit gap <= **0.25**, with changed choices and
actual NLL drift reported. This gate is not an NLL error bound. Never loosen a
failure. Reused Reservoir seeds not rerun still have unbounded cross-run drift;
the all-source window8 check is empirical provenance evidence, not proof for them.

Each query scores disposable forks. Snapshot/verify every live cache and recurrent
state around all queries; re-answer window8 deterministically. The original native
eight-step parity check must pass its unchanged **0.5** gap gate with exact packing.
This short check does not bound long-stream numerical drift.

New records store head/tail ID digests, actual tail embedding/position digests and a
rolling packing-prefix digest: initialize SHA-256 with the bytes of the head
embedding/position tensor digest, then update with each observed step embedding/
position tensor digest. Its counter is copied before append, leaving the live
counter untouched. These are new B0 proofs, not retroactive Test A proofs.

Window40 and Reservoir both retain 40 **visual steps / 640 pooled visual tokens** at
the queries, but timestamp/delimiter tokens can differ with step age. Report actual
KV token counts, KV bytes and recurrent bytes by policy/delay; do not claim an exact
total-byte match. Window8 retains 8 steps. The append-one/evict-one path transiently
holds at most 41/9 steps respectively. Report logical pre-eviction maxima and steady
maxima separately from measured peak CUDA memory. Recurrent bytes must stay bounded.

At each delay report how many sources retain the original target's last snapshot
step in window40, and how many retain any of the eight snapshot steps. Also report
the three-seed mean target-last-step retention rate of the reference Reservoir.
These are timestamp/index audits, not semantic evidence certification or a source
filter. Early prefixes may hold fewer than 40 steps; report actual counts.

## Execution and reporting

Isolated branch, immutable source archive and new run directory. GPUs **2–3**, after
a live availability check (>24 GiB free each). Never stop another job or change the
frozen runtime packages. Authoring PC runs no tests/lint/model code; hosted CI and
A100 startup tests validate implementation before inference.

Smoke: Test A's same two fixed sources, all five queries each, both new windows and
auxiliary seed17. Gate only integrity, finite logits, reproduction/parity, state/KV
budgets and measured cost. Expose no accuracy/NLL summary or question logits in the
smoke bundle. Preserve it before the full launcher refreshes the bundle. Estimate
full wall time from measured vision/question/audit time plus two live LM streams,
not blindly extrapolate the smoke's three streams. Report estimate uncertainty.

Full report requires every source and query: **77 / 385**, 770 new window prediction
vectors plus ten auxiliary reproduction vectors on smoke sources. Never filter on
outcomes or target-step retention. Resume only identical sealed inputs/code. Store
individual per-question reproduction gaps/choice changes, all actual budget audits,
per-source timings and CUDA peak. Report max and quantiles of reproduction gaps.
Independently verify archive hashes and source-paired math before publishing results.

These are anchored, reused development questions; five delays repeat the same fact.
Recent KV can encode earlier history, different videos may share creators, and the
reader was not trained on continuous long streams. State all limitations with the
result. Coverage, privileged target protection, capacity64/128 and cross-question
utility are separate future phases and are not silently added here.
