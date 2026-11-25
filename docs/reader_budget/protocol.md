# Nested cached-reader budget diagnostic, v1

Freeze this protocol and all input/code hashes before scoring. Evaluation only:
no training, new video decoding, writer change, reserved sources, or official
benchmark claim. This is a follow-up chosen after earlier development results.

## Question and fixed inputs

Does reading 16 rather than 8 retained steps improve answer NLL through visual
content on the starting Plain LoRA, under two question-blind read policies?

Use exactly the previously used 115 ActivityNet development sources, one question
per source (81 before, 34 after), manifest SHA-256
`a0a772caeb1f4876e65f76bcf9bd4e599a7fb8461d0885fdb89c8058f2cf150e`.
Retain all sources. Use the original Plain LoRA checkpoint at update 2500 and
the same hash-verified cached 32-history + 4-recent bank. The bank itself is not
expanded. The previous complete controls run at `76b1162` supplies frozen model,
adapter, feature and blank-token identities, plus reproduction reference logits.
No checkpoint selection or outcome-based question exclusion is allowed.

## Reads: 8 / 16 / 36 steps = 128 / 256 / 576 visual tokens

Every read contains the same newest four bank steps. At 8/16 steps, select 4/12
of the same 32 retained history entries. At 36, read all bank entries once.
Sort chosen indices chronologically before native packing. Steps contain the
original two frames/timestamps; do not shift the question time or positions.

- **Nested temporal spread:** start with the oldest history entry, then repeatedly
  select the remaining entry with greatest minimum end-time distance to selected
  entries. Break ties by the oldest bank index. Use prefixes of lengths 4 and 12
  of this one 32-entry ordering.
- **Nested random:** generate one permutation of the 32 history indices for each
  source/seed, using NumPy default_rng seeded by the first eight hex digits of
  digest(["reader-budget-v1", source_group, seed]). Use its prefixes of lengths
  4 and 12. Seeds are 17, 29, 43, 61, 83. Source identity, not question identity,
  determines the seed. Random36 and spread36 coincide and are scored once as
  `all36`, rather than duplicated as independent observations.

This is cached readout of an already retained bounded bank, not an online memory
writer experiment. Neither selection policy sees question/option text, answer,
evidence annotations, model scores, or frames outside that bank. Reject invalid,
unordered, nonfinite or future bank timestamps.

Each read has its own blank twin: replace values with the same hash-verified gray
vision token while retaining exact selected steps, timestamps, visual token count,
native text, and positions. The reference blank has exactly zero temporal-token
variation across its original eight-step clip; verify that property and its hash.
Also score text-only as a descriptive reference and reproduction anchor. Blanks
are unusual input: content contrasts do not establish an unrestricted causal
effect of vision on a deployment distribution.

## Endpoints and inference

Gold NLL is normalized over the two option-letter logits. Average NLL and accuracy
over five random read seeds **within each source**, never logits or independent
question/seed rows. Bootstrap whole sources: 10,000 draws, seed 7, paired reads.

Content benefit at K is NLL(blank K) - NLL(real K). The two co-primary contrasts
are content benefit at 16 minus content benefit at 8, separately for temporal
spread and random. Both lower 95% interval bounds must exceed zero to report
consistent evidence for a 16-versus-8 content benefit. If only one passes, report
policy-specific evidence; if neither passes, report uncertainty and MDE80 rather
than equivalence. No automatic training or scaling decision follows.

Secondary, descriptive endpoints: real/blank arm NLL and accuracy; all36 versus
each 8/16 content benefit; content benefit of each individual policy; per-task NLL
and accuracy; and task-balanced accuracy (mean of before/after accuracy). Do not
replace the frozen source-weighted primaries with whichever subgroup performs
best. All intervals are pointwise, from reused development sources; this is not
an unseen confirmatory result. MDE80 = 2.8 times paired source SE, an approximate
planning statistic, not a post hoc success threshold.

## Execution, gates, timing and report

Two independent workers use GPUs 0-1, one sharded source queue per GPU, same frozen
starting LoRA. Inspect current GPU availability first. Never stop other jobs.
Run CPU integrity/native-packing tests on A100 before model execution. Authoring
PC executes no tests, lint or model code; hosted CI supplies additional checks.

Smoke uses the first source on each shard, fixes no outcomes and exposes no NLL or
accuracy. Reproduce old starting-LoRA text-only, real all-bank and blank all-bank
option logits on every source; maximum absolute logit difference must be <= 0.25.
Report actual gaps; this numerical gate is not a bound on all derived NLL effects.
Do not loosen a failed gate. Verify real/blank positions and visual token budgets
while packing. Resume only identical frozen inputs/code; keep completed prior runs
read-only. Review smoke integrity, peak memory and measured cost before full run.

Record synchronized preparation and LM wall time separately for every real/blank
read, complete input-token count and peak allocated CUDA memory. Preparation
includes cache transfers, placeholder packing and prefix assembly; LM time
includes the forward and option-logit projection. Text-only total time includes
its prompt preparation. Report descriptive mean times by policy; these are not
batched deployment throughput, utilization or optimized latency comparisons.

Full analysis requires all 115 complete source records and every frozen policy,
including all five random seeds and matched blanks (3,105 option-logit vectors).
Archive inputs, smoke receipt, full per-source logits/timings, logs and report;
independently verify archive SHA-256 and recompute source-paired statistics before
publishing a result. Keep completed controls and Test A artifacts immutable.
