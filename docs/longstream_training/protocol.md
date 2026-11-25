# Matched long-stream reader pilot

This is the original-plan track, independent of complementary-memory Issue #7.
Start from PR #6 commit4442c88 and use a separate worktree/branch.

## Fixed comparison

- Same verified Qwen3.5-4B and PlainLoRA2500 initialization for both arms.
- Freeze vision; train LM LoRA in both arms. The learned arm additionally trains a small question-conditioned selector; a standard selector is a baseline component, not a novelty claim.
- Question-blind Reservoir writing,32 historical+4recent steps. Each step is a real two-frame observation compressed to16spatial tokens. Both arms read at most8steps/128visual tokens.
- Baseline reads the latest8retained steps. Learned readout scores its own identical stored bank from question embeddings, stored visual features and relative ages. Hard top-k inference; the explicit straight-through soft surrogate supplies answer-loss gradients during training.
- No evidence interval, gold label or hidden target cutoff enters normal selection. Evidence annotations audit whether an event lies outside the recent window only.

## First data queue

Reuse the existing ActivityNet Captions before/after queue, which has512train and171development source videos before current eligibility checks. Questions and options come from ordered human-caption events in the same natural source video. Stream from0 to the original question time; no unrelated-video concatenation or artificial continuation.

The original40K mixture consists of30K YouTube QA from0–60second pools plus9997NExT temporal QA. Its size is not evidence of long-stream supervision. The new preparation keeps source folds, requires prefix>=60seconds and labelled event end at least9seconds before the question, excludes known prior-QA source aliases, verifies media bytes, and stages on `/data`. Known byte overlap with prior training or across folds stops preparation. No source is selected by model correctness.

This reuses prior ActivityNet development data and is not official or unseen final evaluation. Caption-derived questions can have linguistic shortcuts; text/recent controls and all outcomes must be retained. Training preparation must report actual counts and exclusions before any efficacy claim.

## Resources

User explicitly permits co-running with existing A100 jobs when memory is sufficient. First use GPUs0–1, microbatch1 and bounded activations, with an actual free-memory check and measured peak usage. Do not stop existing jobs. GPUs2–3 are the initial coordination proposal for the separate Issue #7 track; inspect occupancy and coordinate before launch.

Begin with real-data gradient/save/reload smoke; then a fixed small pilot with a matched sample stream and checkpoint schedule. Report QA losses, query-selector/LM gradients, selected frames, actual tokens, elapsed time, and peak memory. Do not treat smoke accuracy or implementation tests as a method gain. Official OVO/StreamingBench adapters and clearly labelled original4B/Plain development baselines are subsequent work during training preparation; final untouched-source evaluation follows model/configuration freeze.
