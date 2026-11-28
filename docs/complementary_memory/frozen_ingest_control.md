# Frozen ingestion / adapted query: a controlled hypothesis test

The completed1024-source continuation did not improve the fixed300s Reservoir primary. Saved-record diagnosis found identical retained frame indices/budgets across checkpoints, but different predictions. One hypothesis is that answer-tail updates to shared LM LoRA change future source ingestion despite detached-prefix training. This is not a diagnosed bug or established explanation.

## Deliberate two-stage model

Use two independently allocated Backbones with the same frozen base/vision/tokenizer/cache configuration. Both start from the preserved64-source Reservoir adapter. One is entirely frozen and builds question-blind source caches; the other learns LM LoRA on question/answer forks. This is a trained two-stage architecture. Do not substitute an adapter into an arbitrary old cache and call it the original system. No runtime adapter switching is used.

`EndpointReaderEpisode(..., prefix_bb=frozen)` checks frozen/eval ingestion, matching cache geometry/device/dtype/tokens, independent parameter storage, parameter versions and configuration. Source appends use the frozen backbone. Answer CE and optimizer/all-reduce own only query LoRA. Prefix parameters receive no gradients. Query updates close episodes; subsequent prefixes are rebuilt and must remain bit-identical for identical inputs.

The default shared-adapter path is retained. Neither path backpropagates through source ingestion. Vision features, prefix observations/positions, Reservoir8+32, memory seed17 and answer supervision remain unchanged. The frozen architecture uses two model allocations; report its measured memory overhead rather than claiming equal memory.

## Matched128-source diagnostic

Run shared and frozen-ingestion arms from the same64-source adapter **and Adam64 moments**, learning rate1e-5/no decay/global clip1. Four source ranks average source gradients before clipping. `--stage-updates32` takes exactly the first32 global batches of the unchanged full1024-source schedule:128 new sources per arm. Both arms finish at Adam96. Same source/question order, native endpoint QA,2fps observation, answer/EOS budget and frozen initialization; no development QA is used for training.

First validate real4B mechanics on a bounded existing pilot source: equal-initial-weight shared/separate cache and CE agreement, query-only gradients/update, frozen parameters and bit-identical rebuilt prefix, and a query checkpoint with immutable ingestion receipt. Next verify fresh four-rank checkpoint resume and that both arms' first global update produces the same query parameter digest before they diverge. Preserve first-update checkpoints. Original shared stage16 digest is an additional continuity diagnostic, not a quality selector.

All source/metadata hashes, initialization identities and run/code/config fingerprints are recorded. Do not overwrite or resume the completed old runs with this changed runtime. New experiment directories and isolated branch only. CPU tests/lint run on hosted CI/A100, not the authoring PC. No kernel/package/model changes.

## Correct evaluation and decision

Evaluate each model under its actual ingestion/query architecture, on all frozen77 development sources /385 questions with all four controls/five delays. The frozen architecture must restore its immutable ingestion adapter as well as query weights. The old single-Backbone evaluator is not a valid evaluator for this model.

`staged_prefix_parity` checks exact native embedding/position packing, then compares a native-packed ingestion prefix plus query-model question against chunked frozen ingestion plus the same query model. This is **staged native prefix/query parity**, not ordinary single-model full-sequence parity. Same-weight equivalence and diverged-query tests precede real-model inference. The existing `source_observations` query-native CE check remains a packing check for the query model alone and cannot prove hybrid-forward equivalence after adaptation.

The intended primary is300s same-Reservoir option-normalized NLL, frozen minus matched shared,95% exact-byte-cluster bootstrap. Accuracy, other delays/read controls, text-adjusted differences and near-time perception proxies are co-reported/exploratory. No partial quality gates or outcome-selected checkpoints/sources. This is a small mechanism pilot, single seed on reused development data; a positive result would need further controlled training and official/unseen evaluation, and would not establish novelty by itself. A null result would not identify supervision as the cause.

## Verified launch on2026-11-28

Training runtime `29cd6af` passed two hosted CI runs (635 passed,5 skipped), followed by31 relevant A100 tests and full src/tests/scripts Ruff. Evaluation runtime `3388f63` passed two hosted CI runs (647 passed,5 skipped), followed by43 relevant A100 tests/Ruff and230 saved-prediction reducer regression checks with zero numeric difference.

The real4B mechanics run used a complete361.836533s /362-step old training source and its5 native QAs. Independent same-weight models produced bit-identical initial caches and exactly equal answer CE. After one question-adapter update, the frozen prefix parameters and rebuilt cache were bit-identical, query weights changed, Adam advanced64→65, and a query-only checkpoint was exported and verified. Peak allocated memory was18.3344GiB, elapsed204.31s. This is mechanics, not a quality result; the old source was already part of pilot training.

The matched four-rank arms then produced the same first-update query parameter digest, `252d4b4082a55e7d1d98b879944f667f86316a9aa5ab0a4bc533899f252bf2f4`, with identical source/QA schedules. Both first updates consumed four new sources. The shared arm has resumed in a fresh process group toward32 updates; frozen continuation follows automatically. Root: `/data/yyy/tracevl-runs/frozen-ingest-29cd6af-003`.

The initial pre-GPU run timed out waiting for a CI receipt while terminal submission was unavailable; a subsequent smoke reached its numerical/cache assertions but failed because the launcher omitted the checkpoint parent directory. Both failed directories are preserved. The active run creates that directory before launch and uses the same validated model/controller/worker code; no package, kernel or scientific recipe changed.

Post-training evaluation is queued at `/data/yyy/tracevl-runs/frozen-ingest-pair-eval-3388f63-001`. Its CPU checks and reducer regression are complete; it waits for both training arms, verifies their actual32-update query checkpoints and immutable64-source ingestion checkpoint, then evaluates all77 sources /385 questions per arm with four controls. One worker/device,26GiB free-memory admission for the dual architecture, bounded retries/deadlines and unchanged old baseline artifacts. No paired efficacy result is available yet. The [launch verification receipt](results/frozen_ingest_launch_verification_20261128.json) records the mechanics and initial bridge.
