# Fresh-base mixed temporal QA pilot (v1)

User authorized this new experiment on 2026-11-15. The failed core-transfer gate remains unchanged and its old continuation remains unexecuted. This is a different, predeclared recipe, not a retry with a relaxed old gate.

## Question and scope

Can ordinary causal temporal processing in the **visual-to-language embedding path** improve video temporal QA over ordinary LoRA trained on exactly the same mixture, without sacrificing the original model's general abilities? This trains the core video model, not an alert classifier or frozen-feature sidecar. The ordinary two-layer temporal adapter is a **T0 baseline**, not a novel architecture claim.

- Original Qwen3.5-4B weights; frozen vision encoder; LM LoRA rank 8, alpha 16, dropout 0.
- Both arms see 16 identical sampled frames, native position/timestamp layout, and 4x5 spatial pooling per temporal unit.
- `plain`: ordinary grid pooling + LM LoRA.
- `temporal`: the same pooling + two causal Transformer layers (width 128, 4 heads) operating independently along time per spatial cell + LM LoRA. Absolute inter-frame time differences are sinusoidally encoded. Zero-initialized output projection makes the initial residual exactly zero. QA loss updates the temporal module and LoRA.
- Spatial cells are fixed, not tracked objects. Within-pair vision encoding is unchanged. Causal masking prevents future **temporal units** from entering earlier units; this is not an event-causality claim or full streaming implementation.
- Same seed 17, AdamW (no weight decay), cosine schedule (10% LR floor), LR 2e-5, 1,000 updates, 4 GPUs x accumulation 4 = 16 questions/update. Same 16,000 question presentations and answer-token budgets. Added temporal parameters/FLOPs are reported; total compute is NOT matched.
- Separate two-update smoke runs, discarded before fresh full training; fixed final checkpoint only. No choosing a checkpoint from external scores.

## Training and exclusions

Use 2,000 temporal questions (TN/TC/TP) from checksum-pinned **official NExT-QA train.csv**, and 2,000 MC questions from the already prepared LLaVA YouTube training split. Select by seeded hash and round-robin video before media availability is checked, never by accuracy. Require at least 300 distinct byte-hash videos per family. No changing seed or replacing missing media to pass the gate. A 1:1 manifest mixture need not be exactly 1:1 in each optimizer update; realized sample sequences are identical across arms and logged.

Exclude all official NExT validation IDs, both OVO halves' video IDs, StreamingBench/Video-MME source IDs, and the existing general validation videos. Verify selected media by full decode; reuse verified byte-identical caches. Check exact-byte overlap with external development media and general validation. Aliases/exact bytes do not establish absence of perceptual duplicates or base-model pretraining contamination. Official NExT training means its validation result is **in-dataset generalization**, not zero-shot transfer; OVO is cross-dataset development.

Reuse the **unchanged** `core-transfer-v1-f1c19a7` external manifests (674 OVO questions after fixed causal-frame exclusions; 1,744 NExT questions from 200 videos). No final test evaluation or training. This development population has already informed research choices; eventual publication needs fresh held-out evaluation after freezing the method.

## Measurements and decision fixed before training

Reevaluate original-grid and both trained arms under identical recorded pixels, frame indices, prompts and time bounds. Primary scoring: first-answer-token option logits. Report strict free generation and format legality separately. Video-cluster paired bootstrap, 4,000 draws, seed 7; per-task comparisons are exploratory/unadjusted.

The new pilot's continuation gate requires **all**:
1. Temporal arm exceeds both plain and original-grid by at least 2 pp on NExT TN/TC/TP, with the 95% interval lower bound above 0.
2. Against both references, interval lower bounds exceed -2 pp on OVO backward, OVO realtime and NExT descriptive questions.

Also report plain minus original to separate data/LoRA effects from architecture effects. Failure may mean inconclusive precision; do not automatically call it degradation. No automatic scaling regardless of outcome. Positive results require seeds and order/history ablations, an appropriate parameter-capacity control, and a fresh held-out benchmark before making a temporal-mechanism or novelty claim.

## Execution

`launch_core_mixed_temporal_a100.sh` uses a commit-pinned source snapshot, an immutable configuration and independent output directory. It runs CPU tests, downloads only pinned official train/val CSVs if absent, checks/reuses/extracts data, runs two-update DDP training and checkpoint evaluation for both arms, verifies identical initial LoRA and realized sample/LR/token schedules, trains both full arms sequentially, evaluates original plus both trained arms on four GPU shards, and packages reports.

Set `TRACE_PREPARE_ONLY=1` or `TRACE_SMOKE_ONLY=1` for a narrower run; default completes the whole queue. `TRACE_GENERAL_DATA` and `TRACE_EXTERNAL_DATA` override existing prepared-data locations. No model or full video download is triggered. Interrupted jobs resume only under matching identities/checkpoint integrity. Logs/progress are under `TRACE_OUT/logs`; per-update metrics include QA loss, gradient norms, temporal gradients, wall time, peak GPU allocation and actual question IDs.

Deliver `reports.tar.gz`: REPORT.md, report.json, training-audit.json, smoke-ok.json, config/provenance, training metrics, predictions, data manifests/inventory and logs. Training curves can be reconstructed from metrics.jsonl. Weights/media stay on the A100.
