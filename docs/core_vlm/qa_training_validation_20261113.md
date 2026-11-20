# Core video QA: distributed training validation (2026-11-13)

The core-model training path now updates Qwen3.5-4B language-model LoRA from video-question answer loss. It is separate from the historical frozen-VLM alert-head experiments. This report establishes engineering functionality, not a novel architecture or an accuracy improvement.

## Implemented contract

- `python -m trace_vl.core_vlm.train`: native Qwen prompt/position packing, frozen visual encoder, full or fixed-grid visual tokens, answer-token cross-entropy including EOS, and LM LoRA. The objective averages per-question losses; questions with longer answers do not automatically receive greater weight.
- `torchrun` uses NCCL DDP, one unpadded question per microbatch, configurable gradient accumulation, and a deterministic global question schedule. Full/grid controls can share sample order and update count while their actual token and compute budgets differ.
- Committed checkpoints contain adapter weights, optimizer state, every rank's CPU/CUDA RNG state, update count and metrics. Cross-rank adapter digests must agree. Files are verified before resume; incomplete temporary saves and conflicting identities cannot masquerade as completed training.
- `python -m trace_vl.core_vlm.evaluate`: held-out val/dev question generation with the original or trained model, independent shards, atomic per-question results, and complete-coverage checking. Report assembly recomputes correctness from generated answers instead of trusting stored correctness flags. It does not open final benchmark splits automatically.

## Real local verification

Two RTX 5090s, original Qwen3.5-4B, bf16, grid 4x5, eight actual sampled frames per question, four real training questions, four optimizer updates, one question per GPU per update. LM LoRA rank 8 has **6,193,152 trainable parameters in 208 tensors**. Learning rate starts at 1e-4 and follows the same four-update cosine schedule in both runs. Vision remains frozen. Timestamp handling was explicitly `native-index-fps`; this retains the documented native index/fps approximation, with actual frame selection checked against PTS.

Two independent runs were compared: uninterrupted four updates, and two updates followed by process restart and the remaining two updates. Final adapter tensors, optimizer state, both ranks' RNG state, per-update loss, gradient norm, learning rate and sample schedule all matched exactly. Peak allocated memory reported on rank 0 was approximately **10.20 GiB** in the interrupted/restarted run. This is not a claim about long-video training memory.

The initial attempt without deterministic algorithms had divergent gradients despite equal first-step loss; those attempts were retained and are not the evidence for exact recovery. The passing comparison explicitly enables deterministic algorithms and the cuBLAS workspace setting.

Restarting a completed run verified the final checkpoint and skipped training. The saved adapter was then independently reloaded for two-shard evaluation of eight held-out questions from eight videos. Both shards and report assembly completed. Normalized text exact match was **1/8**; this tiny, four-update smoke result is not an estimate of generalization or semantic QA accuracy. It provides no evidence of a training gain.

## Automated validation and limits

61 selected core-input, QA, sampling, checkpoint, evaluation and atomic-I/O tests passed. These include native-forward/answer-loss agreement, causal frame checks, gradient flow, deterministic sampling, damaged-checkpoint rejection, four-process manifest publication and missing/duplicate prediction rejection. Ruff passed for the new trainer, evaluator and tests. The entire repository suite was not run in this validation.

Detailed checkpoints, raw predictions and executed-run identities remain local. This commit contains code and aggregate results only.

Still required: formal training-media preparation and benchmark-source exclusion, longer same-data full/grid QA baselines, a distinct temporal architecture and its ordinary-temporal baseline, the complete A100 experiment queue, multiple seeds, and protocol-compatible Mage-VL/MOSS-VL evaluation. The current trainer accepts an existing normalized manifest; it is not the requested full data-preparation pipeline by itself. No additional alert-head training is proposed.
