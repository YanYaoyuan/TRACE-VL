# External transfer / core temporal pilot: local validation, 2026-11-14

This validates engineering, not external accuracy. A100 OVO/NExT-QA evaluation has not been run by the tester.

Full repository regression: **378 passed, 1 skipped**; core QA subset: **95 passed**. Ruff and launcher `bash -n` passed. The first full-suite attempt was interrupted after a sandbox multiprocessing wait; the bounded host rerun completed.

## CPU / data checks

New regression coverage includes:

- Nonzero temporal weights cannot read future features; prefix outputs agree with the corresponding full-prefix outputs.
- Initial temporal output exactly preserves plain grid embeddings.
- Real tiny Qwen QA answer CE updates the temporal output projection on step one and inner attention on step two; reload reproduces embeddings exactly.
- MC first-token choice is independent of the gold answer and is still recorded when free generation emits EOS. Format validity and choice accuracy are separate.
- Both external primary endpoints and both no-harm checks must pass; otherwise the queue ends before any continuation training.
- OVO frozen-manifest hashing and diagnostic-half membership; NExT-QA selection by video before outcomes.
- Source-alias and renamed exact-byte training overlap rejection; future-frame and observation mismatch rejection.
- Real FFmpeg/FFprobe external-media preparation, deterministic manifests and resume; an edited population refuses reuse.
- A continuation checkpoint with different observation settings is refused.

## Real 4B / dual RTX 5090 check

Model: locally cached original Qwen3.5-4B. Parent: `qa-ddp-reference-20261113-002/checkpoint-000004` (an earlier engineering-only grid LoRA checkpoint).

- Two updates of plain-grid continuation and two updates with the causal temporal adapter completed on two GPUs. Same seed 17, rank 8, LR 2e−5, one example/rank/update, 8 sampled frames locally. A100 uses its declared 16-frame population and four GPUs.
- Temporal QA gradient norms after clipping: 0.886625 and 0.979514. Peak allocated memory on rank 0 approximately 10.2 GiB.
- A separate temporal run stopped after its first checkpoint and resumed. All saved LoRA **and temporal** parameters exactly matched the uninterrupted two-update run.
- Both checkpoints were reloaded through the evaluator and completed two real-video questions with first-token MC scoring and free generation. Synthetic distractors were used only to exercise the MC path; these scores are not a quality result.
- Saved states, optimizer/RNG records and completeness checks use the existing integrity-checked checkpoint implementation. Temporal weights live in `training.pt`; PEFT adapter files alone are insufficient for the temporal arm.

Large artifacts are stored under the user-selected disk:
`/media/user/50D2BD30D2BD1AE4/WorkSpace/TRACE-VL/core-vlm-runs/`.

The full A100 data population, long run throughput, transfer gain and T0 benefit remain to be measured. The queue's external gate makes that uncertainty explicit instead of treating local smoke success as evidence of improved understanding.
