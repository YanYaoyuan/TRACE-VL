# 1024-source Reservoir continuation: training complete

The fixed continuation completed successfully at **2026-11-28 08:02:34 Asia/Shanghai**. All1024 new FineVideo training sources /5082 native endpoint QAs were consumed exactly once, in256 synchronous four-source updates. Starting from the preserved64-source Reservoir candidate and its Adam moments, the final Adam step is320. Across both training stages there are1088 metadata groups and5399 QAs. Uploader-field groups do not certify independent video origins.

Training workers took **7h39m36s**, including the two new-data cost updates and the resumed254-update phase. The controller took8h10m31s, including checks and waiting for qualified data. Peak allocated CUDA memory was **11.2809GiB/rank**. All workers exited0; four-rank update and fresh-process-group checkpoint recovery passed before the main stage.

## Verified completion

- Runtime revision: `89e1f3079696434d579f6ec35b68049e56248cb4`; both hosted CI runs passed, followed by24 relevant A100 CPU tests and full src/tests Ruff.
- Run ID: `4ef952337a9ca871b9aa73a7d762ae9df399e3e4fa569cc1ec204c922e3b190c`.
- Final checkpoint: `/data/yyy/tracevl-runs/reader-scale1024-89e1f30-001/train/checkpoint-000256`.
- Final adapter tensor digest: `e14d99f76f2ae6a1e1e6c844fa4b49a58b44a182965375ce2d71e793d59daea8`.
- Final adapter file SHA256: `94b59b329b65aa02f9f1bdd828c471fdedbe9d393c0d94c034f4cc4dfbd9c518`.
- Manifest SHA256: `4429de41c80f3f6579b9af34983db03300d73aab9395d3d1107473355027cddc`.
- Training report SHA256: `791b9b69b804f1f0ecae450050da2f4dbb6a8ec85a761da5336d39c35bb0d413`.

The CPU integrity audit reread every retained checkpoint file and all256 update records. It verified all1024 unique source IDs against the frozen schedule,5082 QA counts, finite losses and positive finite gradient norms, detached prefixes,40 retained KV steps, four saved rank RNG states and final Adam step320. All208 serialized adapter tensors exactly match the optimizer checkpoint after explicit PEFT key-prefix/adapter-name normalization. Final weights differ from initialization. Retained updates are16,64,128,255,256. The [machine-readable receipt](results/reader_scale1024_training_integrity_20261128.json) records the file hashes and checks.

Data preparation qualified1269 of1280 fixed candidates, rejected11, and selected the first1024 passing the declared checks. It verified179 pinned shards /89,742,137,736 bytes, materialized29,280,546,160 selected video bytes on `/data`, validated full PTS and3072 sampled decode checks, and independently reread media/metadata bytes. Actual selected durations are360.07181–659.47s. No dev/reserved/inspection QA was used. An additional exact-byte check found zero overlap with the77-source development cohort; source-family/reencoding independence remains unproven.

## Quality evaluation is running

Training completion is not a quality score. The final checkpoint is now being evaluated on the same77 sources /385 questions as the audited pilot, using the unchanged inference worker and evaluator from `b0a2fad2af8937388136b18543862556a26f91ce`. The four policies remain Reservoir17, window40, recent replay8 and text-only. All frozen source observations, query endpoints, native parity, fork isolation and answer scoring are preserved.

Evaluation root: `/data/yyy/tracevl-runs/reader-scale1024-eval-89e1f30-001`; input ID `197ca8fbcb54d5cb41651a3071a13fb47b7b304df2a0c7139b024bd7db9e9238`. Four A100 workers own19/20/19/19 sources. Pilot/Plain predictions retain their original input IDs and are reused with SHA256 receipts; they are not relabelled as new executions. A10012 evaluator/control tests and Ruff passed. The report reducer reproduced the known pilot scores and verified zero differences/intervals when comparing identical complete predictions before GPU scoring.

The declared primary comparison is300s same-Reservoir option-normalized NLL, expanded minus pilot, with a95% exact-byte-cluster bootstrap interval. Accuracy, other policies/delays and text-adjusted changes are secondary/exploratory. No partial accuracy or NLL is used to select checkpoints, sources or policies. This is reused development data, one training/memory seed, and changes both training amount and global source batch1→4. It is not an official/unseen benchmark result, a pure data-size ablation or a demonstrated memory-specific improvement. No quality result is available yet.
