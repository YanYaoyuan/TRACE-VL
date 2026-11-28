# 1024-source Reservoir continuation: completed development evaluation

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

## Completed development evaluation

The final checkpoint completed all 77 sources / 385 questions at **2026-11-28 16:14:49 Asia/Shanghai**, in **1h12m18s**, using the unchanged inference worker and evaluator from `b0a2fad2af8937388136b18543862556a26f91ce`. All four workers exited0 without retries. Reservoir17, window40, recent replay8 and text-only retain their frozen observations, query endpoints, native parity, fork isolation and answer scoring.

Evaluation root: `/data/yyy/tracevl-runs/reader-scale1024-eval-89e1f30-001`; input ID `197ca8fbcb54d5cb41651a3071a13fb47b7b304df2a0c7139b024bd7db9e9238`. Pilot/Plain predictions retain their original input IDs and are reused with SHA256 receipts. The 12 A100 evaluator/control tests and Ruff passed. Before GPU scoring, the reducer reproduced known pilot scores and zero self-comparison differences/intervals. Afterwards, an independent CPU reduction importing no project scorer verified 231 source receipts, 1155 question records, 4620 score vectors and 1386 file hashes. All 270 numeric checks of means, differences, and 95%/97.5% intervals matched exactly. Raw report SHA256: `fb374c2e730cdd85fdc983762d32cc07fe43a5bbe94112ea6d323df1017b636c`.

The declared primary is 300s same-Reservoir option-normalized NLL, continuation minus pilot, with a 95% exact-byte-cluster bootstrap interval (73 byte clusters). It changed from **0.684881 to 0.693161**, difference **+0.008281**, CI **[-0.021008, +0.033833]**. Lower is better. This provides no evidence of an additional gain on the primary long-memory objective; it does not establish equivalence or harm. Accuracy changed **56/77 → 55/77 (72.73% → 71.43%)**, with one gain and two losses; paired accuracy difference -1.30pp, CI [-5.48, +2.67]pp.

### Same-Reservoir accuracy across delays

| Delay | 64-source pilot | 64+1024-source continuation | Change |
| --- | --- | --- | --- |
| 0s | 58/77 (75.32%) | 61/77 (79.22%) | +3.90pp |
| 15s | 58/77 (75.32%) | 59/77 (76.62%) | +1.30pp |
| 60s | 57/77 (74.03%) | 57/77 (74.03%) | 0.00pp |
| 180s | 58/77 (75.32%) | 53/77 (68.83%) | -6.49pp |
| 300s | 56/77 (72.73%) | 55/77 (71.43%) | -1.30pp |

At 180s there are five losses and no gains; the pointwise paired accuracy CI is [-12.50, -1.32]pp. This is an exploratory regression signal across multiple tested delays, while its NLL difference (+0.001724) has an interval crossing zero. The 0s task is anchored-state recall, not a complete current-perception evaluation.

### Read controls at 300s

| Read policy | Pilot accuracy | Continuation accuracy | NLL change, continuation minus pilot [95% CI] |
| --- | --- | --- | --- |
| Reservoir17 | 72.73% | 71.43% | +0.008281 [-0.021008, +0.033833] |
| Window40 | 67.53% | 70.13% | -0.003322 [-0.025802, +0.016901] |
| Recent replay8 | 59.74% | 64.94% | -0.045250 [-0.087328, -0.007594] |
| Text-only | 57.14% | 58.44% | -0.010822 [-0.032744, +0.010325] |

Recent replay8 gains five questions and loses one (+5.19pp); its accuracy interval includes zero, although its pointwise NLL interval favors continuation. This secondary benefit is worth retaining, but does not replace the fixed primary comparison. Text-adjusted Reservoir training NLL difference is +0.019103 [-0.014919, +0.050196], also without evidence of a memory-specific gain. No mechanism is identified by these observational control comparisons.

Keep both the pilot and continuation weights. The continuation is not a demonstrated upgrade to the historical-memory candidate. Prioritize auditing the 180s losses and the match between native endpoint supervision and historical reads before repeating this scaling recipe. Improvements in recent replay and short-delay recall suggest useful reader adaptation; general QA adaptation, calibration, read/write mismatch and insufficient old-evidence supervision remain hypotheses, not established causes.

The [public summary](results/reader_scale1024_development_20261128.json) contains all reported arms and intervals, and the [independent verification](results/reader_scale1024_evaluation_verification_20261128.json) records paired counts and hashes. No partial accuracy/NLL selected checkpoints, sources or policies. These are reused development data, one training/memory seed, with both training amount and global source batch1→4 changed. They are not official/unseen benchmark results or a pure data-size ablation.

## Post-hoc regression diagnosis and the next controlled evaluation

A CPU-only saved-prediction analysis verified the actual native processor head/tail tokens are identical across all five delays for each of the 77 targets (385 prompt checks), and all saved target snapshots match the delay0 recent-eight-step window. The original Plain snapshot replay predictions also repeat exactly across delays and match its delay0 recent replay. Therefore the **already measured delay0 recent replay** can be interpreted as a privileged fresh-state target-window replay counterfactual at later delays. No new GPU oracle scores were measured, and this control is not a deployed retrieval method or a change to the frozen primary.

For the continuation, that target replay answers66/77 (85.71%) compared with55/77 (71.43%) for live Reservoir at300s. There are14 target-replay-only correct cases and3 live-only correct cases. This is headroom under privileged target selection and a fresh state; it does not separate selection, recurrent-state interference and readout effects.

All385 saved Reservoir/window stream **audit layouts** (retained indices, positions, capacity and state/byte budgets) are identical before and after training. This does not mean their actual KV/recurrent tensor contents are identical. The five180s Reservoir regressions have:

- Four cases correct under the target-window replay counterfactual, under both checkpoints.
- Four cases correct under the continuation's current recent replay8; these are not exactly the same four as above.
- All five correct under the continuation's text-only condition; all five wrong under its window40.
- Four with no retained target-window step and one with1/8. This is target-window overlap, **not annotated evidence coverage**; recurrent state may still retain old information.
- Pilot gold margins ranging from0.0411 to0.4726, becoming negative after continuation. They are not all epsilon-sized ties.

At300s,49/77 live Reservoir reads contain zero of the protected target's eight steps; mean overlap is0.4675/8. At180s it is41/77 and0.6364/8. These are descriptive properties of the unchanged fixed retention policy. They cannot establish that a writer change caused the regression, nor that these questions require old visual evidence. [Saved-prediction diagnosis](results/reader_scale1024_readout_diagnosis_20261128.json) includes the case IDs/margins and contains no question/option text; full private cases remain on A100.

The next evaluation is fixed **before reading its outputs**: checkpoint `checkpoint-000128` (512 new sources,576 total metadata groups, Adam192), compared with the original64-source pilot and final stage256 checkpoint. This is a post-hoc trajectory diagnostic chosen after inspecting the completed final run, not an unbiased best-checkpoint search. All77 sources /385 questions, four policies and five delays remain included. The diagnostic comparison is300s same-Reservoir NLL, **final256 minus midpoint128**, with a95% exact-byte-cluster bootstrap interval. Other controls and180s errors are exploratory. This observes one intermediate point; data composition and update count change together, so it cannot by itself identify overtraining or its cause.

The midpoint evaluation is live at `/data/yyy/tracevl-runs/reader-midpoint128-eval-89e1f30-001`, inputs `a5d83fc054840bbff3bbfad217238728cdc17204d8871acb4be1bd731fa7c75d`, using the same unchanged inference worker on four A100s. Its12 A100 tests, Ruff, original reducer self-comparison and added trajectory-reducer reproduction of the entire existing final-versus-pilot report passed before GPU work. All four workers are processing sources; no partial accuracy/NLL has been inspected. Old runs and checkpoints remain unchanged. No further training has started.
