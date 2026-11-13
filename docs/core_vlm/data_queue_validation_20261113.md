# Core video QA data and experiment queue — 2026-11-13

This queue trains the language-model LoRA of the original Qwen3.5-4B from video-question answer loss. It establishes full-token and fixed-grid baselines for the core-model research. It does not train the historical alert heads, implement the proposed T1 architecture, or establish a paper contribution.

## Data preparation

`core_vlm.prepare` reads only the configured LLaVA subsets and their archives. It parses the first question-answer pair, keeps gold answers out of generation prompts, and handles multiple-choice and open-ended records separately. Conflicting answers to the same question on the same clip are quarantined together. Recognizable source aliases are joined before sampling; exact byte duplicates are grouped before the train/validation split. Distinct scene clips from the same source may legitimately have different answers.

Benchmark IDs from the configured existing annotation inventories are excluded. This is ID/alias exclusion, not perceptual matching of renamed or transcoded benchmark videos. It does not establish complete absence of contamination. Final benchmark questions are never used for training or automatic scoring by this queue.

Original media and downloads are read only. Selected missing media are extracted into the fixed data output directory after archive matching; ambiguous or missing media stop preparation. Full decoding, frame-count agreement, increasing finite PTS and file hashes are checked. Invalid media are quarantined only within the configured maximum failure fraction, without replacement sampling. Train/validation minimum counts must still pass. Completion is recorded only after the manifest is written and checked; changed inputs or incompatible identities are refused on restart.

The A100 configuration starts with the 0–30 s and 30–60 s YouTube subsets: at most 6,000 source groups and six questions per group, a fixed source-level validation split, at least 1,000 training and 100 validation groups, and a maximum 5% decode/timestamp rejection fraction. These are prospective pilot settings, not claims about available data counts. Its benchmark roots are OVO-Bench, StreamingBench, Video-MME and NExT-QA. Availability on A100 is checked by the launcher rather than inferred from local files.

## Local verification

- CPU suite: **74 tests passed**, including real small video archives, full decode, alias grouping, duplicate/conflicting QA, split isolation, failure handling, checkpoint validation, paired reports and mismatched observations. Ruff and shell syntax checks passed.
- Real media preparation: 100 selected ActivityNet source videos were checked locally. Four had a repeated final timestamp and failed the strict increasing-PTS rule. The resulting manifest contained 96 source groups: **67 training videos / 201 questions**, **29 validation videos / 87 questions**. Three conflicting QA pairs (six copies) were quarantined before selection. No other videos were substituted.
- This ActivityNet run is an engineering fixture only. It uses a small synthetic benchmark-ID exclusion fixture, not the official full benchmark inventories; ActivityNet-derived data remain excluded from the proposed A100 YouTube pilot. The local failure allowance was raised from 2% to 5% after inspecting timestamp failures, before model evaluation. This is not a preregistered research result.
- Real queue: two RTX 5090s, original Qwen3.5-4B, eight sampled frames, bf16, rank-8 LM LoRA, two updates per arm, four held-out generation questions and four generated tokens maximum. Both the flow runs and main short full/grid runs completed, independently saved/reloaded their checkpoints, and used identical question schedules across arms.
- Peak allocated memory reported on rank 0 was **10.16 GiB**. Native/full parity and the final cross-arm frame, RGB, prompt and timestamp checks passed. This is not a memory estimate for the 16-frame A100 run.
- All four final evaluations completed. Text exact match was 1/4 for original and trained full-token models, and 2/4 for original and trained grid models. These tiny, truncated-generation smoke scores provide **no evidence of a training gain**, semantic QA accuracy or architectural superiority. Bootstrap output on four videos is only a report-format check.

An initial queue attempt stopped at the free-GPU-memory check because another job occupied the second GPU. That job was not terminated. The same queue configuration was restarted once sufficient memory was available. Data and checkpoints are on the requested external local disk; only aggregate validation findings are committed here.

## A100 execution

Use branch `design/v3-transition-streaming`. Fetch it, extract `scripts/launch_core_qa_a100.sh` from the reviewed code commit, and set `TRACE_REV` to that same commit before running. The launcher uses a separate immutable source snapshot; it does not switch or overwrite the operator's checkout. It uses the existing UV environment at `/data/yyy/TRACE-VL/.venv` and the already downloaded model at `/nas-data/vlm/models/qwen3.5_4b`. No model or dataset download is initiated.

Defaults:

- Data: `/nas-data/vlm/yyy/tracevl-data/core-llava-youtube-v1-<revision>/`.
- Experiment: `/nas-data/vlm/yyy/tracevl-runs/core-qa-<revision>-001/`.
- Preparation: 32 media workers, eight archive workers, 30 GiB free-disk reserve.
- Training: four GPUs with at least 30 GiB free each, 16 sampled frames, two flow updates, then 1,000 updates per full/grid arm, gradient accumulation four, rank eight, learning rate 1e-4, fixed seed seven.

The sequence is data preparation → original native-path parity → full/grid short DDP training and reload checks → original full/grid evaluation → full-token QA training/evaluation → grid QA training/evaluation → paired report. Full/grid share data and sample order but have different token counts and compute costs. This is not an equal-compute architecture comparison. The two main arms each start from original weights, not from the flow checkpoints or the historical Stage-2 checkpoint.

Set `TRACE_PREPARE_ONLY=1` to prepare data without GPU work. Otherwise the queue proceeds automatically after its checks. Rerunning the same command checks identities, validates completed checkpoints and reuses completed evaluation records. A failed job stops the queue and retains its logs. No training results are promised until this actual A100 configuration completes.

Inspect the operator log and `logs/` for progress. Training metrics and final checkpoints are under `train/{full,grid}/`; predictions and scores are under `eval/{original_full,original_grid,trained_full,trained_grid}/`. The final marker is `VERIFIED core QA baseline queue`, followed by `summary.json`.

Send back `summary.json`, `run.json`, `launcher.json`, both resolved configuration files, `native-control/report.json`, per-arm training metrics/status, evaluation scores/predictions, preparation `report.json`, `qa-conflicts.json`, `verification.json`, `complete.json`, and the launcher/job logs. Large model weights, media and optimizer checkpoints are not needed for initial report review.

Free-text exact match is reported separately from strict multiple-choice accuracy. Neither the development queue nor its one-seed paired comparisons constitute an official benchmark result. The next research gate still needs a specified T1 hypothesis, ordinary temporal baseline, appropriate temporal tasks and independent evaluation.
