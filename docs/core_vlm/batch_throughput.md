# True microbatch throughput and W&B

Core QA training now accepts `--microbatch-size`. A microbatch runs multiple
independent questions through the LM together. Per-question answer CE and KL
are averaged equally, with EOS included. Each example retains its native M-RoPE
positions and causal visual observations. Right padding has an attention mask.

For the four A100s, compare these settings at global batch 16.

| Questions per GPU | Accumulation | Global batch |
|---|---|---|
| 1 | 4 | 16 |
| 2 | 2 | 16 |
| 4 | 1 | 16 |

The question order per rank is identical at every optimizer update. Both
architecture arms receive the same setting. Changing these settings requires a
new training directory because configuration and code are part of run identity.
Previously completed runs should be analyzed using their saved source snapshots.

Run `scripts/launch_core_batch_benchmark_a100.sh` with `TRACE_REV` pinned to the
batch commit. The launcher uses the existing prepared mixture at
`/nas-data/vlm/yyy/tracevl-runs/core-scale20k-b2919d8-001/mixture/manifest.jsonl`.
It freezes a deterministic 256-question training subset and verifies its video
hashes. This avoids re-hashing all 20k training media for each short trial.
`TRACE_PREPARED` overrides that preparation directory. It starts six short trials,
plain and temporal at each batch size, for 12 updates each. The first two updates
are warmup. Each trial starts from the original model and follows the first 12
updates of the same declared 2,500-update schedule. Trials write partial
checkpoints in their own directories. `TRACE_BENCH_STEPS` changes the short trial
length, and `TRACE_RECIPE=ce` disables the preservation teacher for a CE benchmark.

`batch-report.json` records throughput, maximum allocated/reserved GPU memory
across all ranks, first/final losses, sample order and checkpoint identities. The
recommendation chooses a common configuration for both arms, with reserved
memory at most 32 GiB and first-loss difference from batch 1 at most 0.02.
These numerical tolerances are benchmark checks. They do not replace efficacy
evaluation. `batch-reports.tar.gz` includes this report and trial logs.

Local validation passed 156 core/run-I/O/tracking tests, including the real W&B
0.31.0 SDK in offline mode. Tests compare losses and every trainable gradient
against sequential accumulation on a tiny Qwen3.5 model with both full and linear
attention, different question/answer lengths, temporal parameters and gradient
checkpointing. CPU tests also check sample isolation and unchanged sampling.
The local NVIDIA driver was unavailable. The subsequent owner-supplied A100
measurements are audited below; local CPU validation does not replace GPU testing.

## W&B

The new benchmark and scale/preservation launchers default to `TRACE_WANDB=1`.
They default to `WANDB_MODE=offline` for the A100 machine without Internet.
The default project is `trace-vl-core`. The existing project extra `track`
provides the SDK. If adding it to an already working environment, install the
locked SDK using that environment's Python and `uv pip install wandb==0.31.0`.

Use `WANDB_PROJECT` and `WANDB_ENTITY` to choose the destination.
`TRACE_WANDB_GROUP` groups plain and temporal runs under one experiment. W&B
training curves include CE, teacher KL, learning rate, gradient norm, batch
sizes, samples/second, answer tokens/second, and GPU memory peaks. Evaluation
report generation logs choice accuracy, strict accuracy and format validity,
and stores its run identity, predictions and report as an artifact.

Only rank 0 initializes the training tracker. The shared existing tracker
handles startup failures by collecting offline when possible and preserves
local metrics if logging fails. Normal tracker shutdown occurs after DDP
shutdown. Resumed invocations create separate W&B runs with the same saved
training identity in config. `WANDB_MODE=offline` explicitly collects locally.
For cloud viewing later, move the offline runs to a connected machine, authenticate
with `python -m wandb login`, then use `trace-vl v3 wandb-sync --runs RUN_DIRECTORY`.
`TRACE_WANDB=0` disables tracking. API keys belong in W&B login or environment
variables. Experiment scripts and configs contain no credentials.

## Audited A100 result, 2026-11-18

The supplied `batch-reports` bundle contains the completed report, six trial logs,
trial summaries and source/data identity. All 12 updates in each trial have finite
losses and nonzero finite gradients. Per-rank question IDs, learning rates, answer
token counts and teacher question counts match across all six runs. Recomputing
the medians from the logs reproduces the report. Initial LoRA digests agree. Saved
checkpoint metadata is included; checkpoint weights themselves were not in this
local bundle and were not independently rehashed here.

| Arm | Microbatch | Accumulation | Seconds/update | Samples/second | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| Plain LoRA | 1 | 4 | 14.958 | 1.070 | 10.781 |
| Plain LoRA | 2 | 2 | 13.074 | 1.224 | 10.781 |
| Plain LoRA | 4 | 1 | 11.904 | 1.344 | 10.783 |
| Temporal + LoRA | 1 | 4 | 15.076 | 1.061 | 10.789 |
| Temporal + LoRA | 2 | 2 | 13.130 | 1.219 | 10.789 |
| Temporal + LoRA | 4 | 1 | 11.975 | 1.336 | 10.791 |

At global batch 16, microbatch 4 reduces the combined step time from 30.034 to
23.878 seconds: 20.5% less time, or 25.8% greater throughput. First-loss differences
from microbatch 1 are 0.01598 (batch 2) and 0.00999 (batch 4), within the previously
declared 0.02 tolerance. GPU results are numerically approximate, not bit-identical.
This is a throughput result, not evidence of accuracy or architectural improvement.

Provenance: prepared completion SHA256
`e26c3e57e97b872db49afea7d3f8baee12ee4e460687eb3aaddf2e40e4cf248d`,
training manifest SHA256
`eddd5110998c5bbfc8ab43df81986f2cd26e8ff2ca4f364cd805c52b81cddd52`.
The prepared completion hash matches the supplied scale-preparation bundle.

The old benchmark's final W&B comparison summary failed because an in-memory
dictionary used integer batch-size keys. These keys are now strings, and a real
offline-SDK test covers the complete recommendation summary. Benchmark results in
the JSON and local logs remain valid. The new queue forces offline mode, so no
W&B authentication or network connection is required.

## Run the prepared expansion

`scripts/launch_core_scale_train_a100.sh` takes a pinned `TRACE_REV` and defaults to:

- Prepared data: `/nas-data/vlm/yyy/tracevl-runs/core-scale20k-b2919d8-001`.
- Batch audit: `/nas-data/vlm/yyy/tracevl-runs/core-batch-43fdd72-001`.
- A new output folder: `core-scale20k-mb4-<revision>-001` under the same runs root.

The launcher runs CPU regressions, checks preparation and benchmark identities,
re-audits the completed preservation pilot checkpoints, and reuses the prepared
mixture. It rehashes training media and checks frozen external data; it does not
rescan archives or repeat full video decoding. Keep the original preparation
folders because manifest paths can refer to media extracted during earlier runs.

The queue is a new owner-authorized expansion experiment after the failed pilot
gates, whose decisions remain recorded. Both arms start from the original
Qwen3.5-4B weights, with the same 19,997 training questions (10,000 general and
9,997 temporal), seed, LR, global batch 16 and 2,500-update schedule. It trains
ordinary LoRA and causal temporal + LoRA sequentially, with original-teacher KL
weight 0.5 on general QA. Both receive a two-update smoke train/evaluation before
full training. Final checkpoints are compared with the original-grid model on
the frozen OVO/NExT-QA development populations, including HLD/non-HLD retention
checks and paired intervals. Final-test data stays frozen.

No additional reviewed answerability pairs are supplied by default. This run is
QA plus preservation KL, not a newly supervised refusal model. An explicitly
reviewed training-only file can be passed as `TRACE_REVIEWED_PAIRS`; it is checked
by the existing answerability loader and recorded in the new identity. Changing
inputs, supervision or source requires a new output directory.

At the short-trial rates, the two 2,500-update trainings project to 16.6 hours
combined, excluding smoke checks, media hashes, checkpoint writes and evaluation.
This is a provisional estimate on 256 sampled training questions; full-mixture
lengths, padding and cache behavior can change throughput. Follow the actual
`train/{plain,temporal}/metrics.jsonl` rates during the run.

W&B runs and curves are collected locally, grouped by the output folder. The final
`reports.tar.gz` bundles reports, training metrics, predictions, smoke diagnostics
and offline W&B runs, excluding model checkpoints. The launcher resumes completed
training/evaluations using existing identities and verified outputs. It never
starts a further expansion after this experiment finishes.
