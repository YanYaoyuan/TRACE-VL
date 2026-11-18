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
The local NVIDIA driver was unavailable. Four-A100 training, GPU memory and
throughput remain to be measured by this launcher.

## W&B

The new benchmark and scale/preservation launchers default to `TRACE_WANDB=1`.
The default project is `trace-vl-core`. Authenticate once with the A100
environment's `python -m wandb login`. The existing project extra `track`
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
Later sync those directories with `trace-vl v3 wandb-sync --runs RUN_DIRECTORY`.
`TRACE_WANDB=0` disables tracking. API keys belong in W&B login or environment
variables. Experiment scripts and configs contain no credentials.
