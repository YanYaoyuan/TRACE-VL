# SwanLab online tracking

The owner selected SwanLab online on 2026-11-18. The current
`scripts/launch_core_scale_train_a100.sh` defaults to `TRACE_TRACKER=swanlab`,
`TRACE_SWANLAB=1`, `SWANLAB_MODE=online` and personal project `trace-vl-core`.
It keeps the prepared mixture, learning objective, two architecture arms, batch
and evaluation populations unchanged. No training is needed just to upload logs.

## Setup on the existing A100 environment

Install into the existing environment without rebuilding CUDA or model packages:

```bash
cd /data/yyy/TRACE-VL
uv pip install --python .venv/bin/python 'swanlab==0.10.5' \
  --index-url https://mirrors.aliyun.com/pypi/simple/
.venv/bin/swanlab login
```

Enter the SwanLab API key in the login prompt, not in scripts, config, Git or chat.
For a full environment installation from the new source, the updated lock also
supports `uv sync --locked --extra dev --extra vlm --extra swanlab`.

`TRACE_SWANLAB_PROJECT` changes the project name. `TRACE_SWANLAB_WORKSPACE` sets
the team workspace; leaving it unset selects the logged-in user's personal space.
These TRACE-prefixed variables deliberately avoid the SDK's structured
`SWANLAB_PROJECT` setting, which expects a settings object in version 0.10.5.

The launcher verifies login before GPU work. Training initializes the tracker only
on rank zero and closes it after DDP shutdown. It reports the actual mode and
experiment URL. Transient network failures may fall back to local recording;
that mode is explicitly reported and is not called a successful cloud upload.
Native `metrics.jsonl`, reports and checkpoints remain available independently.

## What is recorded

- Per-update loss, QA CE, preservation KL, LR, gradient norms, sample and answer
  token throughput, effective/micro batch, and peak GPU memory across ranks.
- Per-run configuration, run identity, completion status and SDK hardware/console
  records. The two architecture arms share one experiment group.
- Evaluation choice accuracy, strict answer accuracy, format validity and reports.
- Final paired comparisons, capability-retention checks and raw predictions.
- Local training metrics and operator logs, when supplied to the uploader.

Public SwanLab's [`save()` API](https://docs.swanlab.cn/en/api/py-save.html) is
self-hosted-only. The public-cloud implementation therefore uploads UTF-8 files
as ordered `swanlab.Text` media. Each file record has its relative path, part
count, raw byte length and SHA256; Unicode and CRLF are preserved. Parts can be
concatenated to reconstruct the original text. This is not a downloadable
checkpoint/artifact API. Weights, dataset videos, training-QA manifests, per-question
cache files, source snapshots and SDK cache directories are not collected.

The uploader does not draw new model-quality conclusions. It publishes the
existing scores, including failed gates, as recorded.

## Existing experiments or report bundles

Use the new source snapshot or a checkout containing this implementation:

```bash
TRACE_RUNS=/nas-data/vlm/yyy/tracevl-runs/core-batch-43fdd72-001 \
  bash scripts/upload_core_swanlab.sh

TRACE_RUNS=/nas-data/vlm/yyy/tracevl-runs/core-preservation-6bfd39f-001 \
TRACE_OPERATOR_LOG=/data/yyy/TRACE-VL/core-preservation.log \
  bash scripts/upload_core_swanlab.sh
```

Only provide `TRACE_OPERATOR_LOG` if that file exists. The collected file paths
remain distinct across arms, checkpoints and external operator logs. Reports
copied to another machine can also be uploaded by pointing `TRACE_RUNS` to the
extracted directory. The receipt is written to `.swanlab-upload/receipt.json`.
Its actual mode and text-file hashes are recorded; an offline receipt is not a
cloud-delivery confirmation. The SDK transport performs the actual cloud upload.
The collector uses a stable cloud experiment ID for the same source root/project/
workspace so repeating an upload resumes that record.

To follow a currently running experiment without restarting its training:

```bash
TRACE_RUNS=/path/to/active/experiment TRACE_SWANLAB_WATCH=15 \
  bash scripts/upload_core_swanlab.sh
```

The collector reads growing metrics/log files and plots newly observed updates.
Ctrl-C finishes it and publishes the current text-file snapshot. This collector
does not change training files or checkpoints. It can run in a separate terminal.

## Validation

Tests cover online initialization parameters with a fake transport, backend
selection, numerical filtering, exact multi-part text reconstruction, relative
paths, existing log replay, a growing training log, and file exclusions. A real
SwanLab 0.10.5 SDK test records scalars and multi-part Unicode/CRLF text in offline
mode, then checks the actual stored media bytes. Online authentication and delivery
must be verified in the owner's logged-in environment; no account credentials are
available to the local test process.

Validation passed 178 core/run-I/O/tracking tests, including real offline tests
for both SwanLab and W&B. The owner's actual six-trial batch bundle also replayed
successfully with the real SDK in offline mode: six scalar series and all nine
text files. This validates the replay path; it was not uploaded to an account.
