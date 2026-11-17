# Core VLM expansion: prepare alongside the running pilot

The main experiment remains video QA through the language model: ordinary LoRA
versus a causal temporal module plus LoRA. This queue uses Qwen3.5-4B and the same
observation settings as the preservation pilot. It is expanded adaptation of a
pretrained model, not foundation-model pretraining. No improvement is assumed.

## Fixed target

| Item | Target |
|---|---|
| General video QA | 10,000 MC questions from downloaded 0–30 s and 30–60 s YouTube subsets |
| Temporal QA | 10,000 TN/TC/TP questions from official NExT-QA **train** |
| Unique videos | At least 500 per family after exclusions |
| Arms | Ordinary LoRA; causal temporal module + identical LoRA |
| Updates | 2,500 per arm, fresh original backbone |
| Effective batch | 4 GPUs × 4 accumulation = 16 QA |
| Presentations | 40,000 per arm, two dataset-pass equivalents |
| Observation | 8 blocks / 16 frames, 65,536 pixels, 4×5 grid |
| Evaluation | Existing frozen OVO development and NExT-QA validation populations |

The arms match examples, seed, learning rate, update count and visual tokens.
They are not FLOP/parameter matched: the temporal module adds computation.
The running pilot chooses between CE and anchored CE+KL; both expanded arms use
the same chosen recipe. A clean CE queue omits teacher computation in both arms.
There are no invented unanswerable labels in this expansion.

20,000 is a requested target, not an assertion about available decoded media.
CPU preflight counts raw temporal QA first, then counts eligible questions after
exclusions before expensive extraction. Insufficient data stops with a report;
there is no silent downsizing, replacement sampling or benchmark-data borrowing.

## Data isolation and resources

The old general-QA validation sources and exact media bytes are protected, even
under renamed files. The existing general source-split rule is retained.
NExT-QA validation videos, frozen external sources, and benchmark inventories are
excluded from training. Official annotations and media checksums are recorded.
No final-test inference is run. The current experiment's data/source snapshot is
read only; new data and reports live under a new NAS run directory.

Default preparation uses 16 CPU workers and four archive readers, with CUDA hidden.
This limits concurrent NAS pressure while the pilot runs. It does not download
weights or datasets. Selected missing media are extracted and fully decoded;
existing verified training media can be reused. Temporary files also go to NAS,
not the nearly full root filesystem. Keep original archives and previous runs.

## A100 operation

Fetch the published revision and extract its launcher without switching the
working tree used by another job. Replace `PUBLISHED_SHA` with the published commit.

```bash
cd /data/yyy/TRACE-VL
git fetch origin design/v3-transition-streaming
REV=PUBLISHED_SHA
git show "$REV:scripts/launch_core_scale_a100.sh" > /tmp/launch_core_scale_a100.sh
nohup setsid env TRACE_REV="$REV" TRACE_SCALE_MODE=prepare \
  bash /tmp/launch_core_scale_a100.sh \
  </dev/null > core-scale-prepare.log 2>&1 &
```

Preparation success prints `PREPARED SCALE DATA` and a report bundle. Submit
`queue-plan.json`, `scale-prepare-reports.tar.gz` and `core-scale-prepare.log`.
The default folder is
`/nas-data/vlm/yyy/tracevl-runs/core-scale20k-<revision first 7 characters>-001`.
Inspect `capacity-preflight.json`, `mixture/capacity.json` and media audits on failure.
Rerun the identical pinned command to reuse validated outputs; keep configuration
and worker settings unchanged for the same directory.

After the running pilot is complete, review its paired external results and
capability-retention checks. Choose one recipe, record the reason and hash the
reviewed `report.json`. The training command uses the **same revision and folder**:

```bash
# Set TRACE_RECIPE to ce or preserved, TRACE_PILOT_REPORT to the completed
# preservation pilot's report.json, TRACE_PILOT_SHA256 to its reviewed SHA256,
# and TRACE_REVIEW_NOTE to the concrete reason for choosing this recipe.
nohup setsid env TRACE_REV="$REV" TRACE_SCALE_MODE=train \
  TRACE_RECIPE="$TRACE_RECIPE" TRACE_PILOT_REPORT="$TRACE_PILOT_REPORT" \
  TRACE_PILOT_SHA256="$TRACE_PILOT_SHA256" TRACE_REVIEW_NOTE="$TRACE_REVIEW_NOTE" \
  bash /tmp/launch_core_scale_a100.sh \
  </dev/null > core-scale-train.log 2>&1 &
```

The launcher verifies pilot completion against actual final checkpoints, runs
mandatory CPU tests and two-step DDP checks, then queues the two full training
arms and external evaluations. All four GPUs must have at least 30 GiB free.
Preparation does not automatically start this GPU phase when the other job ends.
The release record is immutable; recipe changes require a separately reviewed run.

The final report retains the architecture comparison and adds separate HLD and
non-HLD checks against original-grid predictions for both arms (95% CI lower
bound above −2 pp). An inconclusive interval is not proof of degradation. These
are development checks, not camera false-alarm metrics or official leaderboard
scores. No further training is automatically triggered by a result.

## Two-week execution budget

Prepare and audit now while the current pilot runs. Days 2–5 are reserved for the
chosen matched expanded pair; days 6–9 for repeat seeds and temporal ablations if
the external evidence supports continuing. Days 10–12 are reserved for the frozen
final evaluation once the method is fixed, and days 13–14 for tables and writing.
These are planning windows, not measured training-duration promises.

`queue-plan.json` projects training time from positive adjacent update times in
the running pilot, using the median of up to 200 recent intervals. It omits
extraction, evaluation and changed data-length costs. Use that measured estimate
to adjust the schedule before GPU release; do not open multiple new research
tracks or silently alter the comparison to meet a wall-clock target.

Local validation covers core/run-I/O tests, real archive extraction and protected
held-out byte copies, fixed family counts, live-log timing, release identity and
checkpoint failure handling. The full 20k inventory and real four-A100 execution
remain to be validated on the operator's machine.
