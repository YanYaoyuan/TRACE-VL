# Core video VLM long training

The study measures whether the causal temporal adapter gains over plain LoRA
with more distinct videos and a longer schedule. The frozen Qwen3.5-4B
original-grid results are the reference for both arms. Both arms use the same
LoRA initialization, observation pipeline, examples, seed, update count, and
answer CE plus original-model KL on general QA.

| Setting | Value |
|---|---|
| Training data | 30,000 general MC QA + 9,997 official NExT-QA train temporal QA |
| Added questions | 20,000, with earlier unused source videos prioritized |
| Updates and presentations per arm | 5,000 × 32 = 160,000 |
| Checkpoints | Save every 250 updates. Score steps 2,500 and 5,000. Step 5,000 is primary. |
| Transfer and retention | Same OVO diagnostic and NExT-QA validation questions as the 20K run |
| Throughput | Actual examples per second, peak GPU allocation and reserve, sample preparation wall time on rank 0 |

The 20K training mixture stays intact, including the known missing-video
exclusion. The data stage searches the existing 0–30 s and 30–60 s LLaVA YouTube
archives for up to 8,000 additional source groups, checks file availability and
decodes candidates on CPU. Any unused training QA needed to meet the fixed
target comes from the earlier verified general pool. Report the number of new
source groups and distinct video bytes from `mixture/inventory.json`. Missing
and undecodable paths are recorded. Each category has a 0.2% maximum rate in
the new candidate pool. Source and exact-byte overlap checks protect the
existing validation splits. OVO final is reserved for the final evaluation.

The launcher times three layouts with the same global batch of 32 on both
arms using ten steps per layout, discarding two warm-up steps. The layouts are
microbatch 4 with serial decode, microbatch 4 with parallel CPU decode, and
microbatch 8 with parallel CPU decode. Both batch-4 layouts accumulate twice.
Use the fastest common layout after checking loss tolerance and peak reserved
memory. The selection is recorded once per run. Short trials have their own
checkpoints. Long training starts both arms from the same original weights.

The original 20K run measured about 1.3 examples/s on the four A100s. At that
rate the two long runs take about 67–69 hours of training, plus data preparation
and evaluation. The new run writes a projection from the chosen layout to
`long-scale-plan.json`. Compare fixed development checkpoints to measure the
effect of the additional updates. Comparisons with the 20K run combine changes
in dataset size, family mix, effective batch and total step budget.

Run `scripts/launch_core_long_scale_a100.sh` with `TRACE_REV` set to its Git commit.
The launcher uses the existing A100 environment and logs to SwanLab online with
local recovery. `TRACE_PREPARE_ONLY=1` prepares data on CPU if the GPUs are busy.
Rerun with the same revision and output directory to resume. Submit
`reports.tar.gz`, which includes the final and midpoint development results,
data inventory, throughput tests, and training logs.
