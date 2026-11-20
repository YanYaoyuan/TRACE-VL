# Mixed-core checkpoint diagnostics

Evaluation only, after the 1,000-update mixed QA pilot at `97272fb`.
The architecture feasibility gate failed and is unchanged. This follow-up tests
whether the learned temporal residual affects answers and whether its gains
rely on ordered visual features. It launches no training or scaling.

## Fixed inputs

Reuse the final plain-LoRA and causal-temporal-plus-LoRA checkpoints and all
existing ordered results from `core-mixed-temporal-97272fb-001`. Reuse the exact
674 OVO / 1,744 NExT-QA development questions, decoded frames, prompts, 8 two-frame
blocks, 4x5 spatial pools, generation settings and first-token choice scoring.
No resampling, new data, downloads or evaluation of frozen final test data.
Checkpoint contents, old predictions and manifests are hashed before use.

## New evaluations

| Arm | Checkpoint | Change |
|---|---|---|
| temporal_off | temporal | Remove the temporal residual after loading; keep its LoRA |
| plain_shuffle | plain | Permute encoded two-frame content blocks |
| temporal_shuffle | temporal | Apply the identical permutation |

The permutation is a deterministic nonidentity bijection from the video-byte
SHA256 and fixed seed 17. Original chronological timestamp slots and spatial
grids remain fixed. Permute before the temporal module. This preserves the
observed content and token count, but creates a content/timestamp mismatch.
This is **not raw-frame shuffling**: within-block order and encoder time
information may remain. It is an order-sensitivity diagnostic, not proof of
action-order reasoning. Removing a module while keeping its co-trained LoRA
is also an intervention at inference, not an isolated training treatment.

## Checks and statistics

1. Mandatory CPU regression tests, frozen-input and GPU-availability checks.
2. Four fixed questions per dataset: replay the ordered plain and temporal
   arms, requiring identical choices, generated token IDs, observations and
   visual-token counts to the old predictions. Run all three new arms on the
   same questions and audit their intervention metadata and token budgets.
3. Six full evaluations (three arms x two datasets), four GPU shards per arm.
   Each question has a fresh model state. Existing question files resume under
   exact run identities. Keep the original experiment untouched.
4. Recompute paired choice accuracy, strict-format accuracy and format-valid
   rates, with source-video cluster bootstrap (4,000 draws, seed 7). Report
   all questions, the existing primary/no-harm groups and individual tasks.
5. Compute the direct paired interaction
   `(temporal ordered - temporal shuffled) - (plain ordered - plain shuffled)`
   on the same bootstrap draws for all four arms. Comparing separate intervals
   is not an interaction test. A wide interval containing zero is inconclusive.
6. Inventory every HLD question, options, gold label and predictions from all
   arms. Option-letter frequencies are not semantic hallucination measurements.

All intervals are exploratory and unadjusted for multiple comparisons. There
is only one training seed. These diagnostics cannot rescue the previous gate.
Choose any subsequent training change from the complete report, not a favorable
subset; write a new hypothesis and comparison before training.

## Run on A100

Fetch `design/v3-transition-streaming`, export `TRACE_REV` to the reviewed commit,
then extract and execute `scripts/launch_core_diagnostics_a100.sh` from that
commit. The launcher uses the existing UV environment and immutable source
snapshot; TMPDIR is inside the NAS output folder, not the nearly full root disk.
It requires the original model, final checkpoints and prepared media to remain
at their recorded paths. Default output is `core-diagnostics-<revision>-001`.

Send `reports.tar.gz`: it includes `digest.txt`, `report.json`, `hld-analysis.json`,
`hld-before.json`, `inputs.json`, `flow-ok.json`, manifests, reused and new raw
predictions, and per-job logs. No model weights or videos are included. The
outer shell log can additionally be sent if the launcher stops before bundling.
`VERIFIED MIXED DIAGNOSTICS` plus `REPORT BUNDLE` signals successful completion.
