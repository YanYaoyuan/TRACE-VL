# Persistent writer controls prepared during the fixed77-source baseline run

This is CPU-tested infrastructure for a subsequent controlled ablation, not an
accepted architecture, trained method, novelty claim or a result of the77-source
experiment. The currently running immutable3dc8c83 archive remains unchanged.

`PersistentWriter` stores actual old16-token values: at most32 history plus4 recent
steps. Appended values are detached and cloned, so later reuse of the caller's
current-frame buffer cannot replace history. No target/question/options/answer enters
append. Reads use the same latest-at-most8-pre-target steps as the FIFO/reservoir
baseline, at most128 tokens. Each policy has its own budget; do not count a joint
container as one method or give it the recent8 control's extra four steps.

Two deterministic controls:

- `time_coverage`: on historical overflow, preserve first/latest anchors and remove
  the interior sample with the smallest two-neighbor temporal span. It isolates
  temporal coverage from content selection.
- `spatial_change`: use the triangle excess of the normalized spatial feature path
  as the first removal criterion; break ties using the same temporal criterion.
  Preserve original values and timestamps, never average or synthesize evidence.
  Each of16 spatial positions is retained in the distance calculation; swapping
  object positions would be invisible in a global-mean descriptor. The rounding
  resolution1e-6 is fixed before any new-policy predictions, not tuned on answers.

Recompute descriptors only from capacity-bounded stored values in transient CPUFP32
workspace. No persistent descriptor bank, growing archive, learned scorer, or hidden
video replay exists. Audit actual tensor payload plus96 logical metadata bytes/slot;
global counters/hash/Python overhead and transient current/descriptor workspace are
separate. This is not processRSS or a zero-overhead claim. Inference speed, numerical
behavior and payload limits still need validation in a frozen-reader GPU integration.

Tests exercise300-step bounded storage, caller-buffer overwrite, repeated and
out-of-order read queries without writes, causal cutoffs, invalid-input atomicity,
constant-stream interior coverage and an isolated A/B/A event whose global mean
feature is unchanged. These are mechanism checks, not evidence of video-QA gains.

Next integration must retain the77-source report and sealed rows. Compare FIFO,
reservoir, time coverage and spatial change with the identical32/4/128 budget,
sampling and frozen reader; record writer state, transient workspace and actual
latency. Keep text/current/snapshot controls separate. Run original state diagnostic
to completion and independently audit it before deciding whether visual benefit and
retention justify this ablation or whether readout needs attention first. A later
confirmation requires independent sources and additional seeds; this reused
cohort remains development data. No training scale-up from a CPU unit test.
