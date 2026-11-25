# Review follow-up and next research decision

The review correctly identifies the main issue: the experiment record is stronger than the current method contribution. Frozen-reader diagnostics should now close as a development record. The next research experiment should train a question-conditioned reader with a matched long-stream training baseline.

## Independently checked observations

The committed 77-source data reproduce these descriptive results:

| Observation | Recomputed value |
|---|---:|
| Three-seed Reservoir accuracy at 300 seconds | 70.56% |
| Text-only accuracy | 54.55% |
| Three-seed mean minus text | +16.02 pp |
| Additional source-bootstrap 95% interval | [+3.90, +27.71] pp |
| Text and snapshot both correct | 38 sources |
| Text and snapshot both wrong | 9 sources |
| Text wrong, snapshot correct | 26 sources |
| Text correct, snapshot wrong | 4 sources |
| Usable bank at most eight steps | 56 / 231 source-seed cases |

The added interval uses 4,000 draws, seed 7, and the preserved order of the frozen source records. Finite bootstrap endpoints can differ with resampling implementation or row order; the mean effect is unchanged. This is an additional exploratory analysis after review. The original prespecified single-memory-seed +19.48 pp comparison remains in the historical report and is not relabeled as a three-seed result.

The outcome-conditioned 26-source subset yields 62.82% / 69.23% / 67.95% for latest8 / uniform8 / all_bank across three seeds. It is descriptive only, is distinct from the earlier 26-question past-event cohort, and must not replace the primary cohort or become a correctness-based training filter. Agreement between text and snapshot does not prove that a source is uninformative for every possible method.

All-bank reads also change context length: the actual maximum is 25 two-frame steps, whereas the LoRA was trained on eight steps. Its null result combines read coverage with a change relative to the training distribution. It cannot establish that a trained reader would fail.

Machine-readable derivation: [review_followup.json](results/20261125/review_followup.json).

## What to adopt, and what needs qualification

- Stop adding heuristic writer/readout variants on the frozen reader. Retain Reservoir and the complete negative results as controls.
- Make one focused training comparison: train question-conditioned retrieval and LM LoRA with QA supervision, against a matched Reservoir reader trained on the same long-stream examples. Keep initialization, input representation, capacity, data order, updates, and checkpoint-selection rules explicit. A standard retrieval mechanism is a control until a distinct contribution is established.
- Build official OVO-Bench and StreamingBench adapters during training preparation, rather than waiting for final weights. Run original 4B and Plain LoRA development baselines with protocol and modality clearly labeled. Reserve unused sources for final evaluation after configuration and checkpoint selection are frozen.
- Use a small new module for the trainable reader. Reuse verified native packing, causal decoding, data identities, and reporting utilities; do not entangle the new architecture with every historical experiment entry point.
- Increase the number of suitable independent development sources. A target of about 300 is a starting point, not a guarantee of power for a +2 pp effect. Choose sample size using plausible paired disagreement rates, source dependence, the actual endpoint, and a prespecified minimum useful effect. Do not keep adding sources until an interval becomes positive.
- Treat the approximately 12.5 pp snapshot-versus-Reservoir gap as a difference between these specific controls, not a strict performance ceiling. Snapshot selection has privileged timing and may omit relevant evidence. Likewise, an oracle gain below 5 pp cannot alone reject the entire memory direction without checking task validity, evidence coverage, model training, and statistical precision.

## Next experiment contract

1. Audit existing training sources for valid questions that depend on evidence outside the recent window. Natural continuation must preserve the answer and make the historical referent unambiguous. Do not simply remove a numeric cutoff from a state question if that changes its meaning.
2. Normal inference receives the question and causal stream state. Timestamps may be stored with observations, but hidden evidence intervals or target addresses are not passed to the reader unless the public question/protocol provides them. Evidence annotations may support training or a separately labeled oracle control.
3. Compare trained Reservoir and the trainable reader under matched state/read budgets. Include text, recent-window, and clearly labeled oracle-time/native-spatial diagnostics to assess task sensitivity and representation limitations. Larger-token controls are not equal-budget method gains.
4. Before execution, fix the cohort, primary endpoint, uncertainty calculation, no-harm checks, training budget, and review checkpoint. Evaluate whether a proposed 20 December checkpoint is feasible after the data audit and measured throughput; it is not an externally verified conference deadline.
5. Expand to three training seeds after a credible development gain over the matched trained baseline. Preserve all existing state and past-event failures. Public final evaluation follows model/configuration freeze; adapter construction and development baselines can begin earlier.

## CI correction

GitHub Actions run `45575168750` was independently inspected: 22 failures, including 12 missing `peft` imports and 10 missing `ffmpeg` executables. The earlier 538 local passes did not establish hosted CI success.

The prepared fix installs `ffmpeg`/`ffprobe`, includes the existing `lora` extra during dependency synchronization and test execution, and preserves the same extras across all smoke-script commands. Hosted success must be checked on the commit that actually reaches the PR. No experiment results are changed by this infrastructure fix.
