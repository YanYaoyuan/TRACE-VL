# Reader visual-dependence controls (PR #9 follow-up)

Frozen before scoring. Evaluation only; no training or scale-up follows automatically.

## Question and cohort

Does reading retained old video improve answers on the 115 reused ActivityNet
development questions (one per source), and did the 200-update training change
that? All 115 sources stay in every unmatched analysis, including the four whose
bank stores no annotated evidence. The completed CPU audit is reused: its summary
hash is recorded, and its hit counts only serve as an input-identity gate.

Weights: starting Plain-LoRA 2500 (`start`, primary), and both 200-update readers
under every policy. Selector parameters are never loaded; `learned8` replays the
saved selections. The two trained readers must reproduce their own saved option
logits within 0.25, or the run stops.

## Reads

| Arm | Steps / visual tokens |
|---|---|
| `text_only` | 0 / 0 |
| `recent4` | newest 4 / 64 |
| `latest8` | newest 8 retained (4 recent + 4 Reservoir) / 128 |
| `learned8` | saved selector reads / 128 |
| `spread8`, `random8` | even spacing; uniform from the bank / 128 |
| `bank_evidence8` (privileged) | evidence steps first, then random fill / 128 |
| `pair_evidence`, `pair_distractor` | matched subset / at most 128 |
| `all_bank` | 36 / **576**, separate budget and context shift |

Five read seeds (17, 29, 43, 61, 83) apply to `random8`, evidence fill and pair
fill. NLL and accuracy are averaged over seeds within a source before inference.
A step is evidence if either sampled timestamp lies in the annotated interval
(inclusive). More than eight evidence steps are evenly thinned. If none is
stored, `bank_evidence8` is all random fill.

`bank_evidence8` is limited to what the writer retained. **Raw-video evidence
retrieval is a different arm and is not scored here.** Both are privileged.

## Blank controls

Every visual read has a twin with identical steps, timestamps, positions and
token count. A 16-frame constant uint8-128 clip of the same frame size passes
through the frozen vision tower in one eight-step batch; 4x4-pooled step 0 is
reused at each read position. Blanks are generated once per frame size, hashed,
and shared by all weights and shards. Position equality is asserted for each
twin. A blank is still an unusual input, not a perfect absence of information.

`evidence - blank evidence` isolates visual values given the oracle's timing.
`blank evidence - text` and `blank evidence - blank latest` expose timing cues
and prompt-format effects.

## Age-matched distractors

Exact matching can be impossible because membership depends on time. Evidence
steps are matched one-to-one without replacement to non-evidence steps in the
same bank by step end time, caliper 10 s, maximizing pair count and then
minimizing total absolute difference (exact one-dimensional dynamic program).
At most eight evenly thinned pairs are read; both arms share the same fill and
length. Unmatched sources are excluded from this contrast only. `matching.json`
reports failures, unmatched steps, shortened arms and residual signed/absolute
imbalance. Age confounding is **not** claimed to be eliminated, the matched
subset is selected, and adjacent distractors may show the event if annotation
boundaries are loose.

## Analysis

Gain is positive when the first arm is better. Metric: gold NLL normalized over
the question's options; accuracy is co-reported. Source bootstrap: 10,000 draws,
seed 7, 95% percentile intervals; MDE80 = 2.8 SE.

Co-primary, both required, on `start`: `bank_evidence8 - text_only` and
`bank_evidence8 - latest8`. All other contrasts are descriptive and unadjusted.

| Observation | Reading |
|---|---|
| Both primary intervals above 0, and above 0 versus its blank | Headroom supported; may plan the next stage |
| Both primary above 0, blank contrast not | Headroom, but oracle timing cues are unresolved |
| A primary upper bound below 0.05 nats | That headroom is small on this cohort |
| Otherwise | Inconclusive; report MDE80, do not infer absence |
| Text-only accuracy at least 85% | **Scrutiny**, not retirement: weigh oracle headroom and uncertainty |
| A gap shrinks after training (paired interval below 0) | Reduced measured visual dependence; not by itself shortcut learning |

## Next stage, only if useful headroom is established

1. Train matched `random8`/`spread8` readers first. A learned selector must beat
   the trained random/spread reader, not only `latest8`.
2. Change age/content normalization alone.
3. Change the gradient surrogate (for example per-slot) alone.

Each selector must first pass `behavior.probe`: balanced random ages, recall of
question-relevant entries at least 0.9, and different questions change the read
at least 90% of the time. Positive and negative controls for the probe are in
the tests. Nonzero gradients are insufficient; passing is necessary, not
evidence of QA benefit.

## Limits

Reused single-seed development data; two-option caption-order questions; LoRA
trained on eight-step reads; replayed sparse LM context, not a carried stream.
No official score, novelty or efficacy claim. Test A is independent.

## Run

Check GPU occupancy and record it on issue #7 first. Default GPUs 0-1; at least
18 GiB free each. Smoke writes no metric and its bundle excludes scores.

    TRACE_REV=<commit> TRACE_SMOKE_ONLY=1 bash scripts/launch_reader_controls.sh
    TRACE_REV=<commit> bash scripts/launch_reader_controls.sh
