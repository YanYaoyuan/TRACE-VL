# Test A protocol: carried native state × historical KV (issue #7)

Status: code written, **not yet executed**. No result or novelty is claimed. Frozen reader, no training.

**Question.** On a continuous stream, does historical KV add answer value beyond Qwen3.5's carried
Gated DeltaNet state, and does that value depend on the state?

## Frozen setup

- Data: the reused 77-source / 385-question development cohort (`core-longstream-states-3dc8c83-001`,
  input `090db3b5…`), delays 0/15/60/180/300 s. No source or question is selected by any outcome.
- Reader: Plain LoRA-2500 (primary run); original Qwen3.5-4B (`TRACE_WEIGHTS=base`, secondary run).
- Entry: one two-frame step, 16 pooled tokens plus native timestamp and delimiters. Each stream calls the LM
  once per step, question-blind, over the whole causal prefix. One vision encode is shared.
- KV budgets: protected prompt; 8 recent steps; Reservoir adds at most 32 older steps. Budgets count steps,
  and actual tokens/bytes are recorded. Recurrent/conv state is fixed-size and never evicted.
- Reservoir labels 17/29/43 are averaged within source. They are memory seeds, not training seeds.
- Anchored questions state a past cutoff in their text. No normal arm filters entries by it.

## Arms

| Arm | State | KV | Kind |
|---|---|---|---|
| `text_only` | none | none | control |
| `recent_replay` | fresh, last 8 steps | last 8 steps | natural |
| `window_stream` | carried | last 8 steps throughout | natural |
| `reservoir:S1K1` | carried | recent + Reservoir | natural |
| `full_stream` | carried | all steps | unbounded reference |
| `snapshot_replay` | fresh | 8 steps ending at target | privileged reference |
| `S1K0` / `S0K1` / `S0K0` | carried / swapped / swapped | K0 drops history on the fork only | mechanism |

S0 installs recurrent **and** conv state from a fresh replay of the prompt and the same last 8 steps, at the
live coordinates. All four factors share one live stream, prompt, recent KV and weights. Limitation: live KV
was computed with the carried state, so K0 recent entries may still contain history. The factorial
therefore does not erase all history; `window_stream` and `recent_replay` are the natural comparisons.

## Required checks (failures stop the run; accuracy never does)

Exact packing/positions versus the native processor; 8-step cached-versus-uncached letter-logit gap ≤ 0.5 in
bf16, with the same choice unless the margin is ≤ twice the gap; causal prefix counts; live KV tensor
identity and recurrent/conv equality after each question; deterministic re-answer; bounded KV; finite logits;
code, library, model and media identities. Smoke uses the reference's shortest and longest sources and
reports no accuracy. The parity gap is short-horizon; it does not certify bf16 drift over a full stream.

## Analysis

Primary endpoint: 300 s, option-normalized gold NLL, fp32 letter head over bf16 hidden. Accuracy is always
co-reported. Source bootstrap, 10,000 draws, seed 7; pointwise, uncorrected, with SE and an approximate 80%-power MDE.

1. Carried state: `window_stream − recent_replay`.
2. **History KV given state: `S1K1 − window_stream`** (primary decision contrast).
3. Interaction: `(S1K1 − S1K0) − (S0K1 − S0K0)`.
4. Headroom: `full_stream − window_stream` and `snapshot_replay − window_stream`.

| Reading at 300 s (NLL) | Next step |
|---|---|
| 0 s carried-state NLL harm, lower bound > 0.10 | Reader unfit; train a long-stream reader first |
| Contrast 2 helps | Proceed to B |
| Contrast 2 unclear, headroom helps | B tests selection only |
| No headroom | Hold; inconclusive if the MDE is large |

State alone failing does **not** reject the idea. An interaction interval crossing zero means the
state-conditional claim is unsupported, even if KV helps. Passing A does not authorize training or scaling.

## Run

On the A100 host, check `nvidia-smi` and record GPUs on issue #7 first. Defaults are GPUs 2–3.

```bash
TRACE_REV=<commit> TRACE_SMOKE_ONLY=1 bash scripts/launch_complementary_test_a.sh
TRACE_REV=<commit> bash scripts/launch_complementary_test_a.sh                    # resumes, full
TRACE_REV=<commit> TRACE_WEIGHTS=base bash scripts/launch_complementary_test_a.sh # secondary
```

`smoke-ok.json` gives seconds per stream step and peak memory: use it to estimate full cost before step 2.
Submit `reports.tar.gz`. Use a new `TRACE_OUT` after any code change. No shared code was modified.
