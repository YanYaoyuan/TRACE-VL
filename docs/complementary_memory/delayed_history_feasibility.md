# Delayed history mechanics — CPU-only feasibility scope

This prototype addresses the design question posted on PR #15 (7196908151).
It is not a frozen quality experiment, trained method, benchmark result or
replacement for the fully carried R32 baseline. Historical files stay unchanged.
No new-source media is extracted and no real-question prediction is produced.

## Causal construction

At arrival, pool each dense visual feature grid into fine16 and a declared
historical grid (fine16 or coarse4). Retain only the newest eight pairs in a
bounded pre-LM feature queue. When a pair departs, encode its historical version
once into a lagged history-only Reservoir LM. The live recurrent/conv state ends
at that committed boundary. It never contains the current recent queue.

For a question, clone the lagged cache/state, consume recent8 fine16 exactly once
on that disposable fork, then score the question tail. The history state, owner
indices, RNG and feature queue must be unchanged. A future query repeats the
operation from its own causal boundary; no question conditions future writes.

Both history32-fine and history128-coarse must use this delayed contextualization
in a future primary comparison. Original fully carried R32 remains a separate
strong control: matching selected owner indices does not make their past hidden
states equivalent. No post-attention KV averaging or full-state snapshot ring is
used. This frozen-inference prototype does not support training yet.

## What CPU checks can establish

Tests use synthetic dense features and a tiny actual Qwen3.5 language backbone
on **A100 CPU/hosted CI only**. The independent reference starts with the full
native prefix and HF get_rope_index coordinates, selects per-bin native centres,
and averages canonical dense embeddings; it does not call the reader's
PositionCounter/block_inputs. Mixed16/4 text/timestamp/vision positions and
no-eviction cached-versus-uncached logits are checked. Nondivisible7x9 geometry
separately demonstrates area-aware coarsening versus incorrect unweighted means.

Other checks cover consecutive/causal commitment, exact-once recent replay,
original Reservoir owner law, deterministic question forks, caller tensor mutation,
constant raw-feature/state budgets, steady selected steps and actual KV accounting.

These are **synthetic HF native-layout mechanics checks**. They are not real
processor/video parity, long-stream numerical guarantees, efficacy or novelty.
A real-processor mixed-grid audit, stronger controls, explicit protocol/peer review
and measured execution budget are required before any quality evaluation.

## Budget is not just the visual count

At capacity, recent8x16 + history32x16 and recent8x16 + history128x4 both use
640 visual tokens. History128 retains many more timestamp/delimiter tokens, so
actual KV tokens/bytes can remain larger. A conditional audit on completed
300s owner sets predicts1974.37 versus1036.56 total KV tokens (~1.90x), not equal
KV memory. Measure query-fork transient memory, pre-LM feature queue bytes,
recurrent state, first-answer latency and the additional recent replay cost.
A fixed total-KV-budget control must be designed separately before an efficiency
claim. No new GPU model-quality experiment is authorized by these unit checks.
