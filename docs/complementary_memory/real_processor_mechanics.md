# Proposed real-processor mechanics audit (not yet executed)

This is a separately bounded next step after the CPU synthetic feasibility checks.
Review the causal/readout and independent-reference assumptions on PR #16 before
launch. No training, fresh-source extraction, quality experiment or full-run
promotion follows automatically. Historical inference source stays unchanged.

Use only the two already-used Test A smoke sources, selected by the original
sealed input, and their first **16 encoder steps / 32 frames**. Check original
media/metadata hashes, PTS schedule, model, Plain LoRA and runtime identities.
Never use their benchmark question/options/gold as model input. All sources see
one fixed synthetic three-placeholder prompt; its dummy answer is only an input
schema requirement and is never scored.

Use the existing per-step visual encoder and the real HF processor layout over
the same decoded observations. Declare native index/FPS prompt timestamps and
report their difference from actual PTS. Compare delayed fine32 and coarse128
construction separately with independent full-native-layout region means and
bin-centre positions. Sixteen steps ensure eight committed historical steps
and eight recent fine steps; neither history capacity saturates. This therefore
does not measure retention quality or a steady-state efficiency frontier.

Freeze these gates **before the first real audit**:

- Exact embedding shapes, bit-identical embeddings and exact native positions.
  Save maximum embedding difference if exact equality fails. Different bf16
  reduction orders may fail legitimately; stop and review the arithmetic/reference
  rather than replacing this gate after seeing outputs.
- Cached versus uncached synthetic option-logit maximum absolute gap <=0.5,
  finite logits, deterministic repeated question, unchanged live state/RNG/features,
  committed boundary8, recent indices8–15, and expected actual KV length.
  The 0.5 gate is inherited from the existing short native-parity check; extending
  it to16 steps is a mechanics screen, not a long-stream numerical guarantee.
- Record actual KV tokens/bytes, recurrent/conv state, feature queue bytes,
  peak CUDA allocation, source time and processor observation/timestamp hashes.
  No accuracy, gold NLL, real-question predictions or quality confidence intervals.

One A100 GPU must have >24 GiB free. The parent enforces **900 seconds** including
reference/model hashes, GPU check, worker startup, both sources and report checks.
Reuse the reviewed process-group watchdog; stop only groups launched by this run.
A failed attempt cannot reset its deadline in the same output directory. A pass
only establishes this short real-processor mechanics check. Future quality work
still needs matched delayed-readout controls, original fully carried R32, actual
total-KV accounting, fresh-data decisions and a separately frozen protocol.

Proposed invocation from an immutable source archive on A100, after review and
A100 CPU/hosted checks (no GPU launch is part of this submission):

```bash
python -m trace_vl.delayed_history.real_audit run \
  --out /data/yyy/tracevl-runs/delayed-history-real-REV-001 \
  --gpu 0 --execute-mechanics-only
```
