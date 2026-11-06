# PR #3 final local verification — 2026-11-07

Tested scoring/runtime head: `ac3fec5b86a943e9f63ea0a7f836f56d2d21162a`.

The final code passes **87 CPU/CUDA tests** and Ruff. This round verifies engineering functionality and harness fidelity within the tested settings; it is not publication-level model-quality evidence. No merge was performed.

## Final M0 controls

- Exact public-processor token IDs: **90/90 questions**.
- Same declared scoring on both paths: BF16 backbone with FP32 option-letter projection. Agreement **90/90**; reference **51/90 correct**, native **51/90 correct**.
- Original raw BF16 remains reported: agreement **86/90**, reference **50/90**, native **52/90**. The original strict raw gate remains failed; the new result is a separately declared scoring protocol.
- Public-processor FP32 real-video gate: native/reference max **3.05e-05**, zero-gated v3/native **0**; exact tokens, pass.
- All four prior BF16 disagreement cases rerun with a full-FP32 model: **4/4 predictions agree**, max difference **2.78e-4**. Each BF16 case had a top-score tie on one side.

This is the selected first archive shard: **30/900 videos, 90/2700 questions**, no subtitles, non-thinking, option likelihoods. It is not the full Video-MME generation metric or a model-card comparison. Both sides receive identical processor pixels, timestamps and grids. Raw BF16 reference/native predictions are retained in the same final passes.

## Training and state checks (commit-labelled)

- `b538ae1`: 180 s, batch 1 / accum 1, 448², CST 4 layers / window 16, no TBPTT, noul+choice labels, GDN state read on every sample, **2 optimizer steps**, **14.446 GB** peak; finite gradients. Synthetic capacity sequence averages ~53 visual tokens/step, not a worst-case guarantee for all scenes.
- `74eb908`: 2×5090 DDP mixed-alert training, two steps, rank-0 peak **6.338 GB**; default-alert setting also passes.
- `b538ae1`: all **48 KV/GDN/conv tensors** stay unchanged after marker questions and `describe`; continued streaming exactly matches an uninterrupted control.
- `b538ae1`: real ActivityNet-source diagnostic, 40 train QA / 16 videos, 12 updates, **6.554 GB** peak. Held out: 13 QA / 6 disjoint videos, native **11/13**, reloaded v3+LoRA **12/13**. One changed answer does not establish quality gain. Video-MME test examples never enter training.

Environment: 2× RTX 5090, Torch 2.8.0+cu128, torchvision 0.23.0+cu128, Transformers 5.20.2, PEFT 0.21.2, UV. Optimized GDN kernels absent. CUDA peak values are allocated bytes / 1e9, not total board usage. Concurrent reference runs are not latency benchmarks.

## Published evidence

- `docs/reviews/pr3_verification_ac3fec5_20261107.json`: final scoring protocols, all 90 predictions, exact-token assertions, processor gate, settings and source hashes.
- `docs/reviews/pr3_verification_b538ae1_20261107.json`: serialization fix, raw-BF16 gate failure, full-FP32 disagreement controls, long training and cache-fork checks.
- `docs/reviews/pr3_followup_results_20261107.json`: earlier failures/fixes, DDP, real train metrics and complete 600-step resource curves.
- `scripts/validation/pr3_official_processor_control.py`: identical scoring on both sides, processed-patch API, raw BF16 retained, disjoint video sharding.
- `scripts/validation/pr3_alert_backward_probe.py`: independent checkpoint/autocast probe.

All are on `review/pr3-5090-validation-20261107`; model/video payloads remain local. Earlier reports deliberately retain their original tested commits and findings. No surveillance FPR/FNR, alert-delay result, full public benchmark, multi-seed ablation or architecture superiority has been established.
