# Native Qwen video-input validation (2026-11-13)

This verifies the core video QA path, not alert timing or a new architecture contribution.

## Original 4B development control

Original Qwen3.5-4B, no trained adapters, identical eight decoded frames and question prompts per item, greedy decoding without a cache. The existing ActivityNet-QA-derived validation split contains 200 questions from 20 videos. This is not an official benchmark score.

| Path | Visual tokens | Normalized text exact match |
|---|---:|---:|
| Official processor + generate | 240–264 | 84/200 |
| Full-token embedding QA path | 240–264 | 84/200 |
| Fixed 4x5 grid pooling per frame pair | 80 | 83/200 |

Native and full-token generated token sequences agree on all 200 questions. Grid pooling changes 49 sequences. Exact match does not recognize synonymous answers and is not semantic QA accuracy. These numbers do not establish a training improvement.

Frames are selected by actual presentation timestamps no later than the question. The official processor derives prompt timestamps from original frame indices divided by average fps. The default 50 ms guard stopped the first complete attempt on a variable-timing video. A fresh run explicitly selected `--timestamp-policy native-index-fps`; ten questions record a discrepancy over 50 ms (maximum approximately 0.214 seconds). All arms still use the same actual images. The default strict guard remains enabled.

## Real QA training smoke checks

Two real training questions, LM LoRA rank 8 (6,193,152 trainable parameters), four updates, learning rate 1e-4, bf16, frozen vision encoder, single RTX 5090 per arm:

| Path | Mean answer CE before / after | Peak allocated GPU memory | Reload CE difference |
|---|---:|---:|---:|
| Native full tokens | 0.12322 / 0.00618 | 8.89 GiB | 0 |
| Native grid tokens | 0.13742 / 0.00773 | 8.77 GiB | 0 |

LoRA weights changed and official-forward checks passed. Both questions were already answered correctly before training; loss reduction validates optimization and save/reload only. This is not DDP validation.

## Tests and remaining work

50 core-input/QA/scoring/atomic-I/O tests passed. The selected Stage-2/memory/recent-window regression set produced 48 passes and one sandbox loopback failure; the Gloo test then passed separately with local communication permitted. Ruff passed. This is not a claim that every repository test ran.

Detailed local reports and raw predictions are retained but are not included in this commit. Next: benchmark-source exclusion, distributed QA training, same-data full/grid training baselines, and a distinct temporal architecture with controlled ablations. The original 2B versus historical Stage-1 A100 control remains pending; this local 4B check does not replace it.
