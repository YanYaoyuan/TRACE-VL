# v3 experiment plan

Hardware: 2 × RTX 5090 (32 GB) or 4 × A100-40GB. All runs by the user; results posted to the GitHub
tracking issue with the template at the end. Nothing below is a result yet.

## 1. Milestones and go / no-go gates

| Milestone | Work | Gate (must hold to continue) |
|---|---|---|
| **M0** harness | load Qwen3.5-2B; streaming feature extraction; eval harness for Video-MME (short + long subsets), OVO-Bench, StreamingBench real-time, MEVA long-gap; measure tokens/s, KV MB, latency per step on 5090 and A100 | zero-shot Video-MME within ±2 pts of the model card (69.0 thinking, or the non-thinking number we measure, reported both ways) |
| **M1** plumbing | CST + TE + TKM modules, zero-init; streaming cache; parameter report | (a) gates = 0 and all I-steps ⇒ logits match the base model (max abs diff < 1e-2 bf16); (b) one training step runs on 1 × 5090 at 3-minute clips |
| **M2** Stage 1 | distillation on short clips | at ~1/5 of the tokens: ≥ 95% of the full-token teacher's accuracy on MVBench + Video-MME-short, and ≥ +2 pts over B4 (same tokens, pooled) |
| **M3** Stage 2 | streaming SFT on long streams | at equal KV budget: OVO-Bench backward and MEVA long-gap ≥ B1 (StreamingVLM recipe) + 3 pts; no loss > 1 pt on OVO real-time |
| **M4** full eval | all benchmarks, ablations, efficiency | camera/video-level bootstrap CIs exclude 0 for the main claims |

If M2 fails, fall back to v2-style external transition evidence plus CST only (still a paper on
learned evidence selection) and re-plan in the tracker.

## 2. Baselines (all on Qwen3.5-2B unless stated)

| Id | Baseline | Purpose |
|---|---|---|
| B0 | zero-shot, uniform frames at visual budgets 4K / 8K / 16K / native | backbone reference |
| B1 | StreamingVLM recipe: sink + window + long text window, contiguous positions, chunked SFT on the same data | strongest simple streaming baseline |
| B2 | training-free bounded memory: (a) DeltaS-style GDN-drift chunk eviction; (b) curvature keyframes (CurveStream-style); (c) uniform token pooling | query-agnostic compression without training |
| B3 | v2 pixel transitions + crops (external evidence, existing code) | our previous design |
| B4 | v3 with transition tokens replaced by pooled tokens at the same token count | isolates the transition encoder |
| B5 | v3 without CST (TE on raw merged tokens) | isolates the adapter |
| Ext | published numbers: StreamTTT-4B, FOLIO (Qwen3-VL-8B), StreamFlow, Tempo, Qwen3.5-4B | context only (different protocols) |

## 3. Benchmarks and metrics

| Benchmark | What it tests | Metric |
|---|---|---|
| OVO-Bench (backward tracing, real-time, forward) | streaming, questions at timestamps | official accuracy |
| StreamingBench (real-time visual) | streaming perception | official accuracy |
| Video-MME (short / medium / long, w/o subtitles) | offline long video | accuracy |
| LVBench, MLVU | hour-long video | accuracy |
| MVBench | short-clip temporal perception (regression check) | accuracy |
| TOC-Bench, HSR-Bench (StateTrace), ProcObject-10K | object state / transition consistency | official |
| MEVA long-gap (ours, `docs/inspection.md`) | placed-and-removed, duration, count; still-there control | macro balanced accuracy, camera bootstrap |
| Memory-distance law | α(D) from "How Well Can Your Video Model Remember?" | slope at D = 100 / 1000 s |

Efficiency, reported for every row: LM tokens per second of video, KV MB at 10 / 60 min, latency per
step (ms), peak training memory, and GPU-hours to train.

### Alerts (Jev-style classification head)

| Metric | Definition |
|---|---|
| step/frame AUROC and AP per noul question | e.g. "Is anything abnormal happening?" vs frame-level anomaly labels |
| choice accuracy, macro-F1, ECE | incident type; calibration of probabilities and of the confidence output |
| time to alarm | first step above threshold minus the labelled onset (or the due time for duration rules) |
| throughput | steps/s with the alert head vs asking the LM a yes/no question every step (`ask`) |

Small datasets (fit the 20–30 GB limit; check each licence): CUHK Avenue and ShanghaiTech Campus
(frame-level anomaly labels, fixed cameras), plus MEVA activities turned into noul rules.
Baselines: (a) the LM asked a yes/no question per step (autoregressive, slow); (b) v2's detector +
rule replay; (c) the alert head without latent state reads (taps only), as an ablation of the GDN-state
connection.

## 4. Data

Use only sources whose licenses allow research use; record license and version per dataset in
`configs/datasets/downloads.json`.

| Stage | Data | Notes |
|---|---|---|
| 1 | LLaVA-Video-178K subset (captions + QA, ≤ 3 min), ActivityNet / ActivityNet-QA, COIN, Something-Something v2 (state-change verbs), Charades | Something-Something and Charades are non-commercial research licenses |
| 1 | MEVA train split clips | CC BY 4.0 |
| 2 | concatenated streams (10–60 min) from the above; MEVA streams; synthetic long-gap questions generated from annotations | questions inserted at random times |
| eval | benchmarks above; no training on their test splits | |

## 5. Ablations (on Stage 1 + 2 checkpoints)

1. CST: off / temporal window K ∈ {4, 16} / relative bias off / memory read off.
2. TE surprise source: predictor error / raw feature difference / random gate / plus GDN drift.
3. Transition tokens: full (before + after + delta) / after-only / delta-only; slots M ∈ {4, 8, 16}.
4. I-step interval N_I ∈ {4, 8, 16, ∞}.
5. TKM: window only / + transition memory / retention scorer vs FIFO vs DeltaS drift; budget
   B_T ∈ {1K, 4K, 16K}.
6. Losses: −L_KD, −L_trans, −L_vis, −L_budget.
7. Zero-init vs random init of the adapters (reproduces v1's failure mode if random).

## 6. Compute plan *(est.; replace with M0 measurements)*

| Run | Hardware | Est. time |
|---|---|---|
| M0 zero-shot evals | 1 × 5090 | 1–2 days of wall clock |
| Stage 1, 40K clips, 1 epoch (teacher logits cached) | 2 × 5090 | 1–2 days |
| Stage 2, 4K long streams | 4 × A100-40GB | 2–3 days |
| Ablations (shortened schedules) | mixed | ~1 week |

## 7. Recording results on GitHub

One tracking issue holds the plan. Each run gets one comment:

```text
### <milestone> <run name> — <date>
commit: <sha>  config: <path>  hardware: <gpus>  wall clock: <h>
data: <manifest + sha256>  seed: <n>
result: <table: metric, value, CI>
memory/latency: <peak GB, ms/step, KV MB>
notes / anomalies:
artifacts: <outputs dir, checkpoint name>
```

Code changes reference the issue number in their commit messages, and design changes are posted as
comments before they are implemented.
