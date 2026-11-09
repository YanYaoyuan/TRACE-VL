# v3 experiment plan

Hardware: 2 × RTX 5090 (32 GB) or 4 × A100-40GB. All runs by the user; results posted to the GitHub
tracking issue with the template at the end. Nothing below is a result yet.

## 1. Milestones and go / no-go gates

| Milestone | Work | Gate (must hold to continue) |
|---|---|---|
| **M0** harness | load Qwen3.5-2B; streaming feature extraction; eval harness for Video-MME, OVO-Bench, StreamingBench, MEVA long-gap; measure tokens/s, KV MB, latency per step on 5090 and A100 | **harness fidelity to the public processor**, not the model card: (a) processor-originated `v3 check` passes (exact token ids; fp32 letter log-probs within 1e-3); (b) on the same Video-MME subset, **with fp32 letter scoring on both paths over a bf16 backbone** (`Backbone.letter_logprobs`), prediction agreement ≥ 97% and accuracy within 1/90, plus a full-fp32 rerun of every disagreement. Raw bf16-logit agreement is always reported next to it. **Status: passed at `ac3fec5`** (see §8) |
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
| B4 | v3 with only the LM-facing P-step tokens replaced by average-pooled frame regions; the slot machinery feeding CST memory and transition clocks is identical, and with `pooled_match` each P-step emits exactly as many tokens as the transition branch would (`v3_pooled_b4.yaml`, `stage1_a100_b4.yaml`). Budget gate: `v3 budget` on fixed train/val calibration videos, total visual tokens within ±5%, chosen without test accuracy. `pooled_match` equalizes budgets only for identical slots; after the arms are trained independently their slots differ, so the real-checkpoint audit (both trained checkpoints, same calibration videos) is required for every comparison; it fails closed on empty or non-train/val calibration | isolates the transition representation |
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

Small datasets (fit the 20–30 GB limit): CUHK Avenue and ShanghaiTech Campus (frame-level anomaly labels,
fixed cameras; no licence stated on either page, so research use only with citation, no redistribution), plus
MEVA activities turned into noul rules. Implemented: `v3 convert-anomaly` (video-disjoint split of the labelled
videos, normal training videos as negatives), Stage 1 `train.alert_fraction`, `v3 eval-alerts --method head|lm`
(runbook "Alert head").
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

## 8. Validation record (2×5090, 2026-11-06/07)

Reports and raw JSON are on the tester's review branch
(`review/pr3-5090-validation-20261107`): `docs/reviews/pr3_final_local_verification_20261107.md`,
`pr3_verification_ac3fec5_20261107.json`, `pr3_verification_b538ae1_20261107.json`.

| Check | Result | SHA |
|---|---|---|
| Processor-originated equality (fp32, real video) | tokens equal; native vs public 3.05e-5; zero-gated v3 vs native 0.0 | `ac3fec5` |
| Video-MME subset fidelity, fp32 letter scoring both paths | **90/90 agreement**; 51/90 vs 51/90; 90/90 token equality | `ac3fec5` |
| Same passes, raw bf16 logits | **86/90 agreement (fails the strict raw gate)**; 50/90 vs 52/90; all 4 disagreements are exact bf16 ties; full-fp32 rerun 4/4 agree (2.78e-4) | `b538ae1`, `ac3fec5` |
| Query-fork isolation | all 48 KV/GDN/conv tensors unchanged; continuation equals the uninterrupted stream | `ac3fec5` |
| Static-clip gradient regression | 14.0 (was 1.04e12) | `a761f09` |
| 180 s training, mixed alert labels, 1×5090 | 2 updates, peak 14.45 GB | `b538ae1` |
| 2-GPU DDP with the alert head | 2 updates, rank-0 peak 6.34 GB | `74eb908` |
| Alert head (untrained) cost | 1.19 ms/step vs LM 16.8 ms; isolation within 1.2e-7 | `0171f07` |
| CPU + CUDA test suite | 87 passed | `ac3fec5` |

Not established by these tests: any accuracy advantage of v3 (a 12-step ActivityNet diagnostic gave 12/13 vs
11/13 on a 13-question video-disjoint set), official benchmark scores (30 selected videos, no subtitles,
non-thinking, option likelihoods), and surveillance false-alarm/miss rates.


## 9. Paired 4×A100 pilot: 100 updates per arm

The owner-completed short-video validation run is recorded in [a100-pilot-001](results/a100-pilot-001.md), with the [supplied numerical summary](results/a100-pilot-001-summary.json). It covers 488 questions from 29 videos on the pre-gate-fix `f37d2d3` implementation. Budget matching passes, but transition accuracy is 85.25% versus native 88.32% and B4 85.66%; the paired transition-minus-B4 interval crosses zero. Rank-0 training logs are predominantly gist-only. **M2 remains open.** The report documents the procedure, provenance, memory/timing definitions, and the infeasible inherited P-step target (40 versus a maximum of 32) to address before the new-objective rerun.

## 10. Pilot-003: representation at an identical budget (predeclared before the run)

**Why.** In pilot-002 (gate-fix objective, 100 updates), transition scored 85.04%, B4 87.91% and native 88.32%. Transition − B4 = −2.87 pp [−4.55, −1.03]. The learned gate saturated at 8/8 slots (32 tokens per P-step), because the straight-through factor feeds task gradient into activity and the sigmoid is saturated at mass ≈ 4. B4, which has no straight-through path, settled at 20 tokens, so the budgets diverged by 17.8%. Pilot-003 removes budget learning entirely and asks only whether the representation helps.

**Fixed-budget control already run** (owner, separate fixed-20 driver on `7825147`, without the `after` arm; 100 updates, 488 questions / 29 videos). Native 88.32%, transition 85.45%, B4 88.11%; both arms at 46.91 tokens/step. The budget audit matched (gap 0), and all logged P-steps were 20. Transition − B4 = **−2.66 pp [−4.31, −1.02]**. The pilot-002 deficit is therefore not explained by the budget mismatch: at equal budget the trio representation is still worse than grid pools. Pilot-003 keeps the transition/B4 pair as a replication and adds the `after` arm. That arm is now the main open question: are real-token slot pools at least as good as grid pools?

**Design** (`configs/v3/pilot003_*.yaml`, `scripts/run_pilot003.sh`). Same data, splits, seed, schedule and hardware as pilots 001/002: 100 updates, 4×A100, batch 1 per rank, clips ≤ 30 s.
- `fixed_active: 4`: exactly the top-4 slots by mass on every P-step, in training and inference, with no sampling and no straight-through. Every P-step therefore has gist 8 + 12 = 20 tokens.
- Scene-cut I-steps are disabled (`scene_cut_surprise: 1.01`), so I-steps follow only the fixed every-8 schedule. This makes tokens per step identical across independently trained arms; the script verifies it (spread ≤ 0.5%).
- Arms differ only in `p_step_mode`:

  | Arm | LM-facing P-step tokens (after the gist) |
  |---|---|
  | `transition` | before / after / delta per slot |
  | `after` | per slot, 3 real-token pools over equal-weight parts of the slot's region |
  | `b4` | 12 average-pooled grid regions |

  All three share the same slot machinery, CST memory and centroids.
- Evaluation: native plus the three arms on the same 488 validation questions; paired whole-video bootstrap (4,000 replicates, seed 7).

**Decision rule** (fixed now; no post-hoc arm selection):
1. **M2 (primary, transition arm only):** transition ≥ B4 + 2 pp and transition ≥ 95% of native. If transition misses this, M2 fails, even if `after` passes.
2. **Diagnostics.** A difference is called only when the 95% CI excludes 0.
   - `after` − `b4` tests whether surprise-based slot selection helps over a fixed grid.
   - `transition` − `after` tests whether the before/delta tokens help over real-token pools.
3. **Consequences:**

   | Outcome | Next step |
   |---|---|
   | `after` > `b4` and `transition` < `after` | Redesign the transition tokens around real-token pools (e.g. after-pools plus a learned difference channel kept inside the LM's visual distribution). |
   | `after` ≈ `b4` | Surprise selection gives no benefit at this budget on 0–30 s multiple choice. Test on transition-sensitive benchmarks (OVO-Bench forward, TOC-Bench) before continuing. |
   | `transition` ≥ `after` ≥ `b4` | Proceed to longer training (3,000 updates) with the fixed budget. |
4. **Out of scope:** the CST gate (`gate_abs_mean` ≈ 4.5e-4 after 100 updates) is reported, not changed, in this pilot.

**Runner safeguards** (PR #3 review of `cf92794`):
- An arm counts as trained only through the trainer's atomic `complete.json`. It is written after the final module/LoRA save and pins the step, the resolved config and the artifact hashes; the metric log must hold exactly steps 1..N. Incomplete attempts are moved aside and retrained.
- The run identity (`identity.json`) holds the fully resolved per-arm configs, the record hashes, the media identity and the source hash. Any change on restart is rejected.

**Limits:** one seed, 100 updates, 0–30 s YouTube multiple choice only. A null result here does not rule out benefits on long or transition-sensitive video.

### 10.1 Pilot-003 result (operator run at `cc00d60`, 100 updates, 488 questions / 29 videos) — negative

| Arm | Accuracy | Tokens/step | KV MiB |
|---|---:|---:|---:|
| native | 88.32% | 196.3 | 41.2 |
| transition | 85.45% | 46.91 | 11.1 |
| after | 85.45% | 46.91 | 11.1 |
| b4 | 88.11% | 46.91 | 11.1 |

The budget spread is 0, and every logged P-step had 20 tokens. Paired differences, in pp:

| Comparison | Δ | 95% CI |
|---|---:|---:|
| transition − b4 | −2.66 | [−4.48, −1.02] |
| after − b4 | −2.66 | [−4.47, −1.03] |
| transition − after | 0.00 | [−1.23, 1.22] |

**M2 fails** (the +2 pp condition). Both slot arms lose to grid pools at the same budget, and before/delta show no detectable effect over real-token pools.

One observation to check: transition, after, and the earlier fixed-20 transition run all score exactly 85.45%. The slot arms share 8 learned gist tokens plus 12 slot tokens, while B4 sends 20 grid pools and no gist. If the LM barely uses the 12 slot tokens, the gap would be gist versus grid coverage, not slot representation. The final CST gate is |g| ≈ 4.5e-4 in all arms.

### 10.2 Diagnostics (`scripts/diagnose_pilot003.sh`, before any larger run)

1. **Provenance.** Completion records verify; the arms' checkpoints are distinct; the evaluated checkpoint and adapter are the trained ones.
2. **Per-question agreement** between all arm pairs: discordant counts, identical-answer rate, exact McNemar test.
3. **Slot-token ablation.** The trained transition and after checkpoints are evaluated with gist-only P-steps (`emit_slots: false`). If gist-only ≈ full (CI around 0, high agreement), the LM does not use the slot tokens.
4. **Optimization health.**
   - Training curves (first and last 20 updates), and validation gold-letter NLL from the logged letter log-probs.
   - Per-group relative weight change against an exactly replicated initialization (the replica is validated by the untouched alert/retention parameters).
   - LoRA B norms and CST gate statistics.
5. **Activation probe** on 8 validation videos:
   - CST gated-residual / input norm;
   - effective spatial coverage of the tokens the LM sees on P-steps (1 = uniform grid);
   - gist-attention entropy and slot coverage.

   LLaVA multiple-choice data has no region labels, so coverage stands in for evidence retention.

### 10.3 Follow-up, chosen by rule from the diagnostics (one change; same data, seed, schedule and budget)

| Diagnostic outcome | Pilot-004 (single change) | Acceptance criterion |
|---|---|---|
| Optimization unhealthy: CST or slot branch weights ≈ init, LoRA B ≈ 0, or loss still falling steeply at update 100 | Same 3 arms, 1,000 updates (only the schedule changes) | Any slot arm ≥ b4 with the CI lower bound > −1 pp. Otherwise undertraining is ruled out as the explanation. |
| Healthy, and gist-only ≈ full (slot tokens unused) or coverage(slot arms) ≪ coverage(b4) | `after` arm with the 8 learned gist tokens replaced by 8 grid pools (`gist_mode: grid`), against b4 (20 grid pools) | ≥ b4 + 2 pp: surprise-selected tokens add value over grid tokens. CI excluding 0 below: selection is harmful here; stop slot-token work on short clips and move the test to transition-sensitive benchmarks (OVO-Bench forward, TOC-Bench). |
| Healthy, and slot tokens used (gist-only clearly worse than full) | Same 3 arms on a transition-sensitive set (OVO-Bench forward / TOC-Bench), no retraining | Slot arm ≥ b4 + 2 pp on that set |

The CST gate stays as it is in pilot-004. If the probe shows a negligible residual ratio (< 1e-2), opening it (a separate learning rate for the gate) is a later, separate experiment.
