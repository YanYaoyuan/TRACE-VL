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

### 10.4 Diagnostics outcome (operator run, code `7dc2ce9`; qualified after review)

| Finding | Evidence |
|---|---|
| Provenance holds | Completion records verify, arm checkpoints are distinct, and the evaluated checkpoints are the trained ones; the initialization replica is exact |
| **Gist pooling is near-uniform** (strong evidence) | Normalized gist-attention entropy is 1.0000 (min 0.99999958) on 109 probed P-steps, on 8 videos |
| Slot pooling: **not established** | Union coverage (0.995–0.999) cannot distinguish uniform individual slots from complementary pools. The `after` part-pools can differ even when the parent slot is uniform, and before/delta tokens carry temporal, projection and type information. "The slot arms sent copies of one global average" is a **hypothesis** to measure with per-slot entropies and emitted-token redundancy (§11), not a finding. |
| Slot-token ablation | Removing the slot tokens changed few answers: transition full − gist-only = +0.41 pp [−0.41, +1.23], 99.2% agreement; after = −0.82 pp [−1.64, −0.20], 98.4% agreement. The ablation also changes token count and sequence structure, so this is sensitivity to that intervention, not an equal-budget test or proof that the slots are unused. |
| CST residual / input norm ≈ 0.63% | Small, but **not** shown to have no effect: CST's latent z also feeds the gist, slots and surprise independently of the gated residual |
| Optimization signals | CE falls, LoRA changes and gradients are finite. These do not establish healthy optimization in every selector sub-module. About 2% relative parameter movement is not by itself evidence of undertraining (there is no universal threshold). Zero-initialized groups (`cst.gate`, `type_embed`) have an undefined relative change and did train (abs change 0.028 / 0.054 for transition); the earlier digest printed them as 0, which was a formatting bug, now fixed. |

The video-cluster bootstrap is the primary uncertainty estimate; question-level McNemar treats questions within a video as independent. **Pilot-003 stands as a valid negative test of the implemented architecture and training recipe.**

## 11. Pilot-004: training duration only (predeclared)

**Question.** Does extending *this* recipe from 100 to 1,000 updates change the matched-budget comparison? A negative result shows that extending this recipe did not close the gap. It does not rule out undertraining in general.

**Design** (`configs/v3/pilot004_*.yaml`, `scripts/run_pilot004.sh`):
- The three pilot-003 arms (transition / after / b4) with the same data, seed, initialization, optimizer settings, warmup 10, fixed 20-token P-steps, I-step schedule and hardware.
- **Fresh training** for 1,000 updates, not a continuation. The cosine schedule spans 1,000 updates, so the first 100 updates do **not** follow pilot-003's learning-rate trajectory.
- Predeclared evaluation points: checkpoints at **100 / 300 / 1,000**, all on the same 488 validation questions. **The final checkpoint is the primary comparison**; 100 and 300 are trajectory points and are never selected as "best".
- Per-question budget identity across arms is checked at every checkpoint.

**Instrumentation (logging only; outputs are bit-identical with it on or off, which is tested).** Every 10 updates:
- per-gist minimum entropy, gist-logit spread across positions, gist-token cosine;
- per-slot entropy (mean and minimum over active slots) and slot-attention spread;
- `after` part-pool entropies;
- cosine similarity of the slot-token embeddings;
- cosine similarity and effective rank of the **emitted** P-step tokens;
- post-clipping gradient norms and actual update norms for the gist, slots, surprise gate, CST gate and CST layers.

The output is `selector-diagnostics.json`.

**Decision:**
1. **M2 gate (unchanged):** transition ≥ b4 + 2 pp and ≥ 95% of native, at the final checkpoint.
2. **Exploratory continuation criterion** (not an architecture gate): a slot arm ≥ b4 with the video-bootstrap CI lower bound > −1 pp at the final checkpoint.
3. If the slot arms still lose, report the outcome under this recipe and use the selector diagnostics (per-token entropies, emitted-token redundancy, selector update norms) to choose **one** subsequent change. No outcome is precommitted.

### 11.1 Pilot-004 result (operator, `a3f8c29`; report in `docs/v3/results/pilot004-a3f8c29/`): negative against the gate

| Checkpoint | Native | Transition | After | B4 |
|---|---:|---:|---:|---:|
| 100 | 88.32% | 85.66% | 85.45% | 88.32% |
| 300 | 88.32% | 87.09% | 88.93% | 88.52% |
| **1,000 (primary)** | 88.32% | 87.91% | 89.14% | **89.96%** |

Paired differences at the final checkpoint (pp):

| Comparison | Δ | 95% CI |
|---|---:|---:|
| transition − b4 | −2.05 | [−4.32, 0.00] |
| after − b4 | −0.82 | [−3.05, +1.43] |
| transition − after | −1.23 | [−3.28, +0.62] |

- **M2 fails**, and both exploratory continuation checks fail.
- All arms improved with longer training; this is not evidence of an architectural benefit.

Selector diagnostics (rank-0 training samples; a fixed held-out probe is pending):
- **Learned gist: demonstrated non-functional.**
  - Entropy is 1.0000 from update 1 to update 1,000 in both slot arms.
  - Gist-logit std across positions is 0.002–0.006.
  - The 8 gist tokens are near-identical (cosine 0.9994).
  - Gist gradients fall to about 1e-5.
  - Cause: logits start near zero (queries ~ N(0, 0.02²)), so the softmax over 196 positions is uniform and receives almost no gradient.
- **Slots:**
  - transition: entropy 0.996, emitted effective rank 1.86;
  - after: entropy 0.91, minimum 0.82, part-pool entropy 0.68, rank 2.47.
- **Measurement fix:** the training-time B4 rank (12.9) was measured before norm matching. This is fixed in `bfd5490`, and `scripts/launch_probe_pilot004_a100.sh` re-measures every arm on identical held-out clips.

### 12. Pilot-005 (APPROVED 2026-11-09, all three arms; launch after the held-out probe and local verification): replace the non-functional learned gist with grid pools

**Rule applied (data_plan §4, row C).** The slot arms lose, and a selector issue is demonstrated (the gist above). One change addresses it.

**Change.** `transition.gist_mode: grid`: the 8 gist tokens become 8 exact grid-region average pools (the same construction as B4's), and nothing else changes.
- The 12 slot or part tokens stay, so the budget is still 20 per P-step.
- Same data, seed, 1,000 updates, checkpoints, diagnostics and evaluation as pilot-004.
- The flag has no effect on B4, which does not emit a gist. B4 is **retrained** at the same revision as the matched control, which also measures run-to-run reproducibility against pilot-004's B4 (89.96%).

**Question.** With identical grid coverage for 8 of 20 tokens, does replacing B4's other 12 grid pools with 12 surprise-selected slot tokens help, hurt or make no detectable difference?

**Acceptance (predeclared):**
- **M2 (unchanged):** transition ≥ b4 + 2 pp and ≥ 95% of native, at the final checkpoint.
- **Exploratory continuation:** after ≥ b4 with CI lower bound > −1 pp.
- **Superiority:** claimed only if the CI lower bound is > 0.
- Also reported:
  - the held-out-probe emitted-token rank of the slot arms (expected ≥ 8 from the grid gist alone);
  - |B4(005) − B4(004)| as a reproducibility estimate.

**Cost.** About 4 h of training (3 arms × ≈ 76 min) plus about 1 h of evaluation on 4×A100.

**If it fails.** Report the result, and stop slot-token work on 0–30 s multiple choice. The next step becomes evaluating the existing checkpoints on a predeclared temporal subset (data_plan §4, row D / L1).

### 12.1 Pilot-005 result (operator, `6a46fb5`; tester review on issue #4): negative, failure branch taken

Final checkpoint (primary):

| Arm | Accuracy |
|---|---:|
| native | 88.32% |
| transition-grid | 88.12% |
| after-grid | 88.73% |
| b4 | **90.37%** |

| Comparison | Δ (pp) | 95% CI |
|---|---:|---:|
| transition − b4 | −2.25 | [−4.29, −0.21] |
| after − b4 | −1.64 | [−3.31, 0.00] |

- M2 and the exploratory continuation criterion both fail.
- The grid gist raised emitted effective rank (transition 5.3, after 6.0; b4 13.3) but did not close the gap. Transition slots stay near-uniform (entropy 0.996).
- Held-out probe of pilot-004 (16 fixed videos): emitted cosine and rank were transition 0.976 / 1.85, after 0.919 / 2.33, b4 0.743 / 12.98. CST residual / input was about 5%.

**Per §12, iterative slot-token tuning on 0–30 s multiple choice stops here. B4 is the working compression baseline.**

## 13. Temporal diagnostic (evaluation only, predeclared before any model is run)

Do transition tokens help where temporal change matters? This uses existing pilot-005 checkpoints only (`scripts/run_temporal_diagnostic.sh`, `trace_vl.v3.temporal`).

**Data:**
- OVO-Bench, an independent benchmark never used for training or for earlier decisions.
- Videos are hashed (seed 7) into a **diagnostic half**, used once here, and a **frozen final half** that is not evaluated before the paper. The repeatedly inspected pilot validation set is not used.

**Categories, by task definition** (never by model errors):

| Category | Tasks |
|---|---|
| temporal | ASI, ACR, FPD, REC, SSR |
| static | OJR, ATR, OCR, STU |
| not evaluated | EPM, HLD, CRR |

- Questions are limited to query time ≤ 180 s, at most 100 per task (fixed seed).
- Arms: native plus the transition / after / b4 final checkpoints of pilot-005, with identical budgets.

**Decision:**
- **Continue the transition-token track** only if transition − b4 on temporal is ≥ +2 pp with CI lower bound > 0, **and** the interaction (Δtemporal − Δstatic) has CI lower bound > 0. Both use a video-cluster bootstrap.
- **Otherwise stop transition-token work.** The paper follows outcome C (B4 compression, learned memory retention, alert head; transition tokens reported as a negative ablation).
- after − b4 is reported with the same statistics; it is secondary and does not gate.
- One evaluation; no re-selection and no retraining.

### 13.1 Temporal diagnostic result (operator, `9653e2a`, 2026-11-09): stopping rule fails, transition track closed

OVO-Bench diagnostic half: 475 questions / 200 videos (315 temporal, 160 static); the frozen final half (863 / 192) was not evaluated.

| Arm | Temporal | Static | LM tokens/step |
|---|---:|---:|---:|
| native | 49.52% | 59.38% | 190.2 |
| transition-grid | 47.30% | 51.88% | 43.8 |
| after-grid | 45.71% | 55.00% | 43.8 |
| b4 | 47.30% | 60.63% | 43.8 |

transition − b4 (pp):

| Category | Δ | 95% CI |
|---|---:|---:|
| temporal | 0.00 | [−4.48, +4.58] |
| static | −8.75 | [−14.71, −3.45] |
| interaction | +8.75 | [+1.81, +16.20] |

The interaction comes from worse static accuracy, not from a temporal gain. **`continue_transition_track = false`.** Pilots 002–005 and this diagnostic are kept as negative evidence for this implementation and recipe; this is not a universal claim against transition representations. **B4 is the compression baseline. The paper follows outcome C.** The token reduction (−77% vs native) is a count, not a measured latency, FLOP or memory saving; B4 is not claimed lossless or superior to native.

## 14. Stage-2 memory pilot: learned retention vs heuristics at equal budget (predeclared)

**Question:** with the B4 representation and a real memory budget, does a **learned retention policy over age and surprise statistics** keep the right evidence better than surprise mass or FIFO?

**Scope:** in pooled B4 mode, retention groups are whole steps. Of the scorer's six inputs, drift is 0, hits is 1 and the slot/step flags are fixed, so effectively **age and surprise mass** vary. Queried and distractor clips are assigned at random, so semantic relevance is not observable from these inputs. This is **not** content-aware semantic memory, and results are interpreted within that feature set. No semantic features are added in this experiment.

**Data** (`scripts/run_memory_pilot.sh`, `trace_vl.v3.memory_pilot`):
- MC questions of the already-extracted LLaVA subset.
- **Every video in the Stage 1 train/val manifests is excluded.**
- Splits are frozen at **source-video level before stitching**: train / dev / test 80/10/10 by hash. Pilot caps: 3,000 / 400 / 400 videos.
- Streams are about 300 s, stitched within each split:
  - 40% distractor clips (never asked about);
  - delayed questions 45–240 s after their clip (beyond the 16 s window);
  - 20% immediate questions (within-window control);
  - each question names its interval.
- **dev** is used for this pilot. **test** is frozen. The OVO final half is not used.

**Model and budget:**
- Stage 2 from pilot-005's B4 final checkpoint (`memory_pilot_b4.yaml`), 300 updates on 4×A100.
- Memory: a **16 s window + 1,024 older tokens**. B4 emits about 13k tokens per 300 s, so most must be evicted.
- Truncated BPTT (chunks of 8 steps, also cut at question times) and retention BCE against later-question evidence windows. Labels come from annotations at training time only; the scorer's features (age, mass, hits, slot/step) contain no question information at evaluation.

**Evaluation** (dev, 40 streams):
- **The same Stage-2 checkpoint under policies tkm / mass / fifo / full.** Only the policy changes.
- **tkm with `evict_schedule: train`** (`s2_tkm_trainsched`): evicts after exactly the steps where Stage-2 training evicts. Both use `eviction_steps()` from the same `chunk_plan`, including question-time cuts and the final partial chunk. `s2_tkm − s2_tkm_trainsched` measures the effect of evicting every step (inference) versus the training schedule. Parity of retained memory follows from the shared schedule and identical scoring at those steps, which is tested at the schedule level; there is no model-level replay test.
- Separately, **the Stage-1 B4 checkpoint under mass / fifo / full** (does Stage-2 training help?).
- Reported per run:
  - accuracy (all, beyond-window, within-window) with a stream-cluster bootstrap;
  - evidence kept;
  - retained KV tokens and MB;
  - evicted / streamed tokens;
  - evidence age;
  - seconds per stream.

**Stream selection** (protocol `memory-pilot-v2`; metadata only, fixed before any model is evaluated, never using accuracy). The selected manifests `dev_selected.jsonl` and `train_selected.jsonl` are reused by every run, and `data/selection.json` records:
- candidates, eligible streams and exclusions by reason;
- selected IDs with their actual ages and durations;
- seed and rule version.

| Mode | Rule | Selection |
|---|---|---|
| **Smoke** | `smoke-eligible-hash-v1` | Candidate pool of 600/200/200 source videos (configurable). A stream is eligible if its duration is ≥ 240 s **and** it has ≥ 1 *actual* beyond-window question (evaluator due-time rules; nominal delays clipped at the stream end do not count). Eligible streams are ordered by sha256(seed 7 : id). The first two dev streams are evaluated, and all eligible training streams are sampled. |
| **Pilot** | `pilot-first-n-v1` | Unchanged population: the first 40 dev streams in file order, and all training streams. |

- If fewer than two dev streams or fewer than two training streams are eligible, the run stops before training and reports the counts and reasons.
- The frozen test split is never used to fill dev, and no thresholds are lowered.
- Smoke and pilot compatibility is checked on protocol, revision, config, initialization and data sources; sizes and the selection rule are mode-specific by design.
- First A100 smoke attempt (`ac5fa12`, caps 60/20/20): correctly stopped by the data gate (0 beyond-window dev questions in the first two streams). It is preserved for provenance.

**A100 smoke at `77c5050`: passed** (2 updates, 2 dev streams).
- All six checks true; 37 actual beyond-window questions.
- Retention gradient norms 0.21 / 0.25; 2,316 positive / 1,412 negative labels.
- Retained-KV spread 0.98%.
- The formal decision was `invalid` (37 < 100 beyond-window questions), which is expected for a plumbing smoke. The smoke results are descriptive only.

**Formal pilot attempt at `77c5050`:** stopped by the data gate before training. The 400 dev videos produced 25 streams, below the required 30. Preserved for provenance.

**Data-preparation change (dev-only expansion, `TRACE_REUSE_DATA`):**
- The failed pilot's **train and frozen test manifests and streams are reused byte for byte** (hash-verified copies).
- The dev split is rebuilt from **all videos the same frozen split assigns to dev** (same seed and fractions, Stage-1 exclusions), ordered by sha256(`7:dev:<video>`). This is a prefix rule, `TRACE_PILOT_DEV_CAP` optional.
- Train and test are never resampled; `make_splits`' sequential RNG would change test if the dev cap changed.
- `data/prep.json` records the old and new caps, the available dev videos, the reused-file hashes and the old and new dev-stream hashes.
- The model, training settings, first-40 dev rule, ≥ 30-stream and ≥ 100-question gates, and the gain and budget criteria are unchanged.
- A matching smoke is rerun at the new revision; the revision check is not bypassed.

**Token accounting:** evicted and appended counts are both KV tokens (text + visual), reported as `KV evicted X of Y appended`. Visual tokens streamed are reported separately; the old "evicted of streamed" ratio mixed the two.

**Data sufficiency is checked before any GPU time** (`data/report.json`): source videos after exclusion, per-split counts, stream counts and durations, and **actual** evidence ages of the questions the evaluator will ask. A nominal 45–240 s delay is clipped at the stream end, so the actual age is used. Pilot thresholds: ≥ 50 training streams, ≥ 30 selected dev streams, ≥ 100 actual beyond-window dev questions. Otherwise the run stops.

**Decision (three states, `memory_report`):**
- **invalid / inconclusive:** a validity prerequisite fails. That covers missing or incomplete runs against the expected question set, < 100 beyond-window questions, no evictions, FIFO keeping > 80% of the evidence, or unequal retained KV. No conclusion about retention is drawn.
- **negative:** valid, but the gain rule fails. Learned retention over age/surprise statistics is not a contribution; do not scale Stage 2 for it.
- **supported:** valid **and** s2_tkm − s2_mass and s2_tkm − s2_fifo on beyond-window questions are each ≥ +3 pp with a stream-bootstrap CI lower bound > 0. Validity also requires:
  - **memory pressure**: evictions occur, FIFO keeps ≤ 80% of the evidence, and ≥ 100 beyond-window questions;
  - **equal retained KV** (within 2%).
- Otherwise retention is not a contribution: report the heuristics and do not scale Stage 2 for retention.

**Smoke mode** comes first: 2 updates and 2 dev streams per run. It writes a machine-checked `smoke-ok.json` recording the checks plus the revision, resolved config, initial checkpoint identity and data sources. The checks are:
- finite, nonzero retention **gradients**, not just updates (weight decay alone moves weights);
- both retention label classes present;
- evictions occurred;
- all eight expected named runs complete with the expected question set;
- enough smoke questions.

The pilot verifies this record (passed, and matching revision, config, initialization and data sources; only sizes may differ) before doing anything. A readable digest is not an authorization.

**Training reuse** requires the trainer's atomic completion record, written after both final saves. It must match the resolved config, the step count, the contiguous metrics, the artifact hashes and the run identity (source revision, training-stream hash, stream media identity, Stage-1 module and adapter identity). Otherwise the attempt is kept as `train.incomplete-*` and retrained. Changed inputs for an existing output directory are rejected.

**Limits:** truncated gradients across chunks; chunk-end eviction in training (quantified by the evict-8 run); one seed; clip-level evidence windows. Alerts are a separate experiment.

### 14.1 Memory pilot result (`448adf1`, 300 updates; tester analysis on issue #4) and evaluation-only budget repair

**Data:**
- 802 dev questions, of which 571 are actual beyond-window questions across 37 streams; median evidence age 68.6 s.
- 954 dev-assigned videos produced 55 streams; the first 40 were used.
- Train and test data were reused.

**Status: `invalid`.**
- Retained-KV spread across policies was 2.14% (mass vs FIFO), against the 2% limit, which is not relaxed.
- Descriptively, TKM shows no delayed-QA gain over FIFO: 73.9% vs 74.1%, with a 0.15% KV difference; TKM − FIFO = −0.18 pp [−0.57, 0.00].
- Stage-2 training shows promising **descriptive** S2 − S1 gains under fixed policies (FIFO +4.7 pp beyond-window). These need paired intervals.
- **No learned-memory contribution is established.**

**Cause (from code):** `keep_mask` admitted whole groups of varying size under the 1,024-token older budget and skipped groups that did not fit, so different rankings leave different unused capacity.

**One predeclared, accuracy-independent correction:** `memory.fill: exact` (`configs/v3/memory_pilot_b4_exactfill.yaml`). Whole groups are admitted in rank order, then the newest tokens of the next-ranked group fill the budget exactly. Every policy therefore retains min(budget, older tokens), which is identical across policies at every step: the window and protected tokens depend only on time, and all policies append the same tokens. The correction applies to all compared policies.

**Evaluation-only repair run** (`scripts/run_memory_eval_repair.sh`):
- **Inputs:** the frozen S1/S2 checkpoints (completion record, Stage-1 identity verified) and the identical 40-stream manifest (hash recorded). The config is identical except `memory.fill`. Training used `groups`; this mismatch is recorded and does not retroactively validate the `448adf1` run.
- **Steps:**
  1. Paired S2 − S1 (FIFO / mass / full) and per-question TKM vs FIFO/mass agreement, from the **existing** predictions.
  2. Budget **pre-check** on the 2 shortest + 2 longest manifest streams; the complete evaluation runs only if the KV spread is ≤ 2%.
  3. All eight evaluations with the **memory audit**:
     - KV split (protected, window, older, unused older capacity, older groups) at every question;
     - Jaccard overlap of the older tokens TKM/mass keep vs a FIFO counterfactual on the same memory state;
     - Spearman correlation of retention scores with recency and with surprise mass at every eviction. This tests whether the scorer effectively ranks by recency.
- **Decision rule unchanged**; the full policy remains the unbounded reference. The frozen test split is untouched; no new data or training.


### 14.2 Repair result (operator run `memory-evalrepair-af2f961-001`; tester verification on issue #4, 2026-11-10): **negative** for the current TKM; positive paired evidence for Stage-2 training

The tester recomputed every statistic independently (separate implementation; 4,000 stream-cluster bootstrap draws, seed 7). All 14 paired deltas and CIs in `memory-report.json` and `analysis.json` reproduce exactly.

**Validity:**
- All eight arms have the same 802 question IDs on the same 40 streams; 571 questions are beyond the 16 s window.
- Total KV is identical **for every question** across the five restricted every-step arms (S1 FIFO/mass, S2 FIFO/mass/TKM), mean 1891.71 tokens.
- All validity checks pass. The gain criterion fails. **Status: `negative`.** The `448adf1` group-packing run stays `invalid`; this result does not change it.

| Comparison (paired, stream-cluster bootstrap) | Beyond-window | All questions |
|---|---|---|
| S2 FIFO − S1 FIFO | **+4.90 pp [2.46, 7.30]** (424 vs 396 / 571) | **+3.87 pp [1.83, 5.89]** (626 vs 595 / 802) |
| S2 full − S1 full | +2.98 pp [0.17, 5.85] | +2.87 pp [0.60, 5.19] |
| S2 mass − S1 mass | +1.58 pp [−2.17, 5.21] (not established) | — |
| S2 TKM − S2 FIFO | −0.18 pp [−0.57, 0] (423 vs 424) | 0 (626 vs 626) |
| S2 TKM − S2 full | −8.58 pp [−11.73, −5.52] | — |

- TKM and FIFO correctness differs on 2 of 802 questions, one each way. The files log correctness, not the chosen letter, so this is correctness agreement, not proof of identical answers.
- The scorer behaves like recency. Score/recency Spearman 0.927. Same-state counterfactual Jaccard of kept older tokens vs FIFO 0.964. Both are averages of cumulative session summaries taken at question times. They do not compare two independently evolved trajectories.
- FIFO keeps about 84.2% less measured KV than full history. This is KV-cache storage only: not total GPU memory, and no latency speedup is shown.

**Decision:**
1. **The age/surprise-only TKM contribution claim is closed** for this recipe. No more updates are spent trying to make it beat FIFO. No further 300-update TKM run.
2. **The Stage-2 checkpoint and B4 + FIFO are the working baseline.** S2 − S1 under FIFO is paired, statistically supported evidence for the Stage-2-trained checkpoint on this dev benchmark. It does not isolate a loss or module (Stage 2 changes the training data, the BPTT schedule and the retention head together). It is not yet evidence of external generalization or paper novelty.
3. **Preserved:** the S2 checkpoint (`memory-pilot-448adf1-001/train`), all prediction files of both runs, and both experiment records (§14.1, §14.2).

**Audit reporting issue (fixed in code, no rerun needed for the decision):** the evalrepair rows show `kv_older` > 1024 in 783/802 TKM rows (mean 1075.97), despite the exact cap. Two clocks were mixed:
- eviction and the cap use the consumed step time;
- `audit_summary(q["time"])` reclassified recent/older tokens at the later question time, so window tokens that aged out after the last eviction counted as older.

The total-KV match and the paired accuracy result are unaffected. The audit rows now carry three labelled clocks:
- `kv_evict_*` and `audit_evict_t`: the split right after the latest eviction, aged at that time. This is where the cap applies. `audit_older_max_at_eviction` and `audit_older_cap_violations` cover every eviction event.
- `kv_step_*` and `audit_step_t`: aged at the last consumed step.
- `kv_qtime_*`: aged at question time. Legacy `kv_*` fields are read as question-time.

Aggregates are labelled too. `cumulative_at_question` is an average of cumulative summaries at question times. `event_weighted` pools sums and counts over eviction events per stream. `memory_analysis` reads the af2f961 predictions unchanged. Regression tests: `test_audit_checks_the_cap_at_eviction_time_and_labels_question_time_ages` and `test_audit_aggregate_separates_clocks_and_weights_events`.

### 14.3 Next bounded milestone: does the Stage-2 gain transfer? (endpoint corrected after tester review; evaluation only)

**Correction to the first draft.** The §13 manifest (475 q / 200 videos) is **not** a backward + real-time MC set:
- `temporal.py` excluded EPM/HLD/CRR and included the forward tasks REC/SSR.
- Its only backward task is ASI (51 questions).
- OVO-Bench (our `OVO_MCQ_TASKS` and the official repository) defines backward as **EPM/ASI/HLD**.

**Population: `ovo-transfer-dev-v1`** (`src/trace_vl/v3/ovo_transfer.py`). This is a new, explicitly named manifest, not the 475-question one.
- **Videos:** exactly the 200 videos of the §13 diagnostic manifest. Disjointness from the frozen final half is checked on video IDs: no overlap with `final_frozen.jsonl`, and every video hashes to the diagnostic half.
- **Questions:** every official MC question on those videos, re-converted from the official annotation. The conversion must equal the §13 conversion record for record. Included are the backward tasks EPM/ASI/HLD and the six real-time tasks STU/OJR/ATR/ACR/OCR/FPD, with query time ≤ 600 s (the Stage-2 maximum).
  - This expands the question population, and the expansion is disclosed.
  - Eligibility uses data and protocol only. Every exclusion is counted: duplicate ID, missing time, invalid options/answer, query time > 600 s, video unreadable, query after the video end, no complete step before the query.
- **Status:** the videos are external to LLaVA training but already informed the §13 decision. Results are reported as an **external-dataset development evaluation**, not a held-out confirmation. Final confirmation stays separate. The frozen final half is untouched.
- **Adequacy, fixed before GPU work:** at least 150 backward questions, at least 30 backward videos, and no video holding more than 10% of backward questions. Below this, the inventory is reported and the GPU stages refuse to run unless the owner accepts it (`TRACE_ACCEPT_INVENTORY=1`). An underpowered run is not evidence against transfer.

**Protocol:**
- **Arms:** frozen S1 (pilot-005 B4) and S2 (`448adf1`) × FIFO/full. Exact fill, §14 budget and config.
- **Streaming:** one stream per video, cut at its latest eligible query time. Each video is streamed once per arm, with identical step times in every arm.
- **No future frames:** each question is answered after the last step whose two-frame window ends at or before its query time.
- **Fork isolation:** the question runs on a fork. `evaluate_streams` now stops if the live state (memory length, step index, KV bytes, recurrent-state sum) changes across a question.
- **Each prediction row records** `choice`, `t_step` and `step_window_end`.
- **Mandatory coverage checks:** every arm has exactly the manifest's question IDs once each, matching stream and time, window end ≤ query time, and an identical step schedule across arms.
- OVO has no evidence spans, so backward results are **never** relabelled "evidence older than 16 s".

**Predeclared rule** (video-cluster bootstrap, 4,000 draws, seed 7, the same percentile convention as §14):
- **Primary:** S2-FIFO − S1-FIFO on **pooled EPM/ASI/HLD, question-weighted accuracy**. `continue` if Δ ≥ +2 pp and the CI lower bound is > 0.
- **Also reported, no gate:**
  - each backward task, plus a backward task-macro Δ with its own CI;
  - all questions;
  - real-time, with a no-harm check (CI lower bound > −2 pp);
  - S2-full − S1-full;
  - S2-FIFO − S2-full.
- **Statuses:**
  - `invalid_coverage`: missing or invalid coverage; no conclusion.
  - `threshold_not_met_inconclusive`: the CI still includes +2 pp.
  - `threshold_not_met`: the CI excludes +2 pp on this population.
  - `continue`.
- A failed threshold does not establish that Stage-2 gains can never transfer.
- No retraining or scaling is authorized by this milestone.

**Stages** (`scripts/run_ovo_transfer.sh`, pinned launcher `scripts/launch_ovo_transfer_a100.sh`):
1. `inventory`: CPU only. Split-hash check, re-conversion check, manifest and inventory. Submitted before any GPU work.
2. `flow`: S1/S2 FIFO on the 2 shortest + 2 longest streams. Checks coverage, no future frames and identical schedules, and estimates wall time.
3. `full`: the four arms, the report and the decision.

Provenance (`inputs.json`): manifest and question hashes, the resolved config hash, the S1/S2 checkpoint identities (the Stage-2 completion record re-verified), and the hashes of the af2f961 dev baseline predictions.

**14.3a Inventory v1 failed adequacy (A100, `e90d586`, CPU only; operator screenshot, 2026-11-11).**
- **Conversion:** 1,468 MC records on 512 videos, none missing. 200 population videos, 0 overlap with the frozen final half, 0 unreadable.
- **v1 population:** 386 in scope. Exclusions: 14 with query time > 600 s, 1 with no complete step before the query. 371 eligible on 147 streams.
- **Backward: 51 questions (ASI only, 51 videos); EPM 0, HLD 0.** Real-time: 320 questions on 96 videos.
- **Adequacy false** (51 < 150). No GPU stage was run.
- **Cause (protocol, not model or data):** §13 `build_diagnostic` filtered to its temporal/static task categories (dropping EPM/HLD) *before* sampling 200 videos. Restricting v1 to those 200 IDs therefore could not recover EPM/HLD.

**Revised population `ovo-transfer-dev-v2`:**
- **Videos:** every video of the **original diagnostic hash half**, using the same seed and fraction read from the §13 split manifest and the same video-ID mapping.
- **Questions:** all official MC questions on those videos.
- **Exclusions:** every video hashing to the final half, including final-half videos absent from `final_frozen.jsonl`.
- **Unchanged:** the 600 s cap, per-video query cutoff, endpoint, adequacy thresholds (150 / 30 / 10%) and the +2 pp / CI > 0 rule.
- **Inventory stage reports:**
  - EPM/ASI/HLD and real-time counts over the complete conversion and each hash half (`half-counts.json`);
  - v1 next to v2 (`transfer-v1-comparison/` is kept and never evaluated).
- **Adequacy is claimed only if the v2 inventory passes.** If the whole diagnostic half is still too small or lacks task coverage, that is reported first, and nothing is borrowed from the final half.
- Earlier manifests and results are preserved.

**Later (sketch, not proposed for running):** a training-free redundancy-aware selector is a cheap baseline against B4 + FIFO at exact equal budget. Its failure would reject only that selector. It would not rule out learned content-aware or task-relevance memory mechanisms in general.
