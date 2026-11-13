# 历史计划：2026-11-13 已被核心 VLM 主线替代

此文件仅保留历史决策，不是当前执行指令。

# Paper plan: ICCV 2027 (decided 2026-11-09)

**Target:** ICCV 2027. The deadline is **not yet confirmed**. ICCV's main deadline is usually in early March, so this plan assumes **≈ 2027-03-05** and must be re-dated when the official call appears. Everything is planned against an internal **results freeze on 2027-02-12**, three weeks before the assumed deadline.

**Assumptions:**
- One A100 pilot plus one review round takes about 1–2 days.
- The 2×5090s run the alert track in parallel.
- No new data downloads (data on the A100 host only).
- Holidays are budgeted as about one week of reduced throughput in late December.

## Current plan (rewritten 2026-11-12; supersedes the milestone table below)

**Direction (owner, 2026-11-11).** No negative-results paper. The goal is a new, validated method for real-time video understanding on property cameras, in particular proactive alerts. The earlier experiments (experiments.md §14.5) stay as internal evidence and later ablations.

**What is established and what is not.**
- No positive contribution is established yet.
- Closed for this recipe: transition tokens; learned age/surprise retention (ties FIFO); the Stage-2 recipe (does not transfer to OVO, and the deficit persists with fresh per-question state).
- Reference systems: S1 + B4 + FIFO for the earlier track. The new track uses the original Qwen3.5-4B, frozen. The matched original-Qwen vs S1 control (`f9b9a8d`) has no reported result yet.

**Why alerts, and how narrow the opening is** (design_review_2026-11.md).
- MOSS-VL already trains "standing instruction, fire once when the condition is met" at 11B with public weights. JoyAI and Mage-VL train per-second speak/silent decisions.
- What the papers read so far do not report: false triggers per hour on long, mostly-normal footage, and detection delay at a fixed false-trigger rate. MOSS-VL loses on the anomaly-detection subset it reports.
- Our angle is a small model (2B–4B) on fixed cameras with long quiet stretches, judged by deployment metrics. It is not a general proactive model.

### Experiment sequence

| Step | What | Status | Decides |
|---|---|---|---|
| **P0** | Response-timing pilot: frozen Qwen3.5-4B, five gate heads (clock, frame, GRU, temporal, event-state), JoyAI response timestamps (`docs/event_state/README.md`) | Implemented by the tester; A100 report pending | Whether any head uses vision beyond a clock; which head P1 uses |
| **P1 step 1** | 100-sample annotation feasibility check on UCA: rewrite sentences into observable rules, prefix-verify `evidence_ready`, 30 double-annotated (`docs/event_state/causal_label_protocol.md` §10a) | **Tools implemented (`8475bd7`), CPU only; local test and host paths pending** | Whether evidence times can be labelled reliably, how far they are from the raw UCA times and in which direction, and the cost per item |
| **P1 step 2** | Smoke run: arms A / B / T, one seed, flow checks only | Not started | Nothing about accuracy; only that the pipeline is causal and counts correctly |
| **P1 step 3** | Main run: A raw times, B evidence times, C no full negative streams, T time-only; same samples, head, features and BCE loss; 3 seeds | Not started; runs only after steps 1–2 and the P0 report | Whether evidence-time labels improve response timing and beat the time-only baseline |
| Later | Focal vs BCE loss; a second head on the key A/B contrast; sequence-level delay/false-alarm objective (design review H-C); more data | Only if P1 shows a stable gain | — |

### Decisions already fixed for P1

- **One question:** with backbone, features, head and budget fixed, does replacing raw action times with verified evidence times improve response timing?
- **Target:** the model predicts an evidence state for a rule (READY / NOT_READY). First alert, de-duplication and release are one shared external policy for every arm.
- **UNKNOWN** is an annotation and evaluation mask in this round, not an output class.
- **Labels:** `event_start`, `evidence_ready` (confirmed on a clip that ends at the candidate time), `valid_end` (blank when unclear; none for historical-fact rules), uncertain intervals, evidence note, negative type.
- **Samples:** A and B use the identical sample set. Anything B drops after verification is dropped from A too, and reported.
- **Annotation effort:** 100 stratified items first. Calibration and test boundaries are human prefix-verified; training candidates are model-proposed and human-audited, and are not called verified labels if quality is insufficient.
- **Rules:** UCA rewrites are the main line; a few hand-written state rules are a separate trial, reported separately.
- **Negatives:** verified per rule (never / not yet / near miss). Normal video is not assumed negative. 30% negative trajectories is a fixed starting value.
- **Splits:** by source video, 70 / 15 / 15 (train / calibration / internal test). The official UCF-Crime test list is locked and reported separately at the end.
- **Thresholds:** chosen on calibration only, at ≤ 1 false trigger per rule-stream-hour.
- **Metrics:** event recall within 1 / 3 / 5 / 10 s of `evidence_ready`, early-trigger rate, false triggers per rule-stream-hour, delay (median, p95), miss rate, duplicate rate, PR-AUC on valid frames, UNKNOWN share. Un-debounced score metrics are reported next to the policy metrics.
- **Outcomes:** stable gain → re-check with a second head, then consider more data. No gain with narrow intervals → pause and examine labels, frozen features and the task. Wide intervals → "not enough samples", not "does not work".

### Immediate actions

| Who | Action |
|---|---|
| Owner / tester | Run `uv run pytest -q tests/test_event_state_annotation.py` locally |
| Owner / operator | Give the host paths of the UCA JSON files, the official `Anomaly_Test.txt` and the raw UCF-Crime videos |
| Annotators | P1 step 1: rewrite rules, prefix-verify, fill the sheets; deliver the rule list, verification records and `feasibility-report.json` |
| Operator | Finish the P0 A100 run and send its report bundle |
| Assistant | After the step-1 report and the P0 report: lock the P1 protocol, then implement the data adapter, label modes, shared alert policy, metrics, probes and a pinned launcher |

### Not allowed without a new decision

- New backbone training, Stage-2 scaling, or the 2×2 component-swap evaluations.
- Evaluating the OVO final half, or using OVO, StreamingBench or the official UCF-Crime test list for training or threshold selection.
- Promoting a setting to a paper claim after looking at development results.

### Schedule

The ICCV 2027 target and the 2027-02-12 internal freeze are unchanged as dates, but **the milestone table below no longer describes the work**. New milestone dates are set after P1 step 1, when the annotation cost per item is known. Before that, any date would be a guess.

### Risks specific to this plan

| Risk | Consequence | Mitigation |
|---|---|---|
| Evidence times are close to the raw UCA times | The A/B contrast carries little information | Step 1 reports size and direction first; priority then moves to negatives and `valid_end` |
| UCA sentences are long and interpretive | Many items are excluded or need heavy rewriting | Exclusion share and reasons are reported; cost per item is measured in step 1 |
| Annotation cost | Training labels cannot all be human-verified | Model-proposed labels with an audited error rate; small human-verified training set first |
| Only a few hours of normal footage, one dataset | False-trigger rates below about 0.25 per hour cannot be measured; no cross-dataset claim | State the limit in every report; cross-dataset check only after a positive P1 |
| MOSS-VL and JoyAI already cover proactive alerting as a task | Novelty must come from the small-model, deployment-metric setting | Include MOSS-VL-Realtime as a reference row; do not claim the task |
| Frozen 4B features may not carry the signal | Every arm is weak | P0 and the time-only arm show this early |

## Milestones and gates (superseded on 2026-11-12; kept for the record)

| # | Dates (2026–27) | Work | Gate / output |
|---|---|---|---|
| **0** | Nov 9 – Nov 13 | Held-out probe of pilot-004; local verification (full suite incl. real-SDK W&B); pilot-004 W&B upload (no demos); **data audit** (in parallel) | Probe reviewed; tests and CI green; audit digest posted |
| **1** | Nov 13 – Nov 20 | **Pilot-005** (grid gist, 3 arms, about 5 h) and review | Outcome A / B / C (table below) |
| **2** | Nov 20 – Dec 11 | **Main Stage 1 result:** final recipe from the outcome; more training videos from extracted media (E1); 3 seeds × arms; frozen held-out test set evaluated **once** at the end; OVO-Bench and Video-MME (existing converters, local protocol stated) | Main table, with paired video-bootstrap CIs and seed variance |
| **3** | Dec 1 – Jan 15 | **Long-video memory (Stage 2):** build held-out long streams (≥ 5 min; LLaVA longer subsets or MEVA, per the audit); Stage 2 training; {representation} × {tkm, mass, fifo} retention at an equal budget | Long-stream accuracy and evidence-kept table; learned retention must beat mass and fifo |
| **4** | Nov 16 – Dec 23 (2×5090, parallel) | **Alert head:** UCF-Crime / XD-Violence converters after the overlap check; train the head on a frozen Stage 1 checkpoint; evaluate on ShanghaiTech test (evaluation only) and UCF-Crime test against the LM Yes/No baseline (AUROC/AP and ms per decision) | Alert table and the speed comparison |
| **5** | Jan 4 – Feb 12 | **Ablations and robustness:** CST gate (the second candidate issue), token budget sweep, seeds; efficiency (KV, latency, throughput) on A100 and 5090; demo figures from W&B | Ablation tables and efficiency plots |
| **6** | Dec 14 – Feb 26 (overlaps 3–5) | **Writing:** method and related work from `design.md` / `literature.md`; figures; results sections as each table freezes. Internal review on Feb 12 (**results freeze**), full draft Feb 19, polish Feb 26 | Submission-ready draft |
| — | ≈ Mar 5 (TBC) | ICCV 2027 submission; supplementary (code snapshot, configs, run identities, negative pilots) | — |

## Branch after pilot-005 (milestone 1)

| Outcome | Paper claim | Effect on the schedule |
|---|---|---|
| **A.** Slot arms beat B4 (M2 met or near) | Transition tokens, memory and alerts in a ~2B streaming model | As planned |
| **B.** Slot arms ≈ B4 | Transition tokens as an efficient alternative; the claim rests on temporal and long-video benchmarks (L1 first, in about 2 days) | Milestone 2 starts with L1 |
| **C.** Slot arms still lose | A ~2B streaming monitor with B4-style compression, learned memory retention and the alert head; transition tokens reported as a negative ablation (pilots 002–005) | Milestone 2 shrinks; 3–4 become the core; buffer gained |

## Risks and buffers

- **One more failed pilot:** absorbed by milestone 2's three-week window.
- **Stage 2 data:** the held-out long streams depend on the audit. If not enough ≥ 5-minute videos are extracted, extract one LLaVA 1–2 min / 2–3 min archive (sizes in the inventory) **after** the audit shows the layout.
- **Benchmark protocol:** all numbers are labelled local vs official; no leaderboard claims without the official protocol.
- **Single-seed results** are never used in the paper. The main table uses 3 seeds and a once-only held-out test set.

Progress against this plan is posted on issue #4 at each milestone.

## Status 2026-11-10: outcome C

- Pilot-005 failed the gate (experiments.md §12.1).
- The temporal diagnostic failed its predeclared rule (§13.1). The transition track is closed, and **B4 is the compression baseline**.
- The paper follows outcome C: a ~2B streaming model with B4-style compression, **learned memory retention** (to be shown: §14 memory pilot) and the **alert head** (separate audited experiment).
- Transition tokens are reported as a negative ablation for this implementation and recipe.
- Milestone 2 shrinks to B4 + seeds. Milestones 3 (memory) and 4 (alerts) are now the core contributions; neither is established yet.

## Status update 2026-11-10 (evening): memory pilot repair (experiments.md §14.2)

- **Learned retention (TKM) is negative.** At exactly equal KV it ties FIFO: −0.18 pp [−0.57, 0] beyond-window. The scorer tracks recency (ρ 0.93; Jaccard vs FIFO 0.96). The age/surprise-only retention claim is dropped.
- **Stage-2 training is positive on dev.** S2 − S1 under FIFO: +4.90 pp [2.46, 7.30] beyond-window, +3.87 pp [1.83, 5.89] overall. This is a candidate contribution only once it transfers to an independent benchmark (§14.3, proposed).
- Working baseline: **B4 compression + Stage-2 checkpoint + FIFO memory.** A memory contribution now requires a content-aware mechanism that beats this baseline at equal budget. A cheap training-free selector is one baseline for this, not a test of every content-aware design.


## Status update 2026-11-11: Stage-2 gain did not transfer (experiments.md §14.3c, preliminary)

- On the external OVO development population (`ovo-transfer-dev-v2`, 688 q), S2-FIFO − S1-FIFO is **−4.92 pp [−9.62, −0.35]** on backward and −4.18 pp on real-time. The predeclared continuation rule fails, and the real-time no-harm check fails too.
- **Consequence:** the current Stage-2 recipe is not a contribution and is not scaled. The working checkpoint for external evaluation reverts to **Stage 1 (B4) + FIFO** until a Stage-2 recipe is shown to transfer.
- The memory contribution has no positive evidence: TKM ties FIFO (§14.2), and Stage-2 training does not transfer (§14.3c). The **alert head** is unvalidated: a possible direction, not an established contribution.
- **Confirmed** by the tester's independent raw-file audit (issue #4, 2026-11-11): all 39 paired statistics reproduce, and KV matches per question between S1 and S2.
- Scope: this rejects the age/surprise scorer and this Stage-2 recipe, not content-aware memory or long-stream training in general.
- Next diagnostic (specified, not run): the recent-window control, experiments.md §14.4.

## Status update 2026-11-11 (evening): recent-window control (experiments.md §14.4a–§14.6)

- The Stage-2 regression on OVO **persists with fresh per-question state** (fresh16: −4.92 pp [−8.57, −1.40] on backward; interaction with FIFO 0.00 [−4.95, +4.65]). Carried state or eviction alone does not explain it. The new arms are screenshot-based until the raw files are recomputed.
- **No paper contribution is established yet.** The transition track is closed. Learned age/surprise retention ties FIFO. Stage-2 training does not transfer.
- **Reference system:** S1 + B4 + FIFO. No setting is promoted after inspecting development results. The OVO final half stays frozen.
- **Alerting** is an unvalidated, product-relevant direction. A credible study needs timestamped events, false alarms per camera-hour, event recall and detection delay at matched compute. An alert head with clip accuracy is not enough.
- **Next (proposed, evaluation only):** a CPU checkpoint difference, then a 2×2 LoRA × modules swap (§14.6). Any later training needs one hypothesis and one validation gate written down first.
- **Schedule risk:** the paper-plan milestones assumed a memory or Stage-2 contribution by now. The plan needs a new central claim before the milestone dates mean anything.

## Status update 2026-11-12: owner direction and design review

- **Owner direction (issue #4, 2026-11-11):** no negative-results paper. The goal is a new, validated method for real-time video understanding, in particular property-camera monitoring and proactive alerts. The failed experiments stay as internal evidence and later ablations.
- **Design review:** `docs/v3/design_review_2026-11.md` compares prior methods and proposes three hypotheses. It recommends **H-C, alert timing as a learned sequential evidence test**, on the frozen S1 + B4 backbone, measured by event recall at a fixed false-alarm rate per camera-hour and by detection delay.
- Nothing is launched until the review is accepted. The milestone dates in this file are suspended until then.
