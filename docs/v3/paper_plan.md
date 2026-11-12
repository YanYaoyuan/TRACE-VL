# Paper plan: ICCV 2027 (decided 2026-11-09)

**Target:** ICCV 2027. The deadline is **not yet confirmed**. ICCV's main deadline is usually in early March, so this plan assumes **≈ 2027-03-05** and must be re-dated when the official call appears. Everything is planned against an internal **results freeze on 2027-02-12**, three weeks before the assumed deadline.

**Assumptions:**
- One A100 pilot plus one review round takes about 1–2 days.
- The 2×5090s run the alert track in parallel.
- No new data downloads (data on the A100 host only).
- Holidays are budgeted as about one week of reduced throughput in late December.

## Milestones and gates

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
