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
