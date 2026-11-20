# Data-use and evaluation plan after pilot-004 (issue #4 inventory, 2026-11-09)

Scope: what to do with the data already on the A100 host. No downloads. Pilot-004 runs unchanged (experiments.md §11), and nothing here assumes it passes. All numbers marked *(audit)* come from `scripts/launch_data_audit_a100.sh`, not from the operator's `du` listing.

## 0. Facts we already have

| Item | Value | Source |
|---|---|---|
| Stage 1 manifests | train 4,989 questions / 296 videos; val 488 / 29; video-disjoint | pilot-001..004 preflight |
| Training throughput | ≈ 4.9 s per update on 4×A100 (batch 1 clip per GPU, ≤ 30 s clips), ≈ 82 min per 1,000 updates per arm, ≤ 7.4 GB per GPU | pilots 002/003 logs |
| Data consumed | 1,000 updates × 4 clips = 4,000 samples: **< 1 pass over the questions**, but each training video is seen ≈ 13 times | arithmetic |
| Extracted media | One LLaVA subset in `/nas-data/vlm/llava/media` (95 GB); the manifests use 325 of its videos | operator |

The third row means the question count is not binding at pilot-004's duration, but **video diversity is**. That makes "more videos at the same duration" a meaningful single variable.

## 1. Audit first (read-only)

`scripts/launch_data_audit_a100.sh` (code `4de3105`). No GPU, idle I/O priority, extracts and trains nothing. It writes `audit.json`, `audit-digest.txt` and `listings.txt`:
- **Manifests:** counts, unique videos, sampled duration distribution, train/val overlap, missing and unreadable media.
- **Extracted media root:** video files and bytes; how many the manifests use; the **unused** count, bytes and sampled durations.
- **Per LLaVA subset:**
  - archive names and sizes (verified layout before any extraction command is given);
  - annotation files by kind (MC / OE / caption), QA turns and unique videos;
  - how many referenced videos are already extracted, and how many are extracted but unused (training expansion without extraction).
- **Per dataset under `/nas-data/vlm/data`:** extensions, archives, annotation-like files, top level (bounded walk).
- **Converter dry-runs** of the installed OVO-Bench, Video-MME and ShanghaiTech converters, written only into the report directory.
- **Overlap:**
  - NExT-QA video ids in LLaVA's NExT-QA-derived subsets against the standalone NExTQA split files;
  - name and byte-size collisions across UCF-Crime, XD-Violence, CCTV-Anomaly, CUVA, ShanghaiTech and mmFall.

  Videos inside unextracted archives are not seen.

Extraction commands are given only after the audit shows the actual archive names and layout.

## 2. Benchmarks: what runs now and what is local protocol

| Benchmark | Status | Protocol |
|---|---|---|
| OVO-Bench | `convert-ovo [--forward]` exists; dry-run in the audit | Backward/real-time: option-likelihood MC (**local**). Forward REC/SSR/CRR follow the official prompts and scoring rules, but SSR/CRR use first-token Yes/No likelihood (**local**). Not leaderboard-comparable as is. |
| Video-MME | `convert-videomme` exists (needs `pyarrow`); dry-run in the audit | Offline, no subtitles, option likelihood (**local**). The official protocol scores generated answers. |
| ShanghaiTech | `convert-anomaly` and `eval-alerts` exist | Step-level AUROC/AP at 1 step/s (**local**); not comparable with published frame-level AUROC. The test split stays evaluation-only (§5). |
| StreamingBench | **to implement** after the audit shows the layout | — |
| NExT-QA (standalone) | **to implement** (CSV multiple choice; small) | Option likelihood (**local**) |
| UCF-Crime, XD-Violence, CCTV-Anomaly, CUVA, mmFall | **to implement** (alert data), after the overlap and licence check | — |
| PhysicalAI-SmartSpaces (2 TB), finevideo, JoyAI | not planned until a concrete need appears | — |

**Rule:** every number reported as "benchmark X" states the protocol (local vs official). The paper's headline comparisons are within-protocol across arms.

## 3. Split policy (applies to every run below)

- **Development set:** the current `val.jsonl` (488 questions / 29 videos), frozen. All pilot decisions use it.
- **Held-out test set (new, used once):** about 1,000 multiple-choice questions on about 60 extracted-but-unused videos from the same subset. It is video-disjoint from train and val by construction, frozen before the next run, and **not evaluated until a final candidate is fixed**.
- **Independent benchmarks** (Video-MME, OVO-Bench, ShanghaiTech test, NExT-QA test) are never used for training or for choosing checkpoints and arms.
  - LLaVA NExT-QA-derived training data is **excluded** whenever its video ids overlap the standalone NExT-QA split used for evaluation (audit §overlap).
  - Monitoring sources are **not mixed** before the name/size collision check, and content checks on the candidates.

## 4. Next experiment: chosen by pilot-004's final result and selector diagnostics

Each row changes **one** variable relative to pilot-004. B4 is always a control arm, the original M2 definition is kept, and the final checkpoint is primary.

| Pilot-004 outcome (final checkpoint, `selector-diagnostics.json`) | Next run (one change) | Why |
|---|---|---|
| **A.** A slot arm passes M2 (≥ b4 + 2 pp, ≥ 95% native) | **E1:** same recipe and duration, training videos expanded with extracted-but-unused videos | Check that the win holds with more visual diversity before scaling |
| **B.** Exploratory continuation only (slot ≥ b4, CI lower bound > −1 pp, no M2) | **E1** as above | Video diversity is the binding resource (§0); a selector may need more distinct scenes |
| **C.** Slot arms lose, and the diagnostics show a **demonstrated** selector issue: per-slot/part entropies stay ≈ 1, emitted-token effective rank ≈ 1–2, or selector update norms collapse | **R1:** fix that one issue at pilot-004 scale (same data and duration), e.g. the selector logit scale/temperature or the selector learning rate, chosen from which statistic failed | Do not add data to a mechanism that does not select |
| **D.** Slot arms lose with sharp, diverse selectors (entropies well below 1, effective rank ≫ 1) | **L1:** same three trained arms evaluated, without retraining, on a **temporal subset** (OVO-Bench forward/backward, NExT-QA temporal/causal types), then possibly the 30–60 s subset | On 0–30 s multiple choice the representation is genuinely not better; test where transitions should matter before any redesign. If it loses there too, adopt B4 pooling for the compression path and centre the contribution on memory and alerts. |

**Not next in any row:**
- long-stream memory training (Stage 2) before Stage 1 has a selected representation (a passing slot arm or B4);
- 1–2 min or 2–3 min subsets before a 30–60 s result;
- mixing new monitoring data into Stage 1.

**Independent alert track** (can run in parallel on 2×5090; does not depend on transition outcomes):
- The alert head on a frozen Stage 1 checkpoint (B4 or the selected arm).
- Trained on UCF-Crime / XD-Violence training splits after the overlap check.
- Evaluated on ShanghaiTech test (full, evaluation-only) and UCF-Crime test.
- Compared with the LM Yes/No baseline (`eval-alerts --method head|lm`).

## 5. ShanghaiTech protocol

To keep ShanghaiTech an independent evaluation, its labelled test videos are **not** training positives. Convert with `--train-fraction 0` (all labelled videos become test) and `--normal-train 0` for evaluation-only use. Training positives come from other monitoring datasets after the overlap check.

## 6. Run specifications

### E1: more videos, same duration (rows A/B)

**Inputs:**
- **Training data:** MC questions from the extracted subset whose videos are not in val or the new test set (audit "extracted & unused"). Rebuilt with `convert-llava-mc` on the same media root, **no extraction**.
- **Volume:** target ≈ 3,000 training videos. Capped at 1,000 updates × 4 clips, so each video is seen about 1.3 times rather than 13; question repeats about 0.1.
- **Split:** val unchanged; test frozen and unused (§3).
- **Arms:** transition / after / b4 with pilot-004's config. Only `--records` changes, which preflight records in `identity.json`.

**Estimates** (assumptions: pilot-004 throughput, same clip lengths):
- **Training:** 3 arms × ≈ 82 min.
- **Evaluation:** native + 3 arms × 3 checkpoints, about 5 min each on 4 shards.
- **Total** about 5 h on 4×A100.
- **Storage:** about 5–10 GB of checkpoints, plus a few MB of predictions.

**Acceptance:**
- M2 (unchanged) at the final checkpoint.
- Exploratory continuation as in §11.
- Report the change against pilot-004 per arm, with paired video-bootstrap intervals.

**Artifacts:** `summary.json`, `selector-diagnostics.json`, the digest, and the new manifest hashes.

### R1: one selector fix (row C)

**Inputs:** pilot-004 data, duration, arms and budget. One model change, chosen from the failing statistic and declared before the run.

**Estimate:** about 5 h.

**Acceptance:** first the mechanism. The failing statistic must recover (for example per-slot entropy < 0.9 at update 1,000, and emitted-token effective rank > 4). Then M2 and the continuation criterion as above.

### L1: temporal evaluation of existing checkpoints (row D)

**Inputs:** pilot-004's final checkpoints, OVO-Bench (converted), and the NExT-QA test split (converter to implement).

**Estimate:** evaluation only, about 1–2 h.

**Acceptance:** slot arm ≥ b4 + 2 pp on the temporal subset (predeclared subset definition), with paired video-bootstrap CI.

## 7. Reporting

Every run reports three things:
- the code revision;
- the manifest and identity hashes;
- per-arm accuracy with CI, paired video-bootstrap deltas, budget identity, and the selector diagnostics.

Negative results are recorded in experiments.md like pilots 002–004.
