# Research and design review: toward a positive contribution (2026-11-12)

Requested on issue #4 (owner direction, 2026-11-11). This is a design proposal backed by reading, not a promise that the method works. No training or GPU evaluation is launched by this document. **S1 + B4 + FIFO stays the reference system.** The failed experiments (experiments.md §14.5) stay as internal evidence and later ablations.

## 0. How each source was checked

| Level | Meaning |
|---|---|
| **F** | Full text read this week (arXiv HTML or the PDF), mechanism details taken from the paper body |
| **A** | Abstract page only; mechanism details beyond the abstract are not verified |
| **S** | Search-result snippet only; treat every detail as unverified |
| **P** | Earlier project notes (`docs/v3/literature.md`) or my prior knowledge; not re-read this week |

Repository licences were read from the GitHub API on 2026-11-12. "No licence file" means the repository has no detected licence, so reuse is not permitted by default. I hit the fetch limit before reading four sources in full; they are marked A, S or P and listed in §6.

## 1. What each method actually learns, and where its state lives

| Method (level) | What is learned | Where persistent state lives | Reactive or proactive | Data and supervision | Code and licence | 2B feasibility on 2×5090 / 4×A100-40GB |
|---|---|---|---|---|---|---|
| [VideoLLM-online / LIVE](https://github.com/showlab/videollm-online) (P) | LLM fine-tuned with a streaming objective: per frame, predict "stay silent" (EOS) or start a response | The LLM's context / KV | Proactive (per-frame decision by the LLM) | Ego4D narrations and COIN converted to streaming dialogue | Apache-2.0 | Recipe fits; needs time-aligned narration data |
| [StreamingVLM](https://arxiv.org/abs/2510.09608) (P) | SFT on short overlapped chunks that mimic inference | KV: attention sink + ~16 s vision window + long text window | Proactive commentary | Sports commentary streams | Repo not found under the name I tried | Our Stage-2 recipe was modelled on it; our version did not transfer (§14.3c) |
| [ReKV](https://arxiv.org/abs/2503.00540) (A) | Nothing (training-free) | KV caches of the whole video, offloaded to **RAM and disk**, reloaded on retrieval | Reactive | None | No licence file | Runs, but memory is host RAM/disk, not a fixed GPU budget; must be reported as such |
| [Flash-VStream](https://arxiv.org/abs/2506.23825) (A) | A two-part memory module: low-capacity context memory + high-capacity detail memory | Visual-token memories | Reactive | Trained; details not in the abstract | Apache-2.0 | Plausible |
| [Harnessing Streaming Video in the Wild](https://arxiv.org/abs/2606.08615) (F) | Qwen3-VL-8B; vision frozen, connector + LLM fully updated; first output token each second is `</silence>` or `</response>`; token-weighted NLL (first silence 1, repeated 0.8, response 1.5) | Last 200 s as raw vision tokens; 5 mid-term and 42 long-term **text** summaries, prefilled into KV | Proactive, per second | Streaming-Train-248K from 19 public datasets; labels from dense temporal annotations and ASR | "Will open-source"; no repo on the page | Recipe is the expensive part: about **4,096 H200 GPU-hours**. Not reproducible at our scale |
| [JoyAI-VL-Interaction](https://arxiv.org/abs/2606.14777) (F) | 8B model; silence / response / delegate **tokens**; weighted CE (1 / 0.4 / 1.5); then GRPO with separate timing and content rewards | Last 100 s raw vision tokens; text summaries (5 mid, 15 long); dialogue memory | Proactive, per second | > 4M time-aligned clips in six families, including "proactive alerting" with alert onsets confirmed by a dense 1 fps check | Apache-2.0; data JSON on HF (media separate) | Weights usable as an 8B reference; training recipe out of reach |
| [Mage-VL](https://arxiv.org/abs/2607.24904) (F) | 4B; own encoder pretrained on ~560M images + 100M frames; a **gate head on encoder features** predicts speak / silent per step | Recent frames; no long-term memory in the OVO protocol (4 recent frames) | Proactive (System-1 gate, then System-2 decoder) | Gate labels: caption start timestamp = speak, every other timestamp = silent | MIT | Encoder pretraining out of reach; gate design is cheap |
| [SelectStream](https://arxiv.org/abs/2606.16353) (F) | VLM **frozen**; trained: segment encoder, write gates, query encoder, graph attention, calibration layer. Loss = answer + retrieval + sparsity | A 256-node latent memory graph; 8 retrieved latent tokens injected next to the current observation | Reactive, query-conditioned **at read time** | Streamo-Instruct-465K | Project page only | Module sizes fit; needs the 465K instruction set |
| [StreamOPD](https://arxiv.org/abs/2608.16320) (F) | **Qwen3.5-4B** student distilled on-policy from a 9B teacher | None: fixed recent-4-frames protocol | Reactive | ~25k verifiable items built from LLaVA-Video and Tarsier2 | Project page; release not stated | Most relevant recipe for our backbone family; needs a teacher and 8 H200s in their setup |
| [ReMem](https://arxiv.org/abs/2610.05940) (F) | Nothing (training-free) | Fixed-length compressed visual tokens (kept by attention to a generic prompt) + one retrieved clip | Reactive | None | No release stated | Reported ~32 GB peak on an 80 GB H100 with 7–8B models |
| [StreamTTT](https://arxiv.org/abs/2608.13416) (A) | 4B; long history written into online-updated fast weights | Fast weights + short sliding KV | Reactive | Offline long-video QA + a real-time QA corpus | Apache-2.0 | Closest in scale; test-time loss not verified |
| [SimpleStream](https://arxiv.org/abs/2604.02317) (F) | Nothing: last N frames at 1 fps into an off-the-shelf VLM | None | Reactive | None | No licence file | Trivial; our `fresh4` is a relative of it |
| [LiveStarPro](https://arxiv.org/abs/2606.17798) (F) | 7B; trained to caption every frame of a clip; **silence is derived**, not learned: speak when perplexity of the last caption rises by a ratio α | In-context frames pruned by perplexity + an external event tree | Proactive | 83K samples (ActivityNet Captions, Shot2Story, Ego4D, OmniStarPro) | No licence file | Recipe size is feasible |
| [StreamPro](https://arxiv.org/abs/2605.16381) (A) | SFT with a class-balanced stream loss, then GRPO with turn- and trajectory-level rewards | Not stated in the abstract | Proactive | Not stated in the abstract | No repo on the page | Unknown |
| StreamMind, Dispider, MMDuet (S) | A gate network before the LLM; separate decision module; added "informative" and "relevance" heads | — | Proactive | — | MMDuet MIT; Dispider repo not found | — |
| [HIMMEL](https://arxiv.org/abs/2605.08158) (A) | A codec-domain adapter (motion vectors, residuals) aligned contrastively to a frozen ViT | None (offline long video) | Reactive | — | Apache-2.0 | Not a streaming method |
| StreamingEval (ACL Findings 2026) (P) | Evaluation framework | — | — | — | — | Not read this week |

Reported speeds are **not comparable** across rows. Examples of the conditions behind them: Harness "below 1 s" is time-to-first-token with prefix caching on one H200 at 1 fps; ReMem "about 1.0 s per response" is on one 80 GB H100; LiveStarPro "3.96 FPS" is on 5-minute videos; StreamMind "100 fps on a single A100" is from a search snippet and concerns its gate path.

### Proactive and alert evaluation

| Benchmark (level) | What it scores | Findings relevant to us | Access |
|---|---|---|---|
| [SPOT-Bench](https://arxiv.org/abs/2604.24317) (F) | Timeliness-F1 over 5,093 response slots with task-specific tolerance (±0.5 s to ±2.5 s) | Offline models detect events but **repeat responses**; silence post-training makes them **unresponsive**. About half the stream is dead time. Best macro T-F1 is about 10–12% | Code MIT; data on HF |
| [P³IBench](https://www.alphaxiv.org/abs/2610.21568) (A: alphaXiv summary page) | Proactive response score by distance to target time, divided among duplicate replies. Includes **Accident Warning (100)** and **Crime Alert (100)** | Same two failure directions. Fine-tuning Qwen3-VL-4B raised the score from 28.62 to 42.12 | Repo, no licence file |
| Mage-VL Table 6, SoccerNet-Caption (F) | Trigger accuracy, TimVal, per-position F1, PR-AUC | Mage-VL PR-AUC is **9.30**; JoyAI collapses toward silence (TimVal 19.25). Timing is far from solved | — |
| OVO-Bench forward tasks (P) | Polling-time Yes/No (SSR, CRR) and counts (REC) | Already on our host: 3,035 converted records include the forward polling times | MIT |

### Video anomaly detection with language

| Work (level) | Relevance |
|---|---|
| [AnyAnomaly](https://openaccess.thecvf.com/content/WACV2026/papers/Ahn_AnyAnomaly_Zero-Shot_Customizable_Video_Anomaly_Detection_with_LVLM_WACV_2026_paper.pdf) (S) | "Customizable VAD": a user's text defines the abnormal event; zero-shot LVLM scoring. This is prior art for rule-conditioned detection |
| [LaGoVAD](https://arxiv.org/abs/2503.13160) (S) | Language-guided open-world VAD under weak supervision. Prior art for training a detector conditioned on a text definition |
| [A VLM Answer Is Not an Anomaly Score](https://arxiv.org/abs/2608.21244) (A) | Probability-weighted scores beat decoded answers by 7.66–19.95 points. Supports our likelihood scoring; ties must be handled in AUROC |
| [Benchmark AUC Is Not Deployable Reliability](https://arxiv.org/abs/2606.29506) (S) | At a threshold catching 90% of anomalous frames, the median false-alarm rate is about 31,931 per hour; cross-dataset performance falls to chance (feature-based detectors) |
| [Strictly Causal Streaming VAD](https://arxiv.org/abs/2608.24810) (S) | Links a recurrence's decay to detection delay. Non-VLM, but it is the kind of delay analysis we should match |

## 2. Findings across the literature that constrain our design

1. **Recent-window baselines are the bar.** SimpleStream with 4 frames reaches 81.4 real-time / 54.0 backward on OVO-Bench with Qwen3-VL-8B, above every published streaming method it compares to. StreamOPD fixes the same memory-free protocol. Any memory claim must beat a recent-window arm on the same backbone.
2. **Memory helps backward tasks and hurts present-scene perception.**
   - SimpleStream: 16 frames give backward 56.4 but real-time falls from 81.4 to 77.9. Adding retrieval raises EPM by 7.1 and ASI by 6.1, and lowers HLD by 12.4 and OCR by 8.1.
   - Our own controls show the same shape: `fresh4` beats `fresh16` on real-time by +5.5 (S1) and +7.1 (S2), while FIFO beats `fresh16` on all questions by about +3.3 (§14.4a).
3. **The gains others report come from training data and recipe scale, or from a stronger backbone.** Harness used ~4,096 H200 GPU-hours; JoyAI used > 4M time-aligned clips; Mage-VL pretrained its own encoder. We cannot compete on those axes.
4. **Proactive timing is trained as per-step token classification with loss reweighting**, sometimes followed by RL with a timing reward (JoyAI, StreamPro), or derived from perplexity (LiveStarPro). Reported timing quality is low (Mage-VL PR-AUC 9.30; SPOT-Bench T-F1 about 10%). Both failure directions, spamming and silence, are documented.
5. **None of the VLM proactive papers I read reports false alarms per camera-hour or detection delay at a fixed false-alarm rate.** The anomaly-detection literature shows why that matters (finding in 2606.29506). My search found no VLM benchmark with those metrics; that is a search result, not proof of absence.
6. **Query-conditioned memory exists, but conditions at read time.** SelectStream and ReMem write a query-agnostic memory and retrieve when the question arrives. A monitoring rule is different: it is known **before** the stream starts.

## 3. Three hypotheses

Each uses the frozen S1 + B4 backbone as the starting point. Estimated costs are from our own measured throughput where we have it and are marked *(est.)* otherwise.

### H-A. Rule-conditioned evidence state

**Idea.** For a standing natural-language rule ("someone is climbing the fence"), keep a small per-rule state that is updated every step with support and contradiction for that rule, including how long the supporting evidence has lasted. Memory is selected by relevance to the rule at **write** time, not by age or surprise.

| Item | Content |
|---|---|
| **Closest prior work and the difference** | SelectStream: query-conditioned, but at read time over a query-agnostic graph; state grows with the stream up to 256 nodes. MMDuet's relevance head: per-frame relevance to a query, no accumulated state. AnyAnomaly and LaGoVAD: text-conditioned anomaly scores per segment, no persistent evidence state, not trained for timing. Our existing alert head: per-step BCE over the current step plus a FIFO of 256 latent tokens; its memory is selected by recency. **Difference:** the state is O(rules), fixed size, and written by rule relevance. |
| **Architecture and data flow** | Rule text → frozen LM embedding → rule vector r. Each step: cross-attention from r to the step's tapped hidden states (the taps the alert head already reads) → evidence vector v_t and a signed support score. State e_t = gated update of e_{t−1} with v_t (a small GRU-sized cell per rule). Output: support, contradiction, duration. |
| **Loss and labels** | Per-step support labels from timestamped event sentences: a sentence's interval is positive for a rule built from that sentence, negative elsewhere and for rules from other videos. Contradiction needs explicit negatives ("the gate is closed" vs "the gate is open"), which our data mostly lacks. |
| **Frozen / trainable** | Backbone, LoRA and B4 modules frozen. Trainable: rule projection, cross-attention, state cell, output heads (about the size of the current 14M alert head). |
| **Data from our downloads** | UCA: 23,542 timestamped sentences on 1,854 UCF-Crime videos (train/val/test JSON). CCTV-Anomaly (10 classes; whether it has timestamps must be checked). |
| **Equal-budget baselines** | The existing alert head with its FIFO latent memory at the same parameter count; a recent-window-only head; zero-shot yes/no likelihood from S1. |
| **Ablations** | No state (current step only); state without rule conditioning at write; FIFO memory instead of the state. |
| **Latency and memory** | One small cross-attention and one cell update per rule per step. No text generation. State is a few KB per rule. |
| **Decisive experiment** | On UCA validation videos: rule-conditioned state vs the existing alert head, per-step AP on held-out sentences, plus a **paraphrase split** (rules reworded) and a **persistence split** (events longer than 16 s, scored only after the first 16 s). Success: ≥ +3 AP points on the persistence split with a video-cluster CI lower bound > 0, and no loss on the short-event split. |
| **Cost** *(est.)* | One feature-extraction pass shared with H-C (see there); head training on cached features: under 2 GPU-hours per seed. |
| **Why the supervision can teach it** | UCA gives sentence-level intervals, so "is this rule supported now" is directly labelled. It does **not** label contradictions or rule compositions, so those parts would be untrained. This is the main weakness. |
| **Risk** | AnyAnomaly and LaGoVAD already occupy "text-defined anomaly". Without the contradiction/duration behaviour being measurably learned, this reduces to a text-conditioned classifier. |

### H-B. Recent-first perception with learned history arbitration

**Idea.** Keep a short recent window always visible and expose history only when it helps. Train the decision with counterfactual labels: for each training question, does adding history change the answer from wrong to right, or right to wrong?

| Item | Content |
|---|---|
| **Closest prior work and the difference** | SimpleStream states the principle ("recent-first, history-on-demand"). SelectStream implements a learned version with a frozen 8B VLM and reaches 62.2 backward / 82.8 real-time on OVO-Bench. StreamOPD's appendix reports the same trade-off. **The two-path idea is established prior art.** The only candidate for novelty is the counterfactual supervision signal, and SelectStream's retrieval and sparsity losses are close to it. |
| **Architecture and data flow** | Path 1: last 4 s (our `fresh4` shape). Path 2: FIFO history. A small gate reads the question and the recent-window hidden state and outputs "use history" before the answer pass. |
| **Loss and labels** | Label per question = sign of (correct with history − correct without). Ties are ignored. BCE on the gate. |
| **Frozen / trainable** | Everything frozen except the gate (< 1M parameters). |
| **Data** | Stitched LLaVA development streams for training labels (two forward passes each); OVO development population for evaluation only. |
| **Baselines** | Always-recent (`fresh4`), always-history (FIFO), a task-type heuristic, and the oracle. |
| **Latency and memory** | When the gate says "recent only", the answer pass sees fewer tokens. History must still be kept, so memory is unchanged. |
| **Decisive experiment** | **CPU only, from predictions we already have.** On the 688 OVO questions, compute the oracle that picks the better of `fresh4` and FIFO per question for S1. If the oracle is less than +4 pp above the better single arm, stop: there is no headroom worth learning. |
| **Cost** | Minutes of CPU on the existing report bundle. |
| **Why the supervision can teach it** | Counterfactual labels are exact for the training questions. Whether a gate trained on stitched LLaVA streams transfers to OVO is the same transfer risk that sank Stage 2. |
| **Risk** | Highest novelty risk of the three, and an 8B frozen-VLM method already reports strong numbers in this exact space. |

### H-C. Alert timing as a learned sequential evidence test

**Idea.** Treat "raise an alert" as a stopping decision. The head outputs a per-step evidence **increment**; a running statistic accumulates it and resets at zero (CUSUM form: S_t = max(0, S_{t−1} + ℓ_t)); an alert fires when S_t crosses a threshold. Train the increment with a sequence-level objective that trades detection delay against false alarms, instead of per-step classification.

| Item | Content |
|---|---|
| **Closest prior work and the difference** | Mage-VL: binary gate per step, labels at caption starts, cross-entropy. JoyAI and Harness: silence/response tokens with reweighted cross-entropy; JoyAI and StreamPro add RL with a timing reward. LiveStarPro: perplexity-ratio gate. Classical: CUSUM and hysteresis on a per-frame score; online action detection. **Difference:** (1) the model learns the increment of an explicit accumulated statistic under a delay/false-alarm loss, so the operating point is a single threshold with a meaning; (2) accumulation responds to a **state that persists**, not to change, so a low-motion or long-lasting hazard keeps adding evidence. Motion- or surprise-driven gates (Mage-VL's codec selector, our failed age/surprise scorer) should, by construction, be least sensitive there; that is an expectation the duration and motion splits test, not a measured fact; (3) evaluation in false alarms per camera-hour and delay, which the VLM papers I read do not report. |
| **Architecture and data flow** | Frozen S1 + B4 streams the video. The existing taps feed a small head that outputs ℓ_t per rule. S_t is one scalar per rule. On an alert, the rows of the last 16 s (already in KV) give pre-event context for an optional description; that part is not in the first experiment. |
| **Loss and labels** | For an event interval [a, b] with threshold h and temperature T: soft non-detection probability up to t is P_t = Π_{τ=a..t} (1 − σ((S_τ − h)/T)); delay loss = Σ_{t=a..b} P_t (expected delay), plus a miss term P_b. False-alarm loss on normal steps = Σ σ((S_t − h)/T). Total = delay + λ·false alarms; λ sets the operating point. Labels: event onsets and offsets from timestamped annotations; normal videos supply long negative stretches. |
| **Frozen / trainable** | Backbone, LoRA and B4 frozen. Trainable: the increment head only (≤ 14M, the current alert head's size). |
| **Data from our downloads** | Train: UCA train sentences on UCF-Crime videos, plus UCF-Crime normal training videos. Development: UCA val. Held-out: UCF-Crime test videos with temporal anomaly annotations. Cross-dataset check: XD-Violence test (frame-level labels). ShanghaiTech test stays evaluation-only. The overlap and licence audit (`data_plan.md` §1) must pass first; UCA test videos overlap UCF-Crime test and are excluded from training. |
| **Equal-budget baselines** | All on the **same cached features and the same head size**: B0 zero-shot S1 yes/no likelihood per step (probability-weighted); B1 per-step BCE head + threshold; B2 = B1 + the best of EMA, k-of-n and hysteresis, tuned on development; B3 = B1 + CUSUM on its calibrated logit with drift and threshold tuned on development; B4r = the existing alert head with its FIFO latent memory. |
| **Ablations** | Our loss without accumulation (per-step); accumulation with BCE (this is B3's trained twin); without reset at zero; events split by duration (< 8 s, 8–30 s, > 30 s) and by motion level. |
| **Latency and memory** | One small head call per rule per step; one scalar of state per rule; no generation until an alert. The cost is dominated by the streaming step we already pay. Must be measured, not assumed. |
| **Decisive experiment** | One feature-extraction pass, then head training on cached features with 3 seeds per arm. **Primary metric:** event recall at **1 false alarm per camera-hour** on held-out UCF-Crime test, with thresholds chosen on development only. **Success:** ours beats the best of B1–B3 by ≥ 5 pp recall with a video-cluster 95% CI lower bound > 0, **and** median detection delay is not worse (CI upper bound of the delay difference ≤ +1 s), **and** on XD-Violence test recall is not below B2 by more than 2 pp (CI lower bound > −2). **Failure:** anything else. In that case sequence-level training adds nothing over classical post-processing on our features, and H-C is closed. |
| **Cost** *(est.)* | UCF-Crime is about 128 hours of video, so about 460k steps at 1 step/s. Feature extraction: 15–40 GPU-hours (4–10 h on 4×A100); the real rate comes from a 20-video timing run first. Storage: about 6 GB for pooled taps in fp16. Head training: under 1 GPU-hour per arm-seed. |
| **Why the supervision can teach it** | The loss needs only event intervals and normal stretches, which UCA and UCF-Crime provide. The timing trade-off is in the loss itself, not in label reweighting. What it cannot teach: rules never seen in training (zero-shot rules), and rare events with a handful of examples. |
| **Risks** | (1) UCF-Crime normal test footage is tens of hours, so rates below about 0.1 false alarms per camera-hour cannot be measured. (2) Cross-dataset collapse is the documented norm for anomaly detectors. (3) If per-step features from a frozen 2B model are weak, every arm is weak and the comparison is uninformative; B0 and B1 absolute levels are reported first as a sanity gate. (4) "CUSUM with a learned increment" has a long history in sequential detection; the claim would be about training a VLM alert head this way and showing it matters under deployment metrics, not about the statistic. |

## 4. Ranking and recommendation

| Rank | Hypothesis | Novelty risk | Supervision available now | Cost of the decisive test | Fit to property cameras |
|---|---|---|---|---|---|
| **1** | **H-C** alert timing as a sequential test | Medium: the statistic is classical; the training objective and deployment metrics are absent from the VLM papers I read | Yes: event intervals + normal video | About one day of 4×A100 for features, then hours | Direct: false alarms per camera-hour and delay are the product metrics |
| 2 | H-A rule-conditioned evidence state | Medium-high: text-defined anomaly detection exists (AnyAnomaly, LaGoVAD) | Partly: support yes, contradiction no | Shares H-C's features | Direct, but depends on H-C's machinery |
| 3 | H-B recent/history arbitration | High: SelectStream and SimpleStream cover it | Yes | Minutes of CPU | Indirect (QA, not alerts) |

**Recommendation: H-C.** It is the only one whose decisive test uses supervision we already hold, runs on a frozen backbone within our compute, has strong non-VLM baselines that can falsify it, and is measured in the units a property-camera deployment cares about. H-A is the natural second step **only if** H-C succeeds: the increment becomes rule-conditioned. H-B's CPU check is cheap enough to run alongside, as a go/no-go on that direction, not as a contribution.

This is consistent with the tester's requirement for an alert study (timestamped events, false alarms per camera-hour, event recall, detection delay at matched compute).

## 5. Minimal experiment, in order

1. **Data gate (CPU).** Run the pending overlap and licence audit for UCF-Crime, UCA, XD-Violence and CCTV-Anomaly. Confirm the UCF-Crime test temporal annotations and UCA JSON are on the host. Record how many normal camera-hours the held-out set has; that fixes the lowest measurable false-alarm rate.
2. **Predeclare** the protocol in `experiments.md` (§15): splits, rule construction from UCA sentences, the five baselines, the loss, the metric code, thresholds chosen on development only, and the success/failure rule in §3 H-C.
3. **Timing run** on 20 videos to replace the cost estimate with a measurement.
4. **Feature extraction** once with frozen S1 + B4 (evaluation-mode streaming, no training).
5. **Sanity gate.** Report B0 and B1 levels. If B1's AP on development is near chance, stop and report: the frozen features do not carry the signal, and no head design will fix that.
6. **Train the arms** on cached features, 3 seeds each; evaluate once on the held-out set.
7. **H-B CPU check** on the existing 688-question predictions, reported separately.

Nothing here retrains the backbone. A later backbone-training step would need its own single hypothesis and gate.

## 6. What I could not verify, and should be read before the protocol is frozen

- **AnyAnomaly and LaGoVAD full texts.** They are the closest prior art for H-A and partly for H-C; I have only search snippets. If either already trains with a delay/false-alarm objective or reports false alarms per hour for a VLM, H-C's difference narrows.
- **VideoLLM-online and StreamingVLM** were not re-read this week; the table rows rely on earlier notes and prior knowledge. The StreamingVLM repository name I tried did not resolve, so its licence is unknown.
- **StreamPro, StreamTTT, ReKV, Flash-VStream** are at abstract level. StreamPro's "CB-Stream loss" and its GRPO timing reward are the closest training objectives to H-C and need a full read.
- **StreamingEval** (ACL Findings 2026) and the JoyAI paper's last 40k characters (appendix, hardware for the latency claim) were not read.
- **Online action detection** and **early event detection** literature was not surveyed; it is where learned stopping rules with delay objectives are most likely to exist already.
- **CCTV-Anomaly annotations:** whether they carry timestamps is unchecked.
