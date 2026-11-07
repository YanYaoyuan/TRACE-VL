# TRACE-VL v3 design: transition-tokenized streaming on a hybrid VLM (~2.2B)

Status: design proposal, nothing implemented or measured yet. Numbers marked *(est.)* are
back-of-envelope estimates to be replaced by measurements in M0.

## 1. Thesis

A streaming video model should not re-read the world every frame. Most of a stream is unchanged; the
information is in **transitions**: what changed, what it looked like before, and when. v3 makes
transitions the unit of computation and of memory inside a pretrained hybrid VLM:

1. a **causal spatio-temporal adapter** (CST) gives the vision tokens cross-frame context the
   backbone lacks (its ViT attends within 2-frame patches only);
2. a **transition encoder** (TE) turns each step into a few *latent P-frame* tokens: surprise-gated
   before / after / delta tokens, instead of re-sending 196 tokens;
3. a **transition KV memory** (TKM) keeps those tokens, and only those, long-term in the 6
   full-attention layers, while the 18 Gated DeltaNet layers carry a fixed-size gist of everything.

What v1 and v2 taught us, and how v3 answers it:
- v1 failed because randomly initialized modules sat between vision and language. v3 modules are
  **zero-initialized residual adapters in the backbone's own token space**: at step 0 the model *is*
  Qwen3.5-2B, so it can only move away from it where the loss improves.
- v2 showed transitions are useful evidence, but used a hand-built pixel detector and heuristic
  ranking with nothing learned. v3 learns *when* to write (surprise), *what* to write (before / after /
  delta tokens) and *what to keep* (retention), and the transition encoder works on features, so the
  camera need not be fixed.

## 2. Backbone facts that drive the design (Qwen3.5-2B)

| Item | Value | Consequence |
|---|---|---|
| LM layers | 24 = 18 Gated DeltaNet + 6 gated full attention (every 4th) | recurrent state is O(1) per stream; only 6 layers have a growing KV |
| Full-attn KV | 2 KV heads × 256 dims × 6 layers | ≈ 12 KB/token bf16 *(est.)*: 100K cached tokens ≈ 1.2 GB |
| ViT | 24 layers, d = 1024, patch 16, merge 2, temporal patch 2 | at 448², one step (2 frames) = 784 patches → **196 LM tokens** |
| Cross-frame mixing in ViT | none (per temporal patch) | room for the CST adapter |
| Native rate | 2 fps → 1 step/s → 196 tokens/s | 1 h ≈ 706K tokens: infeasible on 32 GB without memory management |
| Feature hook | `mm_encoder_outputs` / `inputs_embeds` + custom 3D position ids | modules plug in without forking `transformers` |

## 3. Architecture

```text
frames (2 fps) ─> Qwen3.5 ViT (frozen) ─> merger ─> X_t ∈ R^{196×2048}       one step = 2 frames
                                                    │
                     ┌──────────────────────────────┴───────────────┐
                     │ CST adapter (4 layers, d_a = 768, zero-gated) │ <- cache of last K steps
                     │  local temporal attn + memory read + FFN      │ <- transition memory slots
                     └──────────────────────────────┬───────────────┘
                                                    │ H_t (= X_t + gate·Δ, gate init 0)
                     ┌──────────────────────────────┴───────────────┐
                     │ Transition encoder                             │
                     │  predictor  Ĥ_t = P(H_{t-1}, memory)          │
                     │  surprise   s_t,i = ‖H_t,i − Ĥ_t,i‖ · gate     │
                     │  slots      ≤ M surprise-weighted slots        │
                     │  tokens     [before, after, delta] per slot    │
                     └──────────────────────────────┬───────────────┘
              I-step (every N_I s): 196 native tokens│P-step: G gist + 3·m transition tokens
                                                    ▼
<t=00:12:31> timestamp text ─> Qwen3.5 LM (LoRA): 18 GDN layers (state = gist of everything)
                                                   6 full-attn layers, KV = sink ∪ recent window ∪ TKM
```

### 3.1 Causal spatio-temporal adapter (CST): the v1 attention, moved to where it can work

Input: merged tokens X_t (196 × 2048) with grid coordinates (x, y) and timestamp τ_t. Project to
d_a = 768. Each of L_a = 4 layers (pre-norm, residual) applies:

1. **Local causal temporal attention.** Token i at step t attends to tokens j at steps t−K+1 … t
   (K = 16 steps ≈ 16 s) within a spatial window of radius r = 2 on the 14 × 14 grid (≤ 25·K keys).
   Logits get the v1 relative bias b_h = MLP(Δx, Δy, log(1 + Δτ)). This is where correspondence and
   motion are computed; it is cheap and causal.
2. **Memory read.** Cross-attention from tokens to the current transition-memory slots (keys carry
   their own time embedding), so the adapter knows what was already recorded.
3. **FFN** (×4).

Output: H_t = X_t + g ⊙ W_out(·), with **g a per-channel gate initialized to 0** and W_out
zero-initialized. At initialization H_t = X_t exactly (unit test M1).

Parameters ≈ 4 × (2.4M attn + 4.7M FFN + 0.6M read) + 2 × 2048 × 768 (in/out) ≈ **34M** *(est.)*.

### 3.2 Transition encoder (TE): latent P-frames

Per step t:
- **Predictor.** Ĥ_t = P([H_{t−1}; read(memory)]), a 2-layer token-wise MLP plus one local
  attention layer (≈ 8M). Trained to predict the next step (JEPA-style, stop-gradient EMA target).
- **Surprise.** s_t,i = ‖LN(H_t,i) − LN(Ĥ_t,i)‖₂ combined with a learned gate
  σ(w·[H_t,i, H_t,i − Ĥ_t,i]). Optional external signal: the drift of the first GDN layer's state
  (DeltaS) when the previous step was consumed.
- **Slots.** Slot attention with M = 8 slots over tokens weighted by surprise produces m ≤ M active
  slots (mass above a learned threshold, capped by the step budget).
- **Transition tokens.** For each active slot k, three LM-space tokens:
  - before_k = attention-pool of H_{t−K'} (the step before onset, from the CST cache) or of the
    matched memory entry, at the slot's locations;
  - after_k = attention-pool of H_t at the slot's locations;
  - delta_k = MLP(after − before, slot position, Δτ).
  Each carries a type embedding (before / after / delta) and its slot box (x1, y1, x2, y2) through
  a small MLP. Parameters ≈ 10M.
- **Gist.** G = 8 learned queries pool H_t into gist tokens (≈ 3M).
- **Persistence.** Slots are matched to existing memory entries (cosine on after-tokens + box IoU).
  A match updates the entry (still present / reverted); otherwise a new entry is written. This is v2's
  active/resolved status, now learned in feature space.

LLM input per step:
- **I-step** every N_I = 8 s, and whenever global surprise exceeds a threshold (scene cut): all 196
  tokens of H_t, so the pretrained pathway sees full frames regularly.
- **P-step** otherwise: G gist tokens + 3m transition tokens (typically 8–32 tokens, versus 196).
- A text timestamp `<t=hh:mm:ss>` before each step, as in Qwen3-VL. Position ids are contiguous
  (StreamingVLM) with the temporal MRoPE component set from the timestamp.

- **Learning to emit (added after the 4×A100 pilot).** A hard threshold on slot mass gives the gate no
  gradient once every slot is below it, and a one-sided budget (penalize mass above target) drives
  surprise toward exactly that regime, so transitions can silently stop firing. The encoder therefore keeps
  a soft activity p_k = σ((mass_k − τ)/T) per slot. The budget loss is two-sided on the expected P-step
  tokens (gist + 3·Σp_k vs `target_p_step_tokens`, which must lie in [gist, gist + 3·max_active]; 20 is the
  predeclared pilot value), training emits slots by Bernoulli(p_k) sampling, and
  emitted trios carry a straight-through factor (1 + p_k − sg(p_k)), so the task loss also tells the gate
  whether a transition helped. Evaluation emits deterministically (p_k ≥ 0.5, i.e. mass ≥ τ). Training
  and the budget audit report active-slot distributions and CST memory depth, so a comparison where no
  transitions fire is visible as such (it would then test gist pooling vs spatial pooling only).

Token rate *(est.)*: (196 + 7 × 16) / 8 ≈ 39 tokens/s, versus 196: **≈ 5× fewer** before any
eviction.

### 3.3 Transition KV memory (TKM) in the 6 full-attention layers

The 18 GDN layers consume every token and keep their recurrent state; nothing is evicted there. For
the 6 full-attention layers the KV cache is partitioned:

| Partition | Content | Size |
|---|---|---|
| sink | system prompt + first I-step | ~300 tokens |
| recent | everything from the last W = 32 s (I- and P-steps) | ~1.2K tokens |
| transition memory | before/after/delta tokens and I-step gists older than W | budget B_T = 4K tokens |
| dialogue | question/answer text (long text window as in StreamingVLM) | ≤ 1K tokens |

When the transition partition is full, evict by a learned retention score
r = MLP(surprise at write, GDN drift, age, number of later attention hits, slot status), trained by
a distillation signal (Section 4: the teacher's attention mass on that evidence). Total KV ≈ 6.5K
tokens ≈ 80 MB *(est.)*, constant in stream length.

Retrieval is plain attention in those layers: differentiable, no top-k, and no question needed at
write time. That is the key difference from v1's ranking and v2's heuristic sort.

### 3.4 Alert head: Jev-style classification, latently connected (kept from v1)

Surveillance often needs only a discriminative answer: is the scene normal, is there a hazard, which
kind of incident. Decoding text for that is slow and unnecessary. v3 keeps v1's non-autoregressive alert
head and gives it the interface of [Jev](https://docs.typesafe.ai/introduction) (TypeSafe AI's
"System One" decision model, 15 Sep 2026): typed questions registered once, evaluated in parallel and in
isolation against the current state, returning typed values instead of text. We borrow the interface
for classification only; Jev itself is closed and text/JSON-only, and open-ended questions stay with the
LM.

- **Question kinds:** `noul` (yes/no, p ∈ [0, 1]: "Is anything abnormal happening?", "Is there a
  hazard?") and `choice` (one of ≤ 255 labels with probabilities and a separate confidence: "Which
  incident is this?"). Each question and label text is encoded once by the frozen LM in its own pass.
- **Isolation:** each question is its own query (batch item) over the state; there is no attention
  between questions, so adding one never changes another's answer (tested).
- **Latent connections, no text:**
  1. the current step's visual-token hidden states tapped from LM layers {11, 17, 23} (with grid
     positions and a layer embedding);
  2. the Gated DeltaNet recurrent states of layers {10, 18, 22}, read as associative memories with a
     learned, L2-normalized key per question, r = Sᵀk per head, exactly how a GDN layer reads its own
     state. This gives the whole stream's compressed history at constant cost;
  3. a bounded memory (256 tokens) of earlier steps' latent tokens (step ends, gist/after/delta) with
     their age;
  4. transition clocks from the bounded `TransitionTracker` (ages and count of active transitions,
     mean surprise), so duration rules ("bag left > 10 min") are answerable.
- **Cost:** one small cross-attention stack (2 layers, width 512) per step for all registered
  questions; no tokens are generated.
- **Training:** BCE for noul and CE for choice, from per-step labels (`alerts` in records; legacy
  `rules` + `alert_spans` map to noul questions); confidence is trained to predict whether the arg-max
  is right. In training the GDN state is only available at the clip end, via an extra no-grad pass on
  1 in 4 samples, and the state read is dropped with p = 0.3 so the head also works without it.

### 3.5 Parameter budget

| Component | Params *(est.)* | Trainable |
|---|---|---|
| Qwen3.5-2B (ViT ≈ 0.3B + LM ≈ 1.9B incl. embeddings) | ≈ 2.2B | ViT frozen; LM via LoRA |
| CST adapter | ≈ 34M | yes |
| Transition encoder (predictor, slots, token heads, gist) | ≈ 21M | yes |
| Alert head (Jev-style, latent connections) | ≈ 14M | yes |
| Retention scorer | < 1M | yes |
| LoRA (r = 32 on q/k/v/o of full-attn layers and in/out projections of GDN layers) | ≈ 25M | yes |
| **Total** | **≈ 2.29B** | ≈ 90M trainable |

Replace the estimates with `parameter_report()` in M1.

## 4. Training

All stages fit 2 × RTX 5090 (32 GB) or 4 × A100-40GB: bf16, frozen ViT, gradient checkpointing,
LoRA, batch 1 with accumulation.

**Stage 1: transition-preserving distillation (short videos, ≤ 3 min).**
Teacher = frozen Qwen3.5-2B fed **all** 196 tokens per step (≤ ~35K tokens, forward only). Student =
v3 stream (I/P steps, bounded KV). Losses:
- L_KD: KL(teacher ‖ student) on answer/caption tokens of video QA and captioning data;
- L_pred: predictor regression (stop-gradient EMA target); trains the surprise signal;
- L_trans: InfoNCE matching each delta token to its true (before, after) pair against in-batch and
  shuffled-time negatives;
- L_vis: hinge so the answer is likelier with the true stream than with another video's stream (the
  v1 failure check, as a training signal);
- L_budget: penalty on tokens per step above the target rate.

**Stage 2: streaming SFT (long streams, 10–60 min).**
Concatenated clips, MEVA streams, and synthetic long-gap tasks with questions inserted at random
times ("is X still there?", "how long ago did Y change?", "how many times …?"). Train on overlapping
chunks with the inference KV policy applied (StreamingVLM recipe), carrying the GDN state across
chunks with truncated backprop. The retention scorer is supervised by the teacher's attention mass on
evicted vs kept evidence for answers in the chunk.

**Stage 3 (optional).** Answer-level preference optimization on long-gap questions.

Data, licensing and volumes are in `docs/v3/experiments.md`.

## 5. Claimed contributions (to be earned by the ablations)

1. **Latent P-frame tokenization.** Surprise-gated before / after / delta tokens learned in the VLM's
   own feature space: a streaming input format between "every frame" and "drop frames".
2. **Transition KV memory for hybrid VLMs.** Linear-recurrent layers keep the gist; the
   full-attention layers keep only transition evidence under a fixed budget, with learned retention.
3. **Causal spatio-temporal adapter with zero-gated insertion,** giving cross-frame correspondence
   to a backbone whose ViT has none, without disturbing it at initialization.
4. **Full-stream → bounded-stream distillation recipe** that trains compression, retention and
   surprise jointly, with a visual-dependence loss.

Positioning:
- StreamingVLM keeps a window; we add a learned, bounded transition store.
- DeltaS evicts chunks by GDN drift without training; we write token-level transitions and learn
  retention (DeltaS is a baseline).
- StreamTTT and video-SALMONN S store history in fast weights; ours is explicit, inspectable token
  memory (a combined variant is future work).
- Codec methods (CoPE-VideoLM, CodecSight) trigger on codec signals; ours triggers on learned
  feature surprise and carries an explicit "before".

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Qwen3.5 needs `transformers` main and GDN kernels (`flash-linear-attention`) on sm_120 (5090) | M0 pins a commit and benchmarks kernels; the A100s are the fallback |
| Custom position ids / streaming cache diverge from the reference forward | M1 equality test: zero gates + all-I-steps must reproduce the base model's logits |
| Surprise collapses (predictor learns the identity) | EMA target, stop-gradient, L_budget lower bound; monitor the surprise histogram |
| Gains come from the extra I-steps, not transitions | ablation with the same tokens spent on pooled tokens (B4) |
| GDN truncated backprop limits long-range credit | intermediate questions across chunks; report the gap-length curve |
| Compute: Stage 1 teacher forwards on long clips | cache teacher logits for answer tokens once; reuse across epochs |
