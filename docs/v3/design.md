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

### 3.4 Non-autoregressive heads: alerts and event embeddings (kept from v1, extended)

Surveillance needs a decision every step, not a sentence. Decoding text is slow and unnecessary for
"should this fire?", so v3 keeps v1's alert head and adds a VL-JEPA-style embedding head
([VL-JEPA](https://arxiv.org/abs/2512.10942), ICLR 2026: predict text embeddings instead of tokens and
decode selectively). Both read the LM hidden state at each step's `<|vision_end|>` token, which the
streaming forward computes anyway, so they cost one small MLP per step and no decoding.

- **Rule-conditioned alert head.** Rules are written into the prompt ("Monitoring rules: 1. … 2. …"),
  so all 24 LM layers condition the stream on them, and the GDN state is rule-aware. For rule r the
  query is the mean prompt hidden state over that rule's tokens:
  logit_r(t) = MLP(W_s h_t ⊙ W_r q_r + W_c c_t), where c_t holds transition clocks (log ages of active
  transitions, their count, mean surprise). The clocks let duration rules ("bag left > 10 min")
  fire from memory rather than from the current frame. Many rules are scored in one pass.
- **Event embedding head (JEPA-style).** e_t = normalize(MLP(h_t)) is trained with InfoNCE to match
  the embedding of a caption covering t (teacher LM mean hidden state, through a small text
  projection). Uses: (1) zero-shot rules by cosine(e_t, embed(rule text)) when no labels exist;
  (2) **selective decoding**: generate a description only when 1 − cos(e_t, e_last_decoded) > τ, as
  VL-JEPA does for streams.
- Training: BCE on per-step rule labels where they exist (MEVA activities as rules; anomaly datasets
  with frame labels, licence permitting); InfoNCE on captioned clips. Both heads train jointly with
  Stage 1; the transition clocks are wired in the streaming session and are zero in the Stage 1
  trainer (TODO: compute them from the transition encoder during training).

### 3.5 Parameter budget

| Component | Params *(est.)* | Trainable |
|---|---|---|
| Qwen3.5-2B (ViT ≈ 0.3B + LM ≈ 1.9B incl. embeddings) | ≈ 2.2B | ViT frozen; LM via LoRA |
| CST adapter | ≈ 34M | yes |
| Transition encoder (predictor, slots, token heads, gist) | ≈ 21M | yes |
| Alert head + event head | ≈ 9M | yes |
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
