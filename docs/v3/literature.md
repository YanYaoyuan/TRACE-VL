# v3 literature and industrial survey (checked 2026-11-06)

Scope: memory for streaming video LLMs, memory modules in LLMs, hybrid linear-attention backbones,
and recent small open VLMs; emphasis on May–November 2026. Entries marked **read** were checked at
abstract/model-card level for this survey. Entries marked **to read** are listed so they are not
missed; do not cite their claims until read.

PDFs are **not** committed: arXiv's default license does not permit redistribution. Run
`bash scripts/fetch_papers.sh` to download the PDFs listed in `docs/v3/papers.txt` into the
git-ignored `papers/` folder.

## 1. Backbone: Qwen3.5-2B (industrial, open, Apache-2.0) — read

- Model card: <https://huggingface.co/Qwen/Qwen3.5-2B>; code: `transformers` main,
  `models/qwen3_5/modeling_qwen3_5.py` (`Qwen3_5ForConditionalGeneration`).
- Language model: 24 layers, d = 2048, FFN 6144; **hybrid**: 6 × (3 × Gated DeltaNet + 1 × gated
  full attention). GDN: 16 heads, head dim 128, conv kernel 4. Full attention: 8 Q heads, 2 KV heads,
  head dim 256, partial RoPE (0.25), interleaved MRoPE sections [11, 11, 10]. 262K context, tied
  embeddings, 1 MTP layer.
- Vision: 24-layer ViT, d = 1024, patch 16, spatial merge 2, temporal patch 2, output 2048. Early
  text–vision fusion. Video input supported (default 2 fps).
- Verified in code: the ViT attends **within each temporal patch only** (`get_vision_attention_seqlens`
  with `merge_temporal=False`), so there is no cross-frame interaction before the LLM. No DeepStack
  path in Qwen3.5's forward. Visual features enter via `masked_scatter` into `inputs_embeds`, and
  `mm_encoder_outputs` accepts precomputed features: a clean hook for our modules.
- Reported (thinking mode): Video-MME 69.0 (w/o subs), MLVU 76.2, MVBench 64.9, LVBench 57.1.
- Why it matters: 18 of 24 LM layers keep a **fixed-size recurrent state**. Streaming memory cost is
  dominated by the 6 full-attention layers' KV cache (2 KV heads × 256 dims): about 12 KB per token in
  bf16, so 100K cached tokens ≈ 1.2 GB.

Also relevant: Qwen3-VL report (arXiv 2511.21631; interleaved MRoPE, DeepStack, text timestamps
before each temporal patch) — **read** (secondary summaries + abstract).

## 2. Streaming video memory (2025-10 → 2026-09)

| Paper | Date | Memory idea | Trained? | What we take / how we differ | Status |
|---|---|---|---|---|---|
| [StreamingVLM](https://arxiv.org/abs/2510.09608) (MIT/NVIDIA, ICLR 2026) | 2025-10 | attention sink + short vision window (~16 s) + long text window; contiguous position ids; SFT on short overlapped chunks that mimics inference | yes | baseline B1; reuse its position scheme and chunked SFT recipe | read |
| [video-SALMONN S](https://arxiv.org/abs/2510.11129) | 2025-10 / 2026-02 | test-time-training layer as long-term memory; modality-aware reader; 3 h at 1 fps | yes | parametric memory alternative; stretch goal, not core | read |
| [CurveStream](https://arxiv.org/abs/2603.19571) | 2026-03 | curvature of the feature trajectory marks semantic transitions; clear/fuzzy memory | no | supports "transitions are where information is"; our surprise is learned and token-level | read |
| [Tempo](https://arxiv.org/abs/2604.08120) (Meta) | 2026-04 | small VLM (Qwen3-VL-2B) as query-aware local compressor, 0.5–16 tok/frame | yes | compression target numbers; ours is query-agnostic (must stream before the question) | read |
| [How Well Can Your Video Model Remember?](https://arxiv.org/abs/2606.20726) | 2026-06 | law: logit-accuracy ∝ log frame budget, slope α(D) decays with distance | — | adopt α(D) as a reported metric | read |
| [FOLIO](https://arxiv.org/abs/2607.13298) | 2026-07 | entity-centric semantic memory + evidence cache, training-free; Qwen3-VL-8B: OVO 82.0 perception / 69.1 backward, StreamingBench 74.5 | no | external reference numbers (8B) | read |
| [StreamFlow](https://arxiv.org/abs/2608.10949) | 2026-08 | dynamics-aware frame filtering (mid-term) + latent long-term memory injected when visual reliance drops; StreamingBench 67.73 | unclear | reference; our filtering is token-level and learned | read |
| [StreamTTT](https://arxiv.org/abs/2608.13416) (NUS Show Lab) | 2026-08 | long history in online-updated fast weights + short sliding KV; 4B | yes | closest competitor at our scale; report against it | read |
| [Beyond Retrieval / LatentStream](https://arxiv.org/abs/2609.04131) | 2026-09 | hierarchical short/mid/long memory, query-time latent memory evolution | yes | query-time refinement idea; later work | read |
| [DeltaS](https://arxiv.org/abs/2609.27470) | 2026-09 | **gated-delta state drift** decides which video chunks to keep in the full-attention KV of a hybrid VLM; training-free; +2.1 avg over best query-agnostic bounded baseline | no | directly relevant to Qwen3.5; baseline B2 and an input to our write/retain score | read |
| [MemEvo](https://arxiv.org/abs/2609.36581) | 2026-09 | LLM-driven search over streaming memory mechanisms | no | — | to read |
| [Harnessing Streaming Video in the Wild](https://arxiv.org/abs/2606.08615) | 2026-06 | — | — | — | to read |
| [Watch, Remember, Reason](https://arxiv.org/abs/2606.07433) (survey) | 2026-06 | taxonomy of streaming/offline memory | — | related-work structure | to read |
| StreamMem (arXiv 2508.15717), HERMES (arXiv 2601.14724) | 2025–26 | KV compression for streaming | — | — | to read |

## 3. Memory modules in LLMs

| Paper | Date | Idea | Relevance | Status |
|---|---|---|---|---|
| [Titans](https://arxiv.org/abs/2501.00663) / ATLAS (arXiv 2505.23735) | 2025 | neural long-term memory updated at test time by surprise (gradient of a reconstruction loss) | "surprise-gated write" principle | read (prior knowledge) |
| [Titans Revisited](https://arxiv.org/abs/2510.09551) | 2025-10 | reimplementation: memory updates alone insufficient for meaningful test-time learning | caution against TTT as the only memory | read |
| [Titan-Llama](https://arxiv.org/abs/2610.19210) (Behrouz et al.) | 2026-10 | retrofit pretrained LLM: full attention on recent tokens, older context compressed into recurrent neural memory modules; Llama-3.2-1B QuALITY 23.9→40.5 | retrofitting recipe for a pretrained backbone | read |
| [Titans-as-a-Layer](https://arxiv.org/abs/2606.08573) | 2026-06 | plug-in adapter writes history into small neural memory, reads back as residual; host model unchanged | zero-impact plug-in design pattern | read |
| [Test-time regression](https://arxiv.org/abs/2501.12352) | 2025 | unifies linear attention / delta rule / TTT as associative-memory regression | theory for treating GDN state as memory | read (prior knowledge) |
| [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) (NVIDIA) | 2026-05 | decoupled channel-wise erase and write gates; best on multi-key RULER at 1.3B | erase/write decoupling for our memory update | read |
| [Short window attention enables long-term memorization](https://arxiv.org/abs/2509.24552) | 2025-09 | SWA + recurrent hybrids | supports small windows in full-attn layers | to read |
| [Distilling to hybrid attention via KL-guided layer selection](https://arxiv.org/abs/2512.20569) | 2025-12 | distillation into hybrids | distillation recipe reference | to read |
| [Memoir](https://arxiv.org/abs/2607.20792) | 2026-07 | should a model write to memory while it thinks | — | to read |

## 4. Hybrid / linear attention for video

- [Video DeltaNet](https://arxiv.org/abs/2609.20744) (2026-09): local softmax for nearby frames +
  delta-rule linear memory for distant context, post-training adaptation (generation). — read
- [SANA-Video 2.0](https://arxiv.org/abs/2607.21553) (2026-07): hybrid gated linear + softmax video DiT. — read
- [FG²-GDN](https://arxiv.org/abs/2604.19021) (2026-04): fine-grained GDN control, 3:1 hybrid with MLA. — read

Consensus pattern: local softmax attention for recent detail plus a delta-rule state for long range.
Qwen3.5 already has this in its LM; our contribution has to be about **what** is written and kept.

## 5. State-change / transition evaluation

- [TOC-Bench](https://arxiv.org/abs/2605.09904) (2026-05): temporal object consistency across
  occlusion, disappearance and state transitions; 2,323 human-verified QA. — read
- [StateTrace / HSR-Bench](https://arxiv.org/abs/2608.18532) (2026-08): hidden-state reasoning while an
  object is out of view; VideoLLaMA3 39.6 → 64.2 with their method. — read
- ProcObject-10K ([arXiv 2512.03479](https://arxiv.org/abs/2512.03479)): object state across
  procedural steps; models shortcut via action patterns. — read
- Our MEVA long-gap set (`docs/inspection.md`).

## 6. Codec / transition-token video tokenization (prior art for "latent P-frames")

From `docs/references.bib`: CoPE-VideoLM, CodecSight, OneVision-Encoder (codec-aligned sparsity),
CoVisco, codec-guided training-free compression, Mage-VL. These use **codec** motion vectors and
residuals. v3's transition tokens are learned in the VLM's own feature space and carry an explicit
"before" from memory; this distinction must be shown experimentally (ablation: pixel-codec trigger
vs learned surprise).

## Takeaways for the design

1. Use Qwen3.5-2B: native video, hybrid GDN state = free long-range gist; we only manage the 6
   full-attention layers' KV.
2. Recent streaming work converges on **sink + short window + selective long-term store**. The open
   question is *what* the long-term store holds. Our answer: before/after/delta transition tokens
   triggered by predictive surprise and backbone state drift.
3. Strongest same-scale competitor: StreamTTT-4B (fast-weight memory). Strongest training-free
   hybrid baseline: DeltaS. Both must appear in our tables.
4. Zero-initialized, plug-in modules (Titans-as-a-Layer, Titan-Llama) avoid v1's failure: the model
   starts exactly at Qwen3.5-2B and can only move away from it if the loss improves.
