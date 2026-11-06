# Figure captions (CVPR/NeurIPS, full width `\linewidth` in `figure*`)

Regenerate with `uv run --with matplotlib python scripts/figures/paper_figures.py`. Files: vector PDF
(TrueType fonts embedded, `pdf.fonttype 42`), SVG, and 300 dpi PNG. Include the PDF in LaTeX:
`\includegraphics[width=\linewidth]{figures/paper/fig1_v2_transition_memory.pdf}`.

Panels marked illustrative need trained weights or real data before submission: replace them with
measured values, or keep the word "illustrative" in the caption.

## Figure 1 — `fig1_v2_transition_memory`

**Transition evidence memory (TRACE-VL v2).** (a) Causal change detection with the default
constants. A slow background ($\tau=600$ s, frozen inside active regions) and a fast background
($\tau=4$ s) are maintained per pixel. A pixel that differs from the slow but not from the fast
background accumulates static evidence; after 6 s the region is confirmed at $t_\mathrm{known}$,
with onset $t_\mathrm{start}$ taken from the start of the continuous difference. Reversion for 4 s
confirms the end at $t^\mathrm{known}_\mathrm{end}$. The strip below shows what the memory exposes
over time: nothing before confirmation, an active transition with unknown end, then a resolved one.
(b) The memory unit: before/after/after-end crops of the changed region with onset, confirmation and
end times. (c) Bounded store and causal snapshot at query time $t_q$: transitions are visible only
after $t_\mathrm{known}$, and endings remain hidden until confirmed (hatched). When full, the store
evicts resolved transitions first, then absorbed and scene-wide ones, and active transitions last
(right column). Curves in (a) are the analytic response to an ideal object placed at 10 s and removed
at 40 s.

## Figure 2 — `fig2_v2_readout`

**Query-conditioned readout under a fixed visual-token budget.** (a) Transitions are sorted
lexicographically by overlap with the query region, whether they end inside the question window,
overlap fraction and recency. (b) Every method receives the same current frame (256 tokens) and the
same evidence budget $B$. Frame baselines spread $B$ over 8 frames; the transition memory spends it
on before/after crops (64 tokens each); text-only variants use no extra visual tokens. Bars show the
planned allocation for $B=1024$. (c) Evidence is interleaved with its time and location text,
followed by the current frame and the question, and a pretrained Qwen3-VL (LoRA on the language
model) answers by the log-probability of the option letters in one forward pass. Attention mass and
probabilities in (c) are illustrative.

## Figure 3 — `fig3_v1_attention`

**Causal spatio-temporal attention (TRACE-VL v1).** (a) Each of the 4 pre-LN blocks applies spatial
self-attention within frame $t$, then temporal attention whose keys and values are the cached
projections of the last $K=8$ frames together with the 32 working-state slots $S_{t-1}$, plus a
learned relative bias $\mathbf{B}$ from $(\Delta x,\Delta y,\log(1+\Delta t))$ and a causal mask
$\mathbf{M}$. (b) Temporal mask for frame $t$: cached frames older than $\tau_c=4$ s are masked; the
state keys are always visible so no query row is fully masked. (c) Shape of the learned per-head
bias as a function of $\Delta t$. Values in (b) and curves in (c) are illustrative.

## Figure 4 — `fig4_v1_memory`

**Working state and transition capsules (TRACE-VL v1).** (a) 32 state slots attend to the previous
state and the current frame tokens; a sigmoid gate mixes the proposal $P$ with $S_{t-1}$. (b) Each
capsule has four latent tokens seeded from pooled states (before, change, after, detail) that attend
to restricted parts of $[S_{t-1}; X'_t; S_t]$: before reads only $S_{t-1}$, after only $S_t$,
detail only $X'_t$, and change reads all three. (c) Capsules are written to a 128-slot event bank
(FIFO or utility-based replacement). At query time the top-$k=16$ capsules known by $t_q$ are
retrieved by $q^\top\bar c_i/\sqrt d$, given a time embedding (age, span, before-valid) and read with
the current tokens and $S_t$ by a 2-layer resampler of 96 queries, then projected into the LLM
prefix.
