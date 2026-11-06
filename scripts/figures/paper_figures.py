"""Paper figures (CVPR/NeurIPS style) for the TRACE-VL attention and memory modules.

Run without adding a project dependency:
    uv run --with matplotlib python scripts/figures/paper_figures.py
Writes figures/paper/fig{1..4}_*.{pdf,svg,png}. Numbers shown come from the code defaults; panels
that would need trained weights (attention/bias values) are marked "illustrative".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "figures" / "paper"
FULL = 6.9  # inches, two-column full width

INK, MUTED = "#222222", "#6B6B6B"
BLUE, LBLUE = "#3B6EA8", "#DCE8F5"
GREEN, LGREEN = "#2F7D4F", "#DDEFE3"
ORANGE, LORANGE = "#C46A00", "#FBE6CC"
RED, LRED = "#B03A2E", "#F5D5D1"
PURPLE, LPURPLE = "#6B4BA1", "#E7DFF3"
GRAY, LGRAY = "#8A8A8A", "#EEEEEE"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "mathtext.fontset": "stixsans",
        "pdf.fonttype": 42,  # embed TrueType, as required by CVPR/NeurIPS
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "hatch.linewidth": 0.5,
    }
)


# ----------------------------------------------------------------------------- helpers
def canvas(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    return ax


def panel(ax, letter, title, x=-0.02, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    ax.annotate(title, xy=(x, y), xycoords="axes fraction", xytext=(15, 0), textcoords="offset points",
                fontsize=7.5, va="bottom", ha="left")  # fmt: skip


def rbox(ax, x, y, w, h, label="", fc=LGRAY, ec=GRAY, fs=6.5, lw=0.7, color=INK, bold=False, pad=0.012, ls="-",
         align="center", z=2):  # fmt: skip
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z
        )
    )
    if label:
        tx = x + w / 2 if align == "center" else x + 0.012
        ax.text(tx, y + h / 2, label, ha=align, va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=z + 1, linespacing=1.25)  # fmt: skip


def arrow(ax, p0, p1, color=INK, lw=0.7, style="-|>", ls="-", rad=0.0, z=1, ms=5):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0)  # fmt: skip
    )


def scene(obj=False, h=36, w=60, light=1.0):
    """Synthetic fixed-camera frame: wall, floor, door, optional object."""
    img = np.ones((h, w, 3)) * np.array([0.86, 0.88, 0.90]) * light
    img[int(h * 0.56) :, :] = np.array([0.73, 0.76, 0.80]) * light
    img[int(h * 0.15) : int(h * 0.56), int(w * 0.08) : int(w * 0.24)] = np.array([0.55, 0.43, 0.32])
    if obj:
        img[int(h * 0.62) : int(h * 0.82), int(w * 0.55) : int(w * 0.70)] = np.array([0.69, 0.23, 0.18])
    return img


def thumb(ax, x, y, w, h, obj=False, label=None, region=None, light=1.0, fs=6):
    ax.imshow(scene(obj, light=light), extent=(x, x + w, y, y + h), aspect="auto", zorder=3, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=MUTED, lw=0.5, zorder=4))
    if region is not None:
        rx, ry, rw, rh = region
        ax.add_patch(Rectangle((x + rx * w, y + ry * h), rw * w, rh * h, fill=False, ec="#D00000", lw=0.8, zorder=5))
    if label:
        ax.text(x + w / 2, y - 0.02, label, ha="center", va="top", fontsize=fs, color=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 1
def fig1_transition_memory():
    """v2: causal transition detection, the memory unit, bounded store with causal snapshot."""
    fig = plt.figure(figsize=(FULL, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.8, 1.3], wspace=0.25)

    # (a) timing diagram with the default detector constants
    sub = grid[0].subgridspec(3, 1, height_ratios=[0.55, 2.1, 0.36], hspace=0.12)
    top, ax, strip = (fig.add_subplot(s) for s in sub)
    tau_fast, thr, confirm, vanish = 4.0, 25.0, 6.0, 4.0
    t = np.linspace(0, 60, 2401)
    present = (t >= 10) & (t < 40)
    d_slow = 150.0 * present
    d_fast = np.where(t < 10, 0, np.where(t < 40, 150 * np.exp(-(t - 10) / tau_fast), 150 * np.exp(-(t - 40) / tau_fast)))
    t_static = 10 + tau_fast * np.log(150 / thr)
    t_start, t_known, t_end, t_end_known = 10.0, t_static + confirm, 40.0, 40.0 + vanish
    ax.fill_between(t, 0, 150, where=(t >= t_static) & (t < 40), color=LGREEN, lw=0, zorder=0)
    ax.text((t_static + 40) / 2, 138, "static evidence", ha="center", va="top", fontsize=6, color=GREEN)
    ax.plot(t, d_slow, color=BLUE, lw=1.1, label=r"$|I_t-B^{\mathrm{slow}}|$")
    ax.plot(t, d_fast, color=ORANGE, lw=1.1, label=r"$|I_t-B^{\mathrm{fast}}|$")
    ax.axhline(thr, color=GRAY, lw=0.6, ls=(0, (3, 2)))
    ax.text(59.5, thr + 3, r"$\delta$", ha="right", va="bottom", fontsize=7, color=MUTED)
    marks = ((t_start, r"$t_{\mathrm{start}}$", "center"), (t_known, r"$t_{\mathrm{known}}$", "center"),
             (t_end, r"$t_{\mathrm{end}}$", "right"), (t_end_known, r"$t_{\mathrm{end}}^{\mathrm{known}}$", "left"))  # fmt: skip
    for x, name, ha in marks:
        ax.axvline(x, color=INK, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
        ax.text(x + {"center": 0, "right": -0.6, "left": 0.6}[ha], 157, name, ha=ha, va="bottom", fontsize=6.5)
    ax.annotate("", xy=(t_known, 112), xytext=(t_static, 112), arrowprops=dict(arrowstyle="<->", lw=0.6, color=GREEN))
    ax.text((t_static + t_known) / 2, 115, "6 s", ha="center", va="bottom", fontsize=6, color=GREEN)
    ax.annotate("", xy=(t_end_known, 112), xytext=(t_end, 112), arrowprops=dict(arrowstyle="<->", lw=0.6, color=BLUE))
    ax.text(t_end_known + 0.8, 112, "4 s", ha="left", va="center", fontsize=6, color=BLUE)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 172)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("pixel difference", labelpad=1)
    ax.set_xticklabels([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), handlelength=1.3, borderaxespad=0.1)
    canvas(top, (0, 60), (0, 1))
    for center, obj in ((5, False), (25, True), (50, False)):
        thumb(top, center - 5.5, 0.08, 11, 0.86, obj)
    panel(top, "a", "Causal change detection (default constants)", y=1.06)
    strip.set_xlim(0, 60)
    strip.set_ylim(0, 1)
    strip.add_patch(Rectangle((0, 0.12), t_known, 0.76, fc="white", ec=GRAY, lw=0.5, hatch="////"))
    strip.add_patch(Rectangle((t_known, 0.12), t_end_known - t_known, 0.76, fc=LGREEN, ec=GREEN, lw=0.6))
    strip.add_patch(Rectangle((t_end_known, 0.12), 60 - t_end_known, 0.76, fc=LBLUE, ec=BLUE, lw=0.6))
    for x0, x1, label, color in ((0, t_known, "not in memory", MUTED), (t_known, t_end_known, "active", GREEN),
                                 (t_end_known, 60, "resolved", BLUE)):  # fmt: skip
        strip.text((x0 + x1) / 2, 0.5, label, ha="center", va="center", fontsize=6, color=color,
                   bbox=dict(fc="white", ec="none", pad=0.6) if color == MUTED else None)  # fmt: skip
    strip.set_yticks([])
    strip.set_xlabel("time (s)", labelpad=1)
    strip.spines[["top", "right", "left"]].set_visible(False)
    strip.set_ylabel("memory", rotation=0, ha="right", va="center", labelpad=2, fontsize=6.5)

    # (b) the memory unit
    ax = canvas(fig.add_subplot(grid[1]))
    panel(ax, "b", "Memory unit")
    rbox(ax, 0.0, 0.0, 1.0, 0.98, fc="#FAFBFD", ec=BLUE, lw=0.8)
    for i, (obj, lab) in enumerate(((False, "before"), (True, "after"), (False, "after end"))):
        thumb(ax, 0.05 + i * 0.315, 0.68, 0.27, 0.23, obj, lab)
    rows = [
        ("box", r"$(x_1,y_1,x_2,y_2)$"),
        (r"$t_{\mathrm{start}}$", "10.0 s  onset"),
        (r"$t_{\mathrm{known}}$", f"{t_known:.1f} s  visible"),
        (r"$t_{\mathrm{end}}$", "40.0 s  reverted"),
        (r"$t_{\mathrm{end}}^{\mathrm{known}}$", "44.0 s  visible"),
        ("status", "active→resolved"),
    ]
    ax.plot([0.05, 0.95], [0.54, 0.54], color=GRAY, lw=0.4)
    for i, (key, value) in enumerate(rows):
        y = 0.47 - i * 0.08
        ax.text(0.07, y, key, fontsize=6.5, va="center")
        ax.text(0.34, y, value, fontsize=6.1, va="center", color=BLUE)

    # (c) bounded store and causal snapshot
    ax = fig.add_subplot(grid[2])
    panel(ax, "c", r"Bounded store; snapshot at $t_q$")
    tq = 70
    items = [  # name, start, known, end, end_known, kind, evict rank
        ("T1", 5, 11, 30, 34, "region", 1),
        ("T2", 20, 26, None, None, "region", 3),
        ("T3", 50, 50, 50, 50, "global", 2),
        ("T4", 40, 46, 66, 72, "region", 4),
        ("T5", 55, 61, None, None, "region", 5),
        ("T6", 64, 74, None, None, "region", None),
    ]
    for row, (name, s, k, e, ek, kind, rank) in enumerate(items):
        y = len(items) - row
        if kind == "global":
            ax.plot([s, s], [y - 0.3, y + 0.3], color=PURPLE, lw=1.4)
            ax.text(s - 1.2, y, "scene-wide reset", fontsize=5.8, va="center", ha="right", color=PURPLE)
        else:
            ax.plot([s, k], [y, y], color=GRAY, lw=0.8, ls=(0, (1, 1)))
            known_end = e if (e is not None and ek <= tq) else tq
            if k <= tq:
                resolved = e is not None and ek <= tq
                ax.add_patch(Rectangle((k, y - 0.28), known_end - k, 0.56, fc=LBLUE if resolved else LGREEN,
                                       ec=BLUE if resolved else GREEN, lw=0.6))  # fmt: skip
            if e is not None and ek > tq:
                ax.add_patch(Rectangle((e, y - 0.28), 80 - e, 0.56, fc="white", ec=GRAY, lw=0.5, hatch="////"))
            if k > tq:
                ax.add_patch(Rectangle((k, y - 0.28), 80 - k, 0.56, fc="white", ec=GRAY, lw=0.5, hatch="////"))
            ax.plot(s, y, marker="|", color=INK, ms=4, mew=0.8)
        ax.text(-1.5, y, name, ha="right", va="center", fontsize=6.5)
        ax.text(86, y, "–" if rank is None else str(rank), ha="center", va="center", fontsize=6.5,
                color=RED if rank == 1 else INK)  # fmt: skip
    ax.axvline(tq, color=RED, lw=0.9, ymax=0.8)
    ax.text(tq, len(items) + 0.55, r"$t_q$", color=RED, ha="center", va="bottom", fontsize=7)
    ax.text(86, len(items) + 0.55, "evict", ha="center", va="bottom", fontsize=6)
    ax.set_xlim(0, 89)
    ax.set_ylim(0.3, len(items) + 2.9)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("stream time (s)", labelpad=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [
        Rectangle((0, 0), 1, 1, fc=LGREEN, ec=GREEN, lw=0.6, label="active"),
        Rectangle((0, 0), 1, 1, fc=LBLUE, ec=BLUE, lw=0.6, label="resolved"),
        Rectangle((0, 0), 1, 1, fc="white", ec=GRAY, lw=0.5, hatch="////", label="not yet known"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)  # fmt: skip
    save(fig, "fig1_v2_transition_memory")


# ----------------------------------------------------------------------------- figure 2
def fig2_readout():
    """v2: query-conditioned selection, equal-budget evidence, interleaved sequence and scoring."""
    fig = plt.figure(figsize=(FULL, 2.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.0, 1.55], wspace=0.42)

    # (a) selection on the frame
    ax = canvas(fig.add_subplot(grid[0]))
    panel(ax, "a", "Relevance ranking")
    ax.imshow(scene(True, h=72, w=120), extent=(0.0, 1.0, 0.38, 0.96), aspect="auto", interpolation="nearest")
    ax.add_patch(Rectangle((0.48, 0.44), 0.32, 0.3, fill=False, ec="#D00000", lw=1.0))
    ax.text(0.80, 0.755, "query region", color="#D00000", fontsize=6, ha="right", va="bottom")
    boxes = [((0.53, 0.49, 0.2, 0.15), "1"), ((0.6, 0.57, 0.13, 0.1), "2"), ((0.07, 0.42, 0.2, 0.42), "3"),
             ((0.86, 0.47, 0.1, 0.18), "4")]  # fmt: skip
    for (x, y, w, h), k in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=BLUE, lw=0.7, ls=(0, (2, 1))))
        ax.text(x + 0.012, y + h - 0.012, k, color="white", fontsize=5.5, va="top", ha="left",
                bbox=dict(boxstyle="circle,pad=0.12", fc=BLUE, ec="none"))  # fmt: skip
    ax.text(0.0, 0.3, "Sort key (descending)", fontsize=6.3, fontweight="bold")
    keys = ["1. overlaps the query region", "2. ends inside window $W$", "3. overlap fraction",
            r"4. recency of $t_{\mathrm{known}}$"]  # fmt: skip
    for i, k in enumerate(keys):
        ax.text(0.0, 0.225 - i * 0.068, k, fontsize=6.1)

    # (b) equal-budget visual tokens
    ax = fig.add_subplot(grid[1])
    panel(ax, "b", r"Equal budget ($B=1024$)", x=-0.42)
    methods = ["none", "current", "uniform", "recent", "textlog", "transition", "trans. after", "trans. meta"]
    frame = np.array([0, 256, 256, 256, 256, 256, 256, 256])
    evidence = np.array([0, 0, 960, 960, 0, 960, 960, 0])  # 8 frames x 120 tok; 8 x 2 crops / 16 x 1 crop x 60 tok
    text_only = [0, 0, 0, 0, 1, 0, 0, 1]
    fills = [LGRAY, LGRAY, LBLUE, LBLUE, LGRAY, LORANGE, LORANGE, LGRAY]
    edges = [GRAY, GRAY, BLUE, BLUE, GRAY, ORANGE, ORANGE, GRAY]
    y = np.arange(len(methods))[::-1]
    ax.barh(y, frame, color="white", ec=GRAY, lw=0.5, height=0.62)
    ax.barh(y, evidence, left=frame, color=fills, ec=edges, lw=0.5, height=0.62)
    for yi, flag, left in zip(y, text_only, frame):
        if flag:
            ax.barh(yi, 260, left=left, color="white", ec=GRAY, lw=0.5, hatch="////", height=0.62)
            ax.text(left + 285, yi, "text only", va="center", fontsize=5.8, color=MUTED)
    ax.axvline(256 + 1024, color=RED, lw=0.7, ls=(0, (3, 2)))
    ax.text(256 + 1024 - 20, len(methods) - 0.4, "256 + B", color=RED, ha="right", va="bottom", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.get_yticklabels()[5].set_fontweight("bold")
    ax.set_xlim(0, 1400)
    ax.set_xticks([0, 256, 768, 1280])
    ax.set_ylim(-0.6, len(methods) - 0.1)
    ax.set_xlabel("visual tokens / question", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) interleaved sequence and option scoring
    ax = canvas(fig.add_subplot(grid[2]))
    panel(ax, "c", "Interleaved evidence and option scoring")
    segs = [("hdr", LGRAY, GRAY, 0.10), ("chg 1\ntext", LGRAY, GRAY, 0.35), ("before", LORANGE, ORANGE, 0.85),
            ("after", LORANGE, ORANGE, 1.0), ("…", "white", "white", 0), ("frame\n$t_q$", LBLUE, BLUE, 0.65),
            ("Q +\nopts", LGRAY, GRAY, 0.5), ("A?", LRED, RED, None)]  # fmt: skip
    w, gap, x = 0.105, 0.017, 0.0
    for label, fc, ec, mass in segs:
        rbox(ax, x, 0.54, w, 0.15, label, fc=fc, ec=ec, fs=5.2, pad=0.008)
        if mass:
            ax.add_patch(Rectangle((x + 0.02, 0.71), w - 0.04, 0.17 * mass, fc=RED, ec="none", alpha=0.55))
        x += w + gap
    ax.plot([0, x - gap], [0.71, 0.71], color=GRAY, lw=0.4)
    ax.text(0.0, 0.93, "attention mass from the answer slot (illustrative)", fontsize=5.8, color=MUTED)
    ax.plot([0.0, x - gap], [0.5, 0.5], color=GRAY, lw=0.5)
    rbox(ax, 0.1, 0.1, 0.42, 0.28, "Qwen3-VL\n(pretrained,\nLoRA on the LM)", fc=LORANGE, ec=ORANGE, fs=6.2)
    arrow(ax, (0.31, 0.5), (0.31, 0.38))
    arrow(ax, (0.53, 0.24), (0.66, 0.24))
    ax.text(0.595, 0.27, "logits", fontsize=5.6, ha="center", va="bottom", color=MUTED)
    bars = ax.inset_axes([0.76, 0.08, 0.22, 0.32])
    bars.bar([0, 1], [0.86, 0.14], color=[LRED, LGRAY], ec=[RED, GRAY], lw=0.5, width=0.6)
    bars.set_xticks([0, 1])
    bars.set_xticklabels(["A", "B"])
    bars.set_ylim(0, 1)
    bars.set_yticks([0, 1])
    bars.set_title(r"$p$(option letter)", fontsize=5.8, pad=2)
    bars.tick_params(labelsize=5.8)
    bars.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_v2_readout")


# ----------------------------------------------------------------------------- figure 3
def fig3_v1_attention():
    """v1: causal spatio-temporal block, its temporal attention mask, and the relative bias."""
    fig = plt.figure(figsize=(FULL, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.25, 0.85], wspace=0.36)

    # (a) block diagram (pre-LN, residual around each sub-layer)
    ax = canvas(fig.add_subplot(grid[0]))
    panel(ax, "a", "Causal spatio-temporal block (×4)")
    x, w = 0.0, 0.62
    rbox(ax, x, 0.0, w, 0.1, "$X_t$: 96 RGB + 16 motion tokens", fc=LGRAY, ec=GRAY, fs=5.6)
    rbox(ax, x, 0.17, w, 0.13, "LN → spatial MHSA\nwithin frame $t$ (12 heads)", fc=LBLUE, ec=BLUE, fs=5.8)
    rbox(ax, x, 0.37, w, 0.24, "LN → temporal attention\n$Q$ from $X_t$;  $K,V$: cache $\\oplus$ $S_{t-1}$\n"
         "logits $+$ bias $\\mathbf{B}$, mask $\\mathbf{M}$", fc=LGREEN, ec=GREEN, fs=5.3)  # fmt: skip
    rbox(ax, x, 0.68, w, 0.1, "LN → FFN (×3 width)", fc=LGRAY, ec=GRAY, fs=5.8)
    rbox(ax, x, 0.85, w, 0.1, "$X_t'$ → memory / next block", fc=LGRAY, ec=GRAY, fs=5.8)
    for y0, y1 in ((0.10, 0.17), (0.30, 0.37), (0.61, 0.68), (0.78, 0.85)):
        arrow(ax, (x + w / 2, y0), (x + w / 2, y1))
    ax.text(x + w / 2 + 0.02, 0.135, "", fontsize=5)
    side = 0.665
    rbox(ax, side, 0.62, 0.335, 0.17, "K/V cache\nlast $K$ = 8 frames\n$\\Delta t \\leq \\tau_c$ = 4 s", fc="white",
         ec=GREEN, fs=5.3, ls=(0, (3, 2)))  # fmt: skip
    rbox(ax, side, 0.38, 0.335, 0.17, "working state\n$S_{t-1}$, 32 slots\nalways visible", fc=LORANGE, ec=ORANGE,
         fs=5.3)  # fmt: skip
    rbox(ax, side, 0.12, 0.335, 0.18, "bias MLP\n$3 \\to 32 \\to H$ on\n$(\\Delta x,\\Delta y,\\log(1{+}\\Delta t))$",
         fc=LPURPLE, ec=PURPLE, fs=5.1)  # fmt: skip
    arrow(ax, (side, 0.66), (w, 0.55), color=GREEN)
    arrow(ax, (side, 0.465), (w, 0.49), color=ORANGE)
    arrow(ax, (side, 0.21), (w, 0.42), color=PURPLE)
    arrow(ax, (w, 0.58), (side + 0.06, 0.79), color=GREEN, ls=(0, (2, 1.5)), rad=-0.35)
    ax.text(side + 0.16, 0.81, "append $K_t, V_t$", fontsize=5.4, color=GREEN, ha="center", va="bottom")
    ax.text(0.0, -0.07, "pre-LN, residual sub-layers; $d$ = 768", fontsize=5.4, color=MUTED, va="top")

    # (b) temporal attention mask
    ax = fig.add_subplot(grid[1])
    panel(ax, "b", "Temporal attention mask", x=-0.12)
    dts = [7.0, 6.0, 4.5, 3.5, 2.5, 1.5, 0.5, 0.0]
    per, rows, state = 4, 8, 4
    cols = per * len(dts) + state
    rng = np.random.default_rng(3)
    values = np.full((rows, cols), np.nan)
    for j, dt in enumerate(dts):
        if dt <= 4.0:
            values[:, j * per : (j + 1) * per] = 1.1 - 0.15 * dt + 0.12 * rng.standard_normal((rows, per))
    values[:, -state:] = 0.6 + 0.08 * rng.standard_normal((rows, state))
    ax.imshow(values, cmap=matplotlib.colormaps["Blues"], vmin=-0.3, vmax=1.5, aspect="auto",
              extent=(0, cols, rows, 0), interpolation="nearest")  # fmt: skip
    masked = sum(dt > 4.0 for dt in dts) * per
    ax.add_patch(Rectangle((0, 0), masked, rows, fc=LGRAY, ec="none", hatch="xxxx", lw=0))
    ax.text(masked / 2, rows / 2, "masked\n$\\Delta t > \\tau_c$", ha="center", va="center", fontsize=5.8,
            color=INK, bbox=dict(fc="white", ec="none", pad=1.0))  # fmt: skip
    for j in range(1, len(dts) + 1):
        ax.axvline(j * per, color="white", lw=0.8)
    ax.add_patch(Rectangle((cols - state, 0), state, rows, fill=False, ec=ORANGE, lw=1.0))
    ax.set_xticks([j * per + per / 2 for j in range(len(dts))] + [cols - state / 2])
    ax.set_xticklabels([f"−{d:g}" if d else "0" for d in dts] + ["$S$"], fontsize=5.6)
    ax.set_xlabel(r"keys: cached frames ($\Delta t$, s) and state slots", labelpad=1)
    ax.set_yticks([0.5, rows - 0.5])
    ax.set_yticklabels(["1", "N"])
    ax.set_ylabel("query tokens of frame $t$", labelpad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # (c) relative-time bias (illustrative per-head shapes)
    ax = fig.add_subplot(grid[2])
    panel(ax, "c", "Relative-time bias", x=-0.2)
    dt = np.linspace(0, 4, 200)
    for h, (a, b, c) in enumerate(((1.2, 0.9, BLUE), (0.4, 0.15, GREEN), (-0.2, -0.35, ORANGE))):
        ax.plot(dt, a - b * np.log1p(dt), color=c, lw=1.0, label=f"head {h + 1}")
    ax.axvline(4.0, color=GRAY, lw=0.6, ls=(0, (3, 2)))
    ax.text(3.9, 1.18, r"$\tau_c$", ha="right", va="top", fontsize=6.5, color=MUTED)
    ax.set_xlabel(r"$\Delta t$ (s), $\Delta x=\Delta y=0$", labelpad=1)
    ax.set_ylabel(r"bias $B_h$", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", handlelength=1.2)
    save(fig, "fig3_v1_attention")


# ----------------------------------------------------------------------------- figure 4
def fig4_v1_memory():
    """v1: gated working state, four-part capsule attention mask, event bank and evidence reader."""
    fig = plt.figure(figsize=(FULL, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.45], wspace=0.42)

    # (a) gated working-state update
    ax = canvas(fig.add_subplot(grid[0]))
    panel(ax, "a", "Gated working state")
    rbox(ax, 0.0, 0.8, 0.44, 0.12, "$S_{t-1} + Q_s$", fc=LORANGE, ec=ORANGE, fs=6.2)
    rbox(ax, 0.56, 0.8, 0.44, 0.12, "$[S_{t-1};\\, X_t']$", fc=LGRAY, ec=GRAY, fs=6.2)
    ax.text(0.22, 0.94, "queries", fontsize=5.6, ha="center", color=MUTED)
    ax.text(0.78, 0.94, "keys / values", fontsize=5.6, ha="center", color=MUTED)
    rbox(ax, 0.0, 0.53, 1.0, 0.15, "$P = \\mathrm{Attn} + \\mathrm{FFN}$", fc=LGREEN, ec=GREEN, fs=6.4)
    rbox(ax, 0.0, 0.29, 1.0, 0.13, "$g = \\sigma(W_g[S_{t-1};\\, P])$", fc=LPURPLE, ec=PURPLE, fs=6.4)
    rbox(ax, 0.0, 0.05, 1.0, 0.14, "$S_t = (1-g)\\odot S_{t-1} + g \\odot P$", fc=LORANGE, ec=ORANGE, fs=6.6)
    arrow(ax, (0.22, 0.8), (0.32, 0.68))
    arrow(ax, (0.78, 0.8), (0.68, 0.68))
    arrow(ax, (0.5, 0.53), (0.5, 0.42))
    arrow(ax, (0.5, 0.29), (0.5, 0.19))
    ax.text(0.5, -0.04, "32 slots, $d$ = 768", fontsize=5.6, ha="center", va="top", color=MUTED)

    # (b) capsule mask
    ax = fig.add_subplot(grid[1])
    panel(ax, "b", "Capsule attention mask", x=-0.3)
    widths = [32, 112, 32]
    total = sum(widths)
    allowed = np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0]])
    colors = [LORANGE, LBLUE, LGREEN]
    edges = [ORANGE, BLUE, GREEN]
    for r in range(4):
        x0 = 0
        for c, w in enumerate(widths):
            if allowed[r, c]:
                ax.add_patch(Rectangle((x0, 3 - r), w, 1, fc=colors[c], ec="white", lw=1.2))
            else:
                ax.add_patch(Rectangle((x0, 3 - r), w, 1, fc="white", ec=LGRAY, lw=0.4))
            x0 += w
    ax.set_xlim(0, total)
    ax.set_ylim(0, 4)
    ax.set_yticks([3.5, 2.5, 1.5, 0.5])
    ax.set_yticklabels(["before", "change", "after", "detail"])
    starts = np.cumsum([0] + widths[:-1])
    ax.set_xticks([st + w / 2 for st, w in zip(starts, widths)])
    ax.set_xticklabels([r"$S_{t-1}$", r"$X_t'$", r"$S_t$"])
    for st, w, e in zip(starts, widths, edges):
        ax.plot([st + 1, st + w - 1], [4.06, 4.06], color=e, lw=2, clip_on=False)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    seeds = [r"$\bar S_{t-1}$", r"$\bar S_t-\bar S_{t-1}$", r"$\bar S_t$", r"$\bar X_t'$"]
    for r, seed in enumerate(seeds):
        ax.text(total + 5, 3.5 - r, seed, va="center", fontsize=6.2, color=INK)
    ax.text(total + 5, 4.0, "query seed", fontsize=5.8, color=MUTED, va="bottom")
    ax.set_xlabel("keys / values (32, 112, 32 tokens)", labelpad=2)

    # (c) event bank and reader
    ax = canvas(fig.add_subplot(grid[2]))
    panel(ax, "c", "Event bank and evidence reader")
    n_slots, sx, sy, sw = 12, 0.0, 0.68, 0.06
    selected = {2, 5, 9}
    for i in range(n_slots):
        x0 = sx + i * (sw + 0.012)
        sel = i in selected
        rbox(ax, x0, sy, sw, 0.16, fc="white", ec=RED if sel else GRAY, lw=0.9 if sel else 0.5, pad=0.004)
        for k, c in enumerate((LORANGE, LGRAY, LGREEN, LBLUE)):
            ax.add_patch(Rectangle((x0 + 0.008, sy + 0.014 + k * 0.035), sw - 0.016, 0.028, fc=c, ec="none", zorder=3))
    pointer = sx + 7 * (sw + 0.012) + sw / 2
    arrow(ax, (pointer, sy + 0.25), (pointer, sy + 0.17))
    ax.text(pointer + 0.02, sy + 0.21, "write (FIFO or utility)", fontsize=5.6, ha="left", va="center")
    ax.text(0.86, sy - 0.03, "128 slots × 4 tokens (12 shown)", fontsize=5.6, color=MUTED, va="top", ha="right")
    rbox(ax, 0.0, 0.4, 0.42, 0.18, "top-$k$ by $q^{\\top}\\bar c_i/\\sqrt{d}$\n$k$ = 16,  $t_{\\mathrm{known}} \\leq t_q$",
         fc=LRED, ec=RED, fs=5.8)  # fmt: skip
    rbox(ax, 0.5, 0.4, 0.5, 0.18, "+ time embedding\n(age, span, before)", fc=LPURPLE, ec=PURPLE, fs=5.6)
    arrow(ax, (0.21, sy), (0.21, 0.58), color=RED)
    arrow(ax, (0.42, 0.49), (0.5, 0.49))
    rbox(ax, 0.0, 0.06, 0.62, 0.22, "reader ×2: 96 queries + $q$\nattend $[X_{\\mathrm{last}};\\, S_t;$ capsules$]$",
         fc=LGREEN, ec=GREEN, fs=5.5)  # fmt: skip
    rbox(ax, 0.7, 0.06, 0.3, 0.22, "projector\n→ LLM prefix", fc=LGRAY, ec=GRAY, fs=5.8)
    arrow(ax, (0.75, 0.4), (0.45, 0.28))
    arrow(ax, (0.62, 0.17), (0.7, 0.17))
    save(fig, "fig4_v1_memory")


if __name__ == "__main__":
    fig1_transition_memory()
    fig2_readout()
    fig3_v1_attention()
    fig4_v1_memory()
    print(OUT)
