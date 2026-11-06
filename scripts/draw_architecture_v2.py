"""Draw the TRACE-VL v2 architecture as an SVG (no dependencies). Usage: python scripts/draw_architecture_v2.py"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1600, 1040
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"
BLUE = ("#EAF2FB", "#3B6EA8")
GREEN = ("#EAF6EE", "#2F7D4F")
ORANGE = ("#FFF1E0", "#C46A00")
GRAY = ("#F4F4F4", "#8A8A8A")
PURPLE = ("#F3EDFA", "#6B4BA1")
out: list[str] = []


def text(x, y, lines, size=15, weight="normal", anchor="start", color="#1F1F1F", gap=1.3):
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        out.append(
            f'<text x="{x}" y="{y + i * size * gap:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(line)}</text>'
        )


def box(x, y, w, h, colors, title=None, lines=(), size=14, radius=10, dash=False):
    fill, stroke = colors
    style = ' stroke-dasharray="6 4"' if dash else ""
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2"{style}/>'
    )
    top = y + 24
    if title:
        text(x + w / 2, top, title, 16, "bold", "middle", stroke)
        top += 24
    if lines:
        text(x + 12, top, list(lines), size)


def arrow(points, color="#444", dash=False, label=None, label_at=None):
    path = "M " + " L ".join(f"{px},{py}" for px, py in points)
    style = ' stroke-dasharray="6 4"' if dash else ""
    out.append(
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"{style} marker-end="url(#head)"/>'
    )
    if label:
        lx, ly = label_at
        text(lx, ly, label, 13, "normal", "middle", color)


def scene(x, y, w, h, obj=None, region=False, label=None):
    """Tiny fixed-camera frame: wall, floor, optional object and red question box."""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#D9DEE4" stroke="#555" stroke-width="1"/>')
    out.append(f'<rect x="{x}" y="{y + h * 0.55}" width="{w}" height="{h * 0.45}" fill="#B9C2CC"/>')
    out.append(
        f'<rect x="{x + w * 0.08}" y="{y + h * 0.15}" width="{w * 0.18}" height="{h * 0.4}" fill="#8C6D52"/>'
    )
    if obj:
        ox, oy = x + w * 0.55, y + h * 0.62
        out.append(f'<rect x="{ox}" y="{oy}" width="{w * 0.16}" height="{h * 0.2}" rx="2" fill="{obj}"/>')
    if region:
        out.append(
            f'<rect x="{x + w * 0.45}" y="{y + h * 0.5}" width="{w * 0.38}" height="{h * 0.42}" '
            'fill="none" stroke="#E00000" stroke-width="2"/>'
        )
    if label:
        text(x + w / 2, y + h + 15, label, 12, "normal", "middle", "#333")


out.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
    '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#444"/></marker></defs>'
    f'<rect width="{W}" height="{H}" fill="white"/>'
)
text(W / 2, 38, "TRACE-VL v2 — streaming video understanding with transition evidence memory", 24, "bold", "middle")
text(
    W / 2,
    62,
    "A pretrained VLM answers from evidence chosen under a fixed visual-token budget; "
    "the memory stores where the scene changed, before/after crops and when",
    14,
    "normal",
    "middle",
    "#555",
)

# ① Observe
out.append(f'<rect x="20" y="80" width="{W - 40}" height="330" rx="14" fill="none" stroke="{BLUE[1]}" stroke-width="1.5"/>')
text(36, 104, "1  Observe — causal, question-independent (one pass per stream, CPU)", 17, "bold", color=BLUE[1])

box(40, 125, 170, 170, BLUE, "Video stream", ["fixed camera", "sampled at 2 fps", "time-ordered"])
scene(62, 222, 126, 58)
box(
    250,
    125,
    320,
    170,
    BLUE,
    "Change detector",
    [
        "fast background (τ = 4 s)",
        "slow background (τ = 600 s)",
        "static change = differs from slow,",
        "   agrees with fast, for ≥ 6 s",
        "slow bg frozen inside active regions",
        "scene-wide change → reset",
    ],
)
box(610, 125, 360, 170, BLUE, "Transition (memory unit)", [])
scene(625, 175, 96, 58, None, label="before")
scene(733, 175, 96, 58, "#C0392B", label="after")
scene(841, 175, 96, 58, None, label="after it ended")
text(625, 270, ["box · t_start · t_known · t_end · t_end_known", "status: active / resolved / absorbed"], 13)
box(
    1010,
    125,
    280,
    170,
    BLUE,
    "Transition store",
    ["bounded: capacity 128", "eviction order:", "  resolved → absorbed →", "  global → active (last)", "active changes kept longest"],
)
box(
    1330,
    125,
    230,
    170,
    PURPLE,
    "Causality",
    ["exists only after t_known", "end hidden until", "  t_end_known", "never reads the question", "answers use frames ≤ tq"],
)
arrow([(210, 210), (250, 210)])
arrow([(570, 210), (610, 210)])
arrow([(970, 210), (1010, 210)])

box(250, 318, 320, 76, GRAY, None, ["Frame buffer (1 fps, ≤ 30 min)", "→ uniform / recent baselines"], dash=True)
box(610, 318, 360, 76, GRAY, None, ["Caption frames every 10 s", "→ VLM caption log (textlog baseline)"], dash=True)
box(1010, 318, 280, 76, GRAY, None, ["Transition log (all events)", "→ causal alert replay"], dash=True)
arrow([(130, 295), (130, 356), (250, 356)], "#888", dash=True)
arrow([(130, 356), (130, 400), (790, 400), (790, 394)], "#888", dash=True)
arrow([(1150, 295), (1150, 318)], "#888", dash=True)

# ② Read out
out.append(f'<rect x="20" y="430" width="{W - 40}" height="420" rx="14" fill="none" stroke="{GREEN[1]}" stroke-width="1.5"/>')
text(36, 454, "2  Read out — query-conditioned (GPU)", 17, "bold", color=GREEN[1])

box(40, 475, 200, 170, GREEN, "Question / rule", ["query time tq", "red-box region", "time window W", "options A, B, …"])
box(
    280, 475, 230, 170, GREEN, "Causal snapshot(tq)",
    ["transitions with", "  t_known ≤ tq", "future endings hidden", "uniform + recent frames", "current frame (≤ tq)"],
)  # fmt: skip
box(550, 475, 210, 170, GREEN, "Rank", ["1. overlaps the region", "2. inside the window", "3. overlap fraction", "4. most recent"])
box(
    800, 475, 290, 170, GREEN, "Evidence packer",
    ["current frame: 256 tokens", "budget B: 512 / 1024 / 2048", "per change: where / when text", "  + before/after crops (64 each)",
     "images sized to the token grid"],
)  # fmt: skip
box(1130, 475, 200, 170, ORANGE, "Qwen3-VL", ["2B-Instruct (4B / 8B)", "pretrained vision", "LoRA on the LM", "loss on answer only"])
box(1370, 475, 190, 78, ORANGE, None, ["Answer:", "argmax option-letter", "log-probability"], size=13)
box(1370, 567, 190, 78, ORANGE, None, ["Alert: changed ≥ N min", "+ VLM before/after", "verification"], size=13)

arrow([(1010, 270), (992, 270), (992, 418), (395, 418), (395, 475)], BLUE[1])
text(700, 440, "ranked transitions known at tq", 13, "normal", "middle", BLUE[1])
arrow([(240, 560), (280, 560)])
arrow([(510, 560), (550, 560)])
arrow([(760, 560), (800, 560)])
arrow([(1090, 560), (1130, 560)])
arrow([(1330, 514), (1370, 514)])
arrow([(1330, 606), (1370, 606)])
arrow([(1290, 380), (1350, 380), (1350, 632), (1370, 632)], "#888", dash=True)

# Packed prompt example
box(280, 665, 1280, 170, ("#FBFBF7", "#9A9A7A"), None, [], radius=8)
text(296, 688, "Example of packed evidence (what the VLM sees, in order)", 15, "bold", color="#6E6E4E")
text(
    296,
    714,
    ["“Fixed security camera, current time 12:30. The red box marks the area the question is about.”",
     "“Change 1: inside the red box; appeared at 08:12 (4 min 18 s ago); changed back at 10:40 (1 min 50 s ago).”"],
    13,
)  # fmt: skip
scene(300, 752, 82, 50, None, label="Before")
scene(410, 752, 82, 50, "#C0392B", label="After")
scene(520, 752, 82, 50, None, label="Changed back")
text(630, 782, "… more changes until the budget is spent …", 13, color="#555")
scene(900, 752, 110, 62, None, region=True, label="Current frame [12:30]")
text(
    1030,
    770,
    ["Q: In the last 10 minutes, did someone put down an object", "inside the red box and later pick it up again?",
     "A. Yes   B. No   — Answer with the option letter only."],
    13,
)  # fmt: skip

# ③ Compared at equal budget
out.append(f'<rect x="20" y="870" width="{W - 40}" height="150" rx="14" fill="none" stroke="{GRAY[1]}" stroke-width="1.5"/>')
text(36, 894, "3  Compared at the same visual-token budget (same VLM, same current frame)", 17, "bold", color="#555")
chips = [
    ("none", "question only"),
    ("current", "current frame"),
    ("uniform", "frames over W"),
    ("recent", "last frames"),
    ("textlog", "caption log"),
    ("transition", "ours"),
    ("transition_after", "no before crops"),
    ("transition_meta", "text only"),
]
for i, (name, note) in enumerate(chips):
    x = 40 + i * 190
    colors = ORANGE if name == "transition" else GRAY
    box(x, 910, 178, 62, colors, None, [])
    text(x + 89, 936, name, 15, "bold", "middle", colors[1])
    text(x + 89, 958, note, 13, "normal", "middle", "#444")
text(
    40,
    998,
    "First benchmark: MEVA fixed-camera long-gap questions (placed-and-removed, duration, count; still-there control). "
    "Requires a fixed camera. Not yet run on GPU — no results.",
    13,
    color="#666",
)
out.append("</svg>")

target = Path(__file__).resolve().parents[1] / "assets" / "architecture_v2.svg"
target.write_text("\n".join(out) + "\n")
print(target)
