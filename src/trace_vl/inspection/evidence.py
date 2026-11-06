"""Turn a question snapshot into interleaved text/image evidence under a visual-token budget.

All methods share the same current frame (`frame_tokens`) and the same extra budget
(`budget_tokens`) for memory evidence. Images are resized here to multiples of `unit` pixels so the
VLM processor does not resize them again and the planned token count is exact; evaluation also logs
the token count the processor actually produced.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from .memory import overlap

METHODS = (
    "none",  # question only: language-prior control
    "current",  # current frame only: memory-free control
    "uniform",  # frames evenly spaced over the question window
    "recent",  # the most recent frames (sliding-window streaming baseline)
    "transition",  # ours: change memory with before/after crops
    "transition_after",  # ablation: no "before" crops
    "transition_meta",  # ablation: change times and locations as text, no crops
    "textlog",  # caption log written online by the same VLM
)

LETTERS = "ABCDEFGH"


@dataclass
class PackConfig:
    unit: int = 32  # pixels per token side: Qwen3-VL 16 px patches x 2 merge; Qwen2.5-VL uses 28
    frame_tokens: int = 256
    budget_tokens: int = 1024
    crop_tokens: int = 64
    frames: int = 8
    caption_limit: int = 120

    @classmethod
    def from_dict(cls, values: dict | None) -> "PackConfig":
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown packing settings: {sorted(unknown)}")
        return cls(**values)


def clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60:02d}:{rest % 60:02d}"


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def ago(seconds: float) -> str:
    return duration(seconds) + " ago"


def fit(image: Image.Image, tokens: int, unit: int) -> Image.Image:
    """Largest resize with sides that are multiples of `unit` and at most `tokens` visual tokens."""
    tokens = max(4, tokens)
    width, height = image.size
    scale = math.sqrt(tokens * unit * unit / (width * height))
    w = max(2, int(width * scale / unit))
    h = max(2, int(height * scale / unit))
    while w * h > tokens:
        if w >= h:
            w -= 1
        else:
            h -= 1
    return image.resize((w * unit, h * unit), Image.BICUBIC)


def tokens_of(image: Image.Image, unit: int) -> int:
    return (image.size[0] // unit) * (image.size[1] // unit)


def draw_region(image: Image.Image, region) -> Image.Image:
    if region is None:
        return image
    image = image.copy()
    w, h = image.size
    box = [region[0] * w, region[1] * h, region[2] * w, region[3] * h]
    ImageDraw.Draw(image).rectangle(box, outline=(255, 0, 0), width=max(2, round(max(w, h) / 200)))
    return image


@lru_cache(maxsize=256)
def _captions(path: str):
    file = Path(path)
    if not file.exists():
        return None
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()]


class Packer:
    def __init__(self, root, config: PackConfig | None = None):
        self.root = Path(root)
        self.cfg = config or PackConfig()

    def _image(self, relative, tokens, region=None):
        image = Image.open(self.root / relative).convert("RGB")
        return fit(draw_region(image, region), tokens, self.cfg.unit)

    def pack(self, snapshot: dict, question: dict, method: str):
        """Return (segments, info); segments are {"type": "text"|"image", ...} items for the VLM."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        cfg, tq = self.cfg, snapshot["query_time"]
        region, window = snapshot.get("region"), snapshot["window"]
        segments, spent = [], 0

        def text(value):
            segments.append({"type": "text", "text": value})

        def image(relative, tokens, with_region=True):
            nonlocal spent
            picture = self._image(relative, tokens, region if with_region else None)
            spent += tokens_of(picture, cfg.unit)
            segments.append({"type": "image", "image": picture})

        header = "You are reviewing footage from a fixed security camera. "
        header += f"The current time is {clock(tq)}."
        if region is not None:
            header += " The red box marks the area the question is about."
        text(header)
        info = {"method": method}

        if method in {"uniform", "recent"}:
            frames = snapshot[method]
            if len(frames) > cfg.frames:
                if method == "recent":
                    frames = frames[-cfg.frames :]
                else:
                    step = (len(frames) - 1) / max(1, cfg.frames - 1)
                    frames = [frames[round(i * step)] for i in range(cfg.frames)]
            if frames:
                per_frame = cfg.budget_tokens // len(frames)
                if method == "uniform":
                    text(f"Earlier frames sampled evenly over the last {duration(window)}:")
                else:
                    text("The most recent earlier frames:")
                for frame in frames:
                    text(f"[{clock(frame['t'])}, {ago(tq - frame['t'])}]")
                    image(frame["image"], per_frame)
            info["frames"] = len(frames)
        elif method.startswith("transition"):
            self._transitions(snapshot, method, text, image, info)
        elif method == "textlog":
            captions = _captions(str(self.root / "streams" / snapshot["stream"] / "captions.jsonl"))
            if captions is None:
                raise FileNotFoundError(f"run the caption stage first for stream {snapshot['stream']}")
            lines = [c for c in captions if tq - window <= c["t"] <= snapshot["frame_time"]]
            lines = lines[-cfg.caption_limit :]
            text("Caption log of earlier frames (oldest first):")
            text("\n".join(f"[{clock(c['t'])}] {c['caption']}" for c in lines) or "(no captions)")
            info["captions"] = len(lines)

        if method != "none":
            text(f"Current frame [{clock(tq)}]:")
            image(snapshot["current"], cfg.frame_tokens)
        text("Question: " + question["question"])
        text("\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(question["options"])))
        text("Answer with the option letter only.")
        info["planned_visual_tokens"] = spent
        return segments, info

    def _transitions(self, snapshot, method, text, image, info):
        cfg, tq, region = self.cfg, snapshot["query_time"], snapshot.get("region")
        transitions = snapshot["transitions"]
        text(
            "Change memory: parts of the scene that changed and stayed changed, as recorded by a motion "
            f"detector ({snapshot['transitions_known']} changes recorded, most relevant first)."
        )
        crops = {
            "transition": ("before", "after", "after_end"),
            "transition_after": ("after", "after_end"),
        }.get(method, ())
        spent, shown = 0, 0
        for item in transitions:
            needed = [c for c in crops if c in item["crops"]]
            if crops and spent + cfg.crop_tokens * len(needed) > cfg.budget_tokens:
                break
            if not crops and shown >= max(1, cfg.budget_tokens // cfg.crop_tokens):
                break
            shown += 1
            if item["kind"] == "global":
                where = "the whole scene (lighting or camera change)"
            elif region is not None and overlap(item["box"], region) > 0.3:
                where = "inside the red box"
            else:
                where = "elsewhere in the scene"
            start = item["t_start"]
            line = f"Change {shown}: {where}; appeared at {clock(start)} ({ago(tq - start)})"
            if item["t_end"] is not None and item["status"] == "resolved":
                line += f"; changed back at {clock(item['t_end'])} ({ago(tq - item['t_end'])})."
            elif item["status"] == "absorbed":
                line += "; stayed changed long enough to become part of the background."
            else:
                line += "; still changed now."
            if method == "transition_meta":
                box = ", ".join(f"{v:.2f}" for v in item["box"])
                line += f" Location (x1, y1, x2, y2 as fractions of the frame): {box}."
            text(line)
            labels = {"before": "Before:", "after": "After the change:", "after_end": "After it changed back:"}
            for name in needed:
                text(labels[name])
                image(item["crops"][name], cfg.crop_tokens, with_region=False)
                spent += cfg.crop_tokens
        if not shown:
            text("No changes were recorded.")
        info["transitions_shown"] = shown
