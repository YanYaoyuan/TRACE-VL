"""Qwen3.5 multimodal RoPE positions for incremental streaming.

Reproduces `Qwen3_5Model.get_rope_index` (transformers 5.20): text tokens advance the position by 1;
a visual block on an (h, w) merged grid starting at position p gets temporal = p, height = p + row,
width = p + col, and then advances the position by max(h, w). Sparse visual tokens (transition or
gist tokens) are placed at their grid row/col inside a block, and the block advances by max(h, w)
as well, so time advances the same way for full and compressed steps.
"""

from __future__ import annotations

import torch


class PositionCounter:
    def __init__(self, start: int = 0):
        self.pos = int(start)

    def text(self, n: int, device=None) -> torch.Tensor:
        ids = torch.arange(self.pos, self.pos + n, device=device)
        self.pos += n
        return ids.view(1, -1).expand(3, -1).clone()

    def grid(self, h: int, w: int, device=None) -> torch.Tensor:
        rows = torch.arange(h, device=device).repeat_interleave(w)
        cols = torch.arange(w, device=device).repeat(h)
        return self.points(rows, cols, h, w)

    def points(self, rows: torch.Tensor, cols: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Tokens at integer grid locations of one (h, w) visual block."""
        rows = rows.long().clamp(0, h - 1)
        cols = cols.long().clamp(0, w - 1)
        start = self.pos
        ids = torch.stack([torch.full_like(rows, start), rows + start, cols + start], dim=0)
        self.pos += max(h, w)
        return ids

    def state(self) -> int:
        return self.pos
