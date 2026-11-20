"""Container for the trainable v3 modules (the backbone is held separately by `Backbone`)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .alert import AlertHead
from .config import V3Config
from .modules import CSTAdapter, RetentionScorer, TransitionEncoder


class TraceV3Modules(nn.Module):
    def __init__(self, cfg: V3Config, model_dim: int = 2048):
        super().__init__()
        self.cfg = cfg
        self.cst = CSTAdapter(cfg.cst, model_dim)
        self.transition = TransitionEncoder(cfg.transition, cfg.cst.dim, model_dim)
        # Jev-style alert classification every step (noul / choice), latently connected to the backbone.
        self.alert = AlertHead(model_dim, cfg.alert)
        self.retention = RetentionScorer()

    def parameter_report(self) -> dict:
        report = {name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()}
        report["total"] = sum(report.values())
        return report

    def save(self, path: str | Path, extra: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"modules": self.state_dict(), "config": self.cfg.to_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path: str | Path, cfg: V3Config, model_dim: int = 2048, strict: bool = True):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        modules = cls(cfg, model_dim)
        modules.load_state_dict(saved["modules"], strict=strict)
        return modules


def stream_step(modules: "TraceV3Modules", x: torch.Tensor, grid: tuple[int, int], t: float, cst_cache, te_cache):
    """The single per-step path shared by streaming, training and the budget audit (PR #3 review):
    CST adapter -> transition encoder -> write the step's slot memory back into the CST cache.

    Keeping this in one place guarantees the recurrent visual state (and hence surprise, active slots and
    token counts) evolves identically everywhere.
    """
    dtype = next(modules.parameters()).dtype
    adapted, z = modules.cst(x.to(dtype), grid, t, cst_cache)
    info = modules.transition(adapted, z, grid, te_cache)
    slot_memory = info.aux.get("slot_memory")
    if slot_memory is not None:
        cap = modules.cfg.transition.memory_slots
        times = torch.full((slot_memory.shape[0],), t, device=slot_memory.device, dtype=slot_memory.dtype)
        cst_cache.memory = slot_memory if cst_cache.memory is None else torch.cat([cst_cache.memory, slot_memory])[-cap:]
        cst_cache.memory_times = (
            times if cst_cache.memory_times is None else torch.cat([cst_cache.memory_times, times])[-cap:]
        )
    return adapted, z, info
