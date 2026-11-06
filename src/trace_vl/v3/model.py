"""Container for the trainable v3 modules (the backbone is held separately by `Backbone`)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import V3Config
from .modules import AlertHead, CSTAdapter, EventHead, RetentionScorer, TransitionEncoder


class TraceV3Modules(nn.Module):
    def __init__(self, cfg: V3Config, model_dim: int = 2048):
        super().__init__()
        self.cfg = cfg
        self.cst = CSTAdapter(cfg.cst, model_dim)
        self.transition = TransitionEncoder(cfg.transition, cfg.cst.dim, model_dim)
        self.alert = AlertHead(model_dim)
        self.event = EventHead(model_dim)
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
