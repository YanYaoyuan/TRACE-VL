"""Validated experiment configuration; no implicit model downloads."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    backend: str = "tiny"
    dim: int = 64
    vision_dim: int = 64
    language_dim: int = 64
    heads: int = 4
    st_layers: int = 2
    ffn_ratio: int = 3
    spatial_tokens: int = 8
    motion_tokens: int = 4
    cache_frames: int = 4
    cache_seconds: float = 4.0
    state_slots: int = 4
    event_capacity: int = 8
    capsule_tokens: int = 4
    read_events: int = 2
    read_tokens: int = 8
    reader_layers: int = 2
    image_size: int = 32
    memory_mode: str = "capsule"
    write_policy: str = "every_frame"
    write_threshold: float = 0.5
    retention_policy: str = "fifo"
    normalize_memory: bool = False
    freeze_vision: bool = False
    freeze_language: bool = False
    vision_model: str = "google/siglip2-large-patch16-384"
    language_model: str = "Qwen/Qwen3-1.7B"
    local_files_only: bool = False
    query_encoding: str = "embedding_mean"
    lora_rank: int = 0
    dropout: float = 0.0
    text_limit: int = 256

    def __post_init__(self):
        if self.backend not in {"tiny", "hf"}:
            raise ValueError("backend must be tiny or hf")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        for key in (
            "dim",
            "heads",
            "spatial_tokens",
            "cache_frames",
            "state_slots",
            "event_capacity",
            "read_events",
            "read_tokens",
            "reader_layers",
            "image_size",
            "text_limit",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.st_layers < 0 or self.motion_tokens < 0 or self.cache_seconds <= 0:
            raise ValueError("invalid temporal configuration")
        if self.memory_mode not in {"capsule", "summary", "recent"}:
            raise ValueError("invalid memory_mode")
        if self.capsule_tokens != (1 if self.memory_mode == "summary" else 4):
            raise ValueError("summary requires 1 capsule token; other modes require 4")
        if self.write_policy not in {"every_frame", "learned"}:
            raise ValueError("invalid write_policy")
        if self.retention_policy not in {"fifo", "priority"}:
            raise ValueError("invalid retention_policy")
        if not 0 < self.write_threshold < 1:
            raise ValueError("write_threshold must be between 0 and 1")
        if self.query_encoding not in {"embedding_mean", "last_hidden"}:
            raise ValueError("invalid query_encoding")

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model settings: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> dict:
    """Resolve a YAML's optional relative `extends`, then validate model values."""

    def read(p: Path, chain: set[Path]):
        p = p.resolve()
        if p in chain:
            raise ValueError(f"configuration inheritance cycle: {p}")
        value = yaml.safe_load(p.read_text()) or {}
        base = value.pop("extends", None)
        merged = read(p.parent / base, chain | {p}) if base else {}
        return merge(merged, value)

    def merge(base, extra):
        result = copy.deepcopy(base)
        for k, v in extra.items():
            result[k] = merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else v
        return result

    config = read(Path(path), set())
    config["model"] = ModelConfig.from_dict(config.get("model", {})).to_dict()
    return config
