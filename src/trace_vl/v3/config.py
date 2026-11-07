"""v3 configuration (YAML sections: backbone, stream, cst, transition, memory, train)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


def _build(cls, values):
    values = dict(values or {})
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


@dataclass
class BackboneConfig:
    name: str = "Qwen/Qwen3.5-2B"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    local_files_only: bool = False
    thinking: bool = False  # answer directly; the chat template gets an empty think block


@dataclass
class StreamConfig:
    step_seconds: float = 1.0  # one LM step = one temporal patch = 2 frames
    frame_gap: float = 0.5  # seconds between the two frames of a step (2 fps)
    pixels_per_frame: int = 448 * 448  # 448x448 -> 14x14 merged grid -> 196 tokens per step
    unit: int = 32  # patch 16 x spatial merge 2
    max_steps: int = 0  # offline mode: >0 subsamples steps uniformly to this many


@dataclass
class CSTConfig:
    enabled: bool = True
    dim: int = 768
    layers: int = 4
    heads: int = 12
    window: int = 16  # past steps visible to temporal attention
    ffn_ratio: int = 4
    memory_read: bool = True
    checkpoint: bool = True  # activation checkpointing per CST layer in training (needed for 180 s clips)


@dataclass
class TransitionConfig:
    enabled: bool = True
    slots: int = 8
    slot_iters: int = 3
    gist: int = 8
    max_active: int = 8  # transition slots emitted per P-step at most
    mass_threshold: float = 1.0  # slot surprise mass needed to emit before/after/delta tokens
    before_lag: int = 4  # steps back used for the "before" tokens
    i_step_every: int = 8  # full 196-token I-step every N steps
    scene_cut_surprise: float = 0.6  # mean surprise that forces an I-step
    memory_slots: int = 64  # adapter-space memory read by the CST
    # Baseline B4 (experiments.md §2): "pooled" replaces gist + transition tokens on P-steps by a fixed
    # grid of average-pooled frame tokens; set pooled_tokens to the transition run's measured tokens/step.
    p_step_mode: str = "transition"  # transition | pooled
    pooled_match: bool = True  # B4 emits, per P-step, exactly as many tokens as the transition branch would
    pooled_tokens: int = 36  # fixed K per P-step when pooled_match is false (must be 1 <= K <= grid cells)


@dataclass
class AlertConfig:
    """Alert head (Jev-style classification: noul / choice) with latent connections to the backbone."""

    hidden: int = 512
    layers: int = 2
    heads: int = 8
    taps: list = field(default_factory=lambda: [11, 17, 23])  # LM layers whose hidden states are read
    state_layers: list = field(default_factory=lambda: [10, 18, 22])  # Gated DeltaNet states queried by key
    memory_tokens: int = 256  # earlier steps' latent tokens kept for alerts
    state_dropout: float = 0.3  # train without the GDN-state read sometimes (it is only final-step in training)
    train_steps: int = 16  # steps per clip at which the alert head is trained (always includes the last)
    state_every: int = 4  # 1 in N samples gets an extra no-grad pass for end-of-clip GDN states


@dataclass
class MemoryConfig:
    policy: str = "full"  # full | window | drift | tkm | fifo
    sink_steps: int = 1  # first steps always kept (plus the prompt)
    window_seconds: float = 32.0  # recent steps always kept
    budget_tokens: int = 4096  # tokens kept beyond sink + window + dialogue


@dataclass
class TrainConfig:
    lr: float = 1e-4
    lora_lr: float = 5e-5
    weight_decay: float = 0.01
    warmup: int = 100
    steps: int = 2000
    grad_accum: int = 8
    lora_rank: int = 32
    lora_alpha: int = 64
    kd_weight: float = 1.0
    ce_weight: float = 0.5
    pred_weight: float = 0.5
    trans_weight: float = 0.1
    vis_weight: float = 0.0  # needs a negative stream per sample (doubles student cost)
    vis_margin: float = 0.2
    budget_weight: float = 0.01
    alert_weight: float = 1.0
    target_tokens_per_step: float = 40.0
    kd_temperature: float = 1.0
    tbptt_steps: int = 0  # detach recurrent adapter state every N steps (0 = never)
    save_every: int = 500
    seed: int = 7


@dataclass
class V3Config:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    cst: CSTConfig = field(default_factory=CSTConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, values: dict | None) -> "V3Config":
        values = dict(values or {})
        parts = {
            "backbone": BackboneConfig,
            "stream": StreamConfig,
            "cst": CSTConfig,
            "transition": TransitionConfig,
            "memory": MemoryConfig,
            "alert": AlertConfig,
            "train": TrainConfig,
        }
        unknown = set(values) - set(parts) - {"extends", "data", "eval"}
        if unknown:
            raise ValueError(f"unknown v3 config sections: {sorted(unknown)}")
        config = cls(**{name: _build(kind, values.get(name)) for name, kind in parts.items()})
        if config.transition.p_step_mode not in {"transition", "pooled"}:
            raise ValueError("transition.p_step_mode must be transition or pooled")
        k = config.transition.pooled_tokens
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("transition.pooled_tokens must be a positive integer")
        if config.memory.policy not in {"full", "window", "drift", "tkm", "fifo"}:
            raise ValueError("memory.policy must be full, window, drift, tkm or fifo")
        if config.cst.dim % config.cst.heads:
            raise ValueError("cst.dim must be divisible by cst.heads")
        return config

    def to_dict(self):
        return asdict(self)


def load_v3_config(path: str | Path) -> tuple[V3Config, dict]:
    """Return (config, raw dict); the raw dict keeps `data` and `eval` sections for the scripts."""

    def read(p: Path, seen: set[Path]) -> dict:
        p = p.resolve()
        if p in seen:
            raise ValueError(f"config inheritance cycle at {p}")
        raw = yaml.safe_load(p.read_text()) or {}
        base = raw.pop("extends", None)
        merged = read(p.parent / base, seen | {p}) if base else {}
        for key, value in raw.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    raw = read(Path(path), set())
    return V3Config.from_dict(raw), raw
