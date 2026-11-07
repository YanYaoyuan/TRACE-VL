"""v3 configuration (YAML sections: backbone, stream, cst, transition, memory, alert, train, stage2)."""

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
    # Learning signal for the emission gate (PR #3 A100 pilot: slots went inactive). Soft activity
    # p_k = sigmoid((mass_k - mass_threshold) / activity_temperature); eval emits p_k >= 0.5 (= mass >= threshold);
    # training samples Bernoulli(p_k) and passes task gradients to p_k straight-through.
    activity_temperature: float = 0.25
    train_sampling: bool = True
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
    alert_fraction: float = 0.0  # share of samples drawn from alert-only records (anomaly sets) when both exist
    # Expected tokens per P-step (gist + 3 x active slots), feasible range [gist, gist + 3 * max_active].
    # Predeclared pilot value 20 (PR #3 A100 report); the old I-inclusive name is rejected, not reinterpreted.
    target_p_step_tokens: float = 20.0
    target_tokens_per_step: float | None = None
    kd_temperature: float = 1.0
    tbptt_steps: int = 0  # detach recurrent adapter state every N steps (0 = never)
    save_every: int = 500
    seed: int = 7


@dataclass
class Stage2Config:
    """Long-stream training through the streaming cache (KV policy + Gated DeltaNet state), see train_stage2.py."""

    chunk_steps: int = 8  # steps per gradient chunk (truncated BPTT); chunks also end at question times
    max_seconds: float = 600.0  # streams are cut to this length
    steps: int = 1000
    grad_accum: int = 2  # streams per optimizer update per rank
    question_weight: float = 1.0  # CE of delayed questions answered on a forked cache
    retention_weight: float = 0.5  # BCE of the TKM retention scorer against evidence windows
    lr_scale: float = 0.5  # Stage 2 learning rates = train.lr / train.lora_lr x this


@dataclass
class V3Config:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    cst: CSTConfig = field(default_factory=CSTConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)

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
            "stage2": Stage2Config,
        }
        unknown = set(values) - set(parts) - {"extends", "data", "eval"}
        if unknown:
            raise ValueError(f"unknown v3 config sections: {sorted(unknown)}")
        config = cls(**{name: _build(kind, values.get(name)) for name, kind in parts.items()})
        if config.transition.p_step_mode not in {"transition", "pooled"}:
            raise ValueError("transition.p_step_mode must be transition or pooled")
        if config.train.target_tokens_per_step is not None:
            raise ValueError(
                "train.target_tokens_per_step was replaced by train.target_p_step_tokens (expected tokens per "
                "P-step: gist + 3 x active slots); set that instead"
            )
        low = config.transition.gist
        high = config.transition.gist + 3 * config.transition.max_active
        if not low <= config.train.target_p_step_tokens <= high:
            raise ValueError(
                f"train.target_p_step_tokens={config.train.target_p_step_tokens} is infeasible: a P-step emits "
                f"between {low} (gist only) and {high} (gist + 3 x max_active) tokens"
            )
        if not 0.0 <= config.train.alert_fraction <= 1.0:
            raise ValueError("train.alert_fraction must be in [0, 1]")
        if config.transition.activity_temperature <= 0:
            raise ValueError("transition.activity_temperature must be positive")
        k = config.transition.pooled_tokens
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("transition.pooled_tokens must be a positive integer")
        if config.stage2.chunk_steps < 1 or config.stage2.max_seconds <= 0 or config.stage2.grad_accum < 1:
            raise ValueError("stage2.chunk_steps / grad_accum must be >= 1 and stage2.max_seconds positive")
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
