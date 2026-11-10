"""Run identity and completion checks for paired pilots (scripts/run_pilot003.sh), testable on CPU.

PR #3 review of cf92794:
- a trained arm counts as complete only through an atomic completion record written by the trainer after
  the final module/LoRA save; the record pins step, resolved config and content hashes of both artifacts;
- the run identity holds the fully resolved per-arm configs (not wrapper YAML text), the record files and
  the media identity, so a changed inherited setting or moved/replaced media cannot mix into a resumed run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

COMPLETE = "complete.json"
ARMS = {"transition": "transition", "after": "after", "b4": "pooled"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_hashes(root: Path, prefix: str = "") -> dict:
    """Hashes of one module checkpoint + LoRA adapter pair (adapter weights and config must exist)."""
    lora = root / "lora"
    files = sorted(f for f in lora.rglob("*") if f.is_file()) if lora.is_dir() else []
    if not any(f.suffix in {".safetensors", ".bin"} for f in files):
        raise FileNotFoundError(f"{lora}: no adapter weight file")
    if not (lora / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{lora}: no adapter_config.json")
    if not (root / "modules.pt").is_file():
        raise FileNotFoundError(f"{root}: no modules.pt")
    return {f"{prefix}modules.pt": _sha256(root / "modules.pt"),
            **{f"{prefix}lora/{f.relative_to(lora)}": _sha256(f) for f in files}}  # fmt: skip


def artifact_hashes(output: Path, checkpoint_steps=()) -> dict:
    """Content hashes of the final module/LoRA pair and of every required intermediate checkpoint pair."""
    output = Path(output)
    hashes = _pair_hashes(output)
    for step in sorted(checkpoint_steps):
        name = f"checkpoints/step-{step:05d}/"
        hashes.update(_pair_hashes(output / name, name))
    return hashes


def write_completion(output: Path, step: int, config: dict, extra: dict | None = None):
    """Called by the trainer (rank 0) only after the final module and LoRA saves returned. `extra` pins further
    run identity (e.g. Stage 2: training streams, initial checkpoint, source revision)."""
    output = Path(output)
    steps = (config.get("train") or {}).get("checkpoint_steps") or ()
    record = {"step": step, "config": config, "artifacts": artifact_hashes(output, steps)}
    if extra is not None:
        record["extra"] = extra
    tmp = output / (COMPLETE + ".tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output / COMPLETE)


def training_complete(output: Path, config: dict, steps: int, extra: dict | None = None) -> tuple[bool, str]:
    """(True, "") only if the completion record matches the expected config/step, the artifacts on disk still
    hash to the recorded values, and the metric log holds exactly steps 1..`steps`."""
    output = Path(output)
    record_path = output / COMPLETE
    if not record_path.is_file():
        return False, "no completion record (interrupted or never finished)"
    try:
        record = json.loads(record_path.read_text())
    except json.JSONDecodeError:
        return False, "unreadable completion record"
    if record.get("step") != steps:
        return False, f"completed step {record.get('step')} != {steps}"
    if record.get("config") != config:
        return False, "trained with a different resolved config"
    if extra is not None and record.get("extra") != extra:
        return False, "trained with a different run identity (streams, initialization or source)"
    try:
        if artifact_hashes(output, (config.get("train") or {}).get("checkpoint_steps") or ()) != record.get("artifacts"):
            return False, "checkpoint/adapter files differ from the completion record"
    except FileNotFoundError as error:
        return False, str(error)
    metrics = output / "metrics.jsonl"
    rows = [json.loads(s) for s in metrics.read_text().splitlines() if s.strip()] if metrics.is_file() else []
    if [r.get("step") for r in rows] != list(range(1, steps + 1)):
        return False, "metric log is not exactly steps 1..N"
    return True, ""


def media_identity(records: list[dict], video_root: Path) -> str:
    from .harness import _video_identity

    return _video_identity(records, str(video_root))


def resolved_configs(config_dir: Path, model: str | None, runs: Path, prefix: str = "pilot003") -> dict:
    """Write per-arm wrapper configs (repo arm config + local model path) and return the resolved dicts."""
    from .config import load_v3_config

    out_dir = Path(runs) / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for arm, mode in ARMS.items():
        text = f"extends: {Path(config_dir).resolve() / f'{prefix}_{arm}.yaml'}\n"
        if model:
            text += f"backbone:\n  name: {model}\n  local_files_only: true\n"
        (out_dir / f"{arm}.yaml").write_text(text)
        cfg, _ = load_v3_config(out_dir / f"{arm}.yaml")
        t = cfg.transition
        if t.p_step_mode != mode:
            raise ValueError(f"{arm}: p_step_mode {t.p_step_mode} != {mode}")
        if t.fixed_active <= 0 or t.scene_cut_surprise <= 1.0:
            raise ValueError(f"{arm}: needs fixed_active > 0 and scene cuts disabled")
        if cfg.train.target_p_step_tokens != t.gist + 3 * t.fixed_active:
            raise ValueError(f"{arm}: target_p_step_tokens must equal gist + 3 x fixed_active")
        resolved[arm] = cfg.to_dict()
    stripped = []
    for d in resolved.values():
        d = json.loads(json.dumps(d))
        d["transition"]["p_step_mode"] = None
        stripped.append(d)
    if any(d != stripped[0] for d in stripped[1:]):
        raise ValueError("arm configs differ beyond p_step_mode")
    return resolved


def preflight(data_dir: Path, video_root: Path, runs: Path, config_dir: Path, model: str | None,
              clip_seconds: float, gpus: int, source_dir: Path, prefix: str = "pilot003") -> dict:  # fmt: skip
    """Validate inputs and pin the run identity; a restart with any changed input raises."""
    data_dir, video_root, runs = Path(data_dir).resolve(), Path(video_root).resolve(), Path(runs).resolve()

    def read(p):
        return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

    train, val = read(data_dir / "train.jsonl"), read(data_dir / "val.jsonl")
    for split, rows in (("train", train), ("val", val)):
        if not rows or any(r.get("split") != split for r in rows):
            raise ValueError(f"invalid {split} records")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate {split} ids")
        missing = [r["video"] for r in rows if not (video_root / r["video"]).is_file()]
        if missing:
            raise ValueError(f"{len(missing)} {split} videos missing under the video root, e.g. {missing[0]}")
    if {r["video"] for r in train} & {r["video"] for r in val}:
        raise ValueError("train/val video overlap")
    source = hashlib.sha256()
    for p in sorted(Path(source_dir).glob("*.py")):
        source.update(p.name.encode() + p.read_bytes())
    identity = {
        "pilot": prefix,
        "configs": resolved_configs(config_dir, model, runs, prefix),
        "train_sha256": _sha256(data_dir / "train.jsonl"),
        "val_sha256": _sha256(data_dir / "val.jsonl"),
        "media": media_identity(train + val, video_root),
        "video_root": str(video_root),
        "source_sha256": source.hexdigest(),
        "clip_seconds": float(clip_seconds),
        "gpus": int(gpus),
    }
    target = runs / "identity.json"
    if target.exists():
        old = json.loads(target.read_text())
        changed = sorted(k for k in identity if old.get(k) != identity[k])
        if changed:
            raise ValueError(f"inputs changed since this pilot started ({changed}); use a new TRACE_RUNS")
    else:
        tmp = runs / "identity.json.tmp"
        tmp.write_text(json.dumps(identity, indent=2) + "\n")
        os.replace(tmp, target)
    return {"train_questions": len(train), "val_questions": len(val), "identity": str(target)}
