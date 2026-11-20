"""Publish shared evaluation metadata without exposing partially written files."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path


def atomic_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def ensure_run_manifest(path: Path, fingerprint: dict):
    """Create once under a process lock; identical workers never truncate it.

    A corrupt pre-existing manifest is refused, not silently overwritten: its
    predictions may belong to a different experiment. Use a new output directory.
    The Linux A100/local hosts support flock; locks are released on process exit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(fingerprint.get("id"), str) or not fingerprint["id"]:
        raise ValueError("run fingerprint requires a nonempty id")
    with path.with_name(f".{path.name}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            try:
                previous = json.loads(path.read_text())
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"corrupt run manifest {path}; preserve it and use a new output directory") from exc
            if previous != fingerprint:
                raise ValueError(f"{path.parent} holds a different run; use a new output directory")
            return
        atomic_text(path, json.dumps(fingerprint, indent=2) + "\n")
