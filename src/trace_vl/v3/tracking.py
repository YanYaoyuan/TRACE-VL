"""Optional Weights & Biases tracking (extra `track`). Everything here is a no-op unless enabled, so training and
evaluation behave identically without wandb.

Enable with TRACE_WANDB=1 (or `--wandb` on the CLI commands that take it). Standard wandb variables apply:
  WANDB_PROJECT (default "trace-vl"), WANDB_ENTITY, WANDB_MODE=offline on hosts without access to wandb.ai
  (then run `wandb sync <dir>/wandb/offline-run-*` from a machine with access), WANDB_DIR, WANDB_API_KEY
  (or `wandb login` once). Never put keys in configs or scripts.

Only rank 0 creates a run. Runs of one experiment share a group (TRACE_WANDB_GROUP, default the run directory
name) so training arms, evaluation summaries and demos appear together.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path


def enabled_from_env() -> bool:
    return os.environ.get("TRACE_WANDB", "0").lower() in {"1", "true", "yes", "on"}


class Tracker:
    """Thin wrapper over a wandb run; every method is a no-op when the run is None."""

    def __init__(self, run=None):
        self.run = run

    @classmethod
    def start(cls, enabled: bool, name: str, group: str | None, job_type: str, config: dict | None = None,
              tags: list[str] | None = None, directory: str | Path | None = None) -> "Tracker":  # fmt: skip
        if not enabled:
            return cls(None)
        try:
            import wandb
        except ImportError:
            warnings.warn("TRACE_WANDB is set but wandb is not installed (uv sync --extra track); tracking disabled")
            return cls(None)
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "trace-vl"), entity=os.environ.get("WANDB_ENTITY"),
                         name=name, group=group or os.environ.get("TRACE_WANDB_GROUP"), job_type=job_type,
                         config=config or {}, tags=tags, dir=str(directory) if directory else None,
                         reinit="create_new")  # fmt: skip
        return cls(run)

    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self, values: dict, step: int | None = None):
        if self.run is None:
            return
        clean = {k: v for k, v in values.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}  # fmt: skip
        if clean:
            self.run.log(clean, step=step)

    def log_media(self, values: dict, step: int | None = None):
        if self.run is not None:
            self.run.log(values, step=step)

    def summary(self, values: dict):
        if self.run is not None:
            for key, value in values.items():
                self.run.summary[key] = value

    def upload(self, name: str, kind: str, paths: list[str | Path], metadata: dict | None = None):
        """Files as one versioned artifact (missing paths are skipped; directories are added recursively)."""
        if self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=_safe(name), type=kind, metadata=metadata or {})
        added = 0
        for path in map(Path, paths):
            if path.is_dir():
                artifact.add_dir(str(path), name=path.name)
                added += 1
            elif path.is_file():
                artifact.add_file(str(path), name=path.name)
                added += 1
        if added:
            self.run.log_artifact(artifact)

    def table(self, key: str, columns: list[str], rows: list[list]):
        if self.run is None:
            return
        import wandb

        self.run.log({key: wandb.Table(columns=columns, data=rows)})

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None


def _safe(name: str) -> str:
    """wandb artifact names allow alphanumerics, dashes, underscores and dots."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)[:120]
