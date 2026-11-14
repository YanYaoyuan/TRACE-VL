"""Optional Weights & Biases tracking (extra `track`). Everything here is a no-op unless enabled, so training and
evaluation behave identically without wandb, and **tracking can never stop training**:

- every wandb call is wrapped; the first failure is reported once (warning) and tracking is switched off for the
  rest of the run, while local logs (metrics.jsonl, run.json, reports) are always written as before;
- if an online `wandb.init` fails or times out (TRACE_WANDB_INIT_TIMEOUT seconds, default 60), the run falls
  back to **offline** collection in the same directory; sync later with `trace-vl v3 wandb-sync --runs DIR`
  (or `wandb sync DIR/**/wandb/offline-run-*`);
- only rank 0 tracks, and the trainers finish the wandb run only after the process group is destroyed, so a slow
  network at shutdown cannot leave other DDP ranks waiting in a collective.

Enable with TRACE_WANDB=1 (or `--wandb`). Standard variables apply: WANDB_PROJECT (default "trace-vl"),
WANDB_ENTITY, WANDB_MODE=offline (collect only), WANDB_API_KEY or a one-time `wandb login`. Never put keys in
configs or scripts. Runs of one experiment share a group (TRACE_WANDB_GROUP, default the run directory name).

Artifacts keep **relative paths** (relative to `base`, e.g. the run directory), so eval/<arm>/step-00100/score.json
and eval/<arm>/step-01000/score.json stay distinct instead of colliding as score.json.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path


def enabled_from_env() -> bool:
    return os.environ.get("TRACE_WANDB", "0").lower() in {"1", "true", "yes", "on"}


def artifact_name(path: Path, base: Path | None) -> str:
    """Name of a file inside an artifact: its path relative to `base` when it lies under it, else its file name."""
    path = Path(path)
    if base is not None:
        try:
            return path.resolve().relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            pass
    return path.name


class Tracker:
    """Thin, failure-proof wrapper over a wandb run; every method is a no-op when the run is None."""

    def __init__(self, run=None, mode: str | None = None):
        self.run, self.mode, self.error = run, mode, None

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
        kwargs = dict(project=os.environ.get("WANDB_PROJECT", "trace-vl"), entity=os.environ.get("WANDB_ENTITY"),
                      name=name, group=group or os.environ.get("TRACE_WANDB_GROUP"), job_type=job_type,
                      config=config or {}, tags=tags, dir=str(directory) if directory else None,
                      reinit="create_new")  # fmt: skip
        requested = os.environ.get("WANDB_MODE", "online")
        if requested == "disabled":
            return cls(None)
        timeout = float(os.environ.get("TRACE_WANDB_INIT_TIMEOUT", "60"))
        try:
            run = wandb.init(mode=requested, settings=wandb.Settings(init_timeout=timeout), **kwargs)
            return cls(run, requested)
        except Exception as error:  # noqa: BLE001 - network/auth/server problems must not stop training
            if requested == "offline":
                warnings.warn(f"wandb offline init failed ({error!r}); tracking disabled, local logs unaffected")
                return cls(None)
            warnings.warn(f"wandb online init failed ({error!r}); collecting offline for a later `wandb sync`")
        try:
            run = wandb.init(mode="offline", **kwargs)
            return cls(run, "offline")
        except Exception as error:  # noqa: BLE001
            warnings.warn(f"wandb offline fallback failed ({error!r}); tracking disabled, local logs unaffected")
            return cls(None)

    @property
    def active(self) -> bool:
        return self.run is not None

    def _guard(self, what: str, fn):
        if self.run is None:
            return None
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - optional tracking: report once, then stop tracking
            self.error = f"{what}: {error!r}"
            warnings.warn(f"wandb {what} failed ({error!r}); tracking disabled for this run, local logs unaffected")
            run, self.run = self.run, None
            try:
                run.finish(exit_code=1)
            except Exception:  # noqa: BLE001
                pass
            return None

    def log(self, values: dict, step: int | None = None):
        clean = {k: v for k, v in values.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}  # fmt: skip
        if clean:
            self._guard("log", lambda: self.run.log(clean, step=step))

    def log_media(self, values: dict, step: int | None = None):
        self._guard("log", lambda: self.run.log(values, step=step))

    def summary(self, values: dict):
        def write():
            for key, value in values.items():
                self.run.summary[key] = value

        self._guard("summary", write)

    def upload(self, name: str, kind: str, paths: list[str | Path], metadata: dict | None = None,
               base: str | Path | None = None, names: dict | None = None):  # fmt: skip
        """Files/directories as one versioned artifact with relative paths (see module doc). `names` maps a path
        to an explicit artifact name (e.g. files outside `base`). Missing paths are skipped. Returns the artifact
        (or None), so callers and tests can inspect its manifest."""

        def build():
            import wandb

            artifact = wandb.Artifact(name=_safe(name), type=kind, metadata=metadata or {})
            used: dict[str, Path] = {}
            for path in map(Path, paths):
                if not path.exists():
                    continue
                target = (names or {}).get(str(path)) or artifact_name(path, base)
                if target in used and used[target] != path:
                    raise ValueError(f"artifact path collision: {target} ({used[target]} and {path})")
                used[target] = path
                if path.is_dir():
                    artifact.add_dir(str(path), name=target)
                else:
                    artifact.add_file(str(path), name=target)
            if used:
                self.run.log_artifact(artifact)
            return artifact

        return self._guard("artifact upload", build)

    def table(self, key: str, columns: list[str], rows: list[list]):
        def write():
            import wandb

            self.run.log({key: wandb.Table(columns=columns, data=rows)})

        self._guard("table", write)

    def finish(self):
        if self.run is not None:
            run, self.run = self.run, None
            try:
                run.finish()
            except Exception as error:  # noqa: BLE001
                warnings.warn(f"wandb finish failed ({error!r}); offline data (if any) remains in the run directory")


def _safe(name: str) -> str:
    """wandb artifact names allow alphanumerics, dashes, underscores and dots."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)[:120]


def offline_runs(root: str | Path) -> list[Path]:
    """Offline run directories collected under `root` (for `wandb sync`)."""
    return sorted(p for p in Path(root).rglob("offline-run-*") if p.is_dir())


def sync_offline(root: str | Path, dry_run: bool = False) -> dict:
    """`wandb sync` every offline run under root; returns per-run exit codes (0 = synced)."""
    import subprocess
    import sys

    results = {}
    for run in offline_runs(root):
        if dry_run:
            results[str(run)] = None
            continue
        proc = subprocess.run([sys.executable, "-m", "wandb", "sync", str(run)], capture_output=True, text=True)
        results[str(run)] = {"returncode": proc.returncode, "output": (proc.stdout + proc.stderr)[-500:]}
    return results
