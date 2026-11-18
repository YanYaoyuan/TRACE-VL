"""SwanLab scalar tracking and lossless UTF-8 report/log records for public cloud.

Public SwanLab does not support the self-hosted-only save() API. Text files are
therefore stored as ordered Text media with relative paths, byte counts and SHA256.
Only rank zero calls this interface; training finishes it after DDP shutdown.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from pathlib import Path

from .tracking import artifact_name


def scalars(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            result.update(scalars(value, name + "/"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[name] = value
    return result


class SwanTracker:
    def __init__(self, run=None, mode=None):
        self.run, self.mode, self.error = run, mode, None

    @classmethod
    def start(cls, enabled, name, group, job_type, config=None, tags=None, directory=None, *, run_id=None):
        if not enabled or os.environ.get("SWANLAB_MODE", "online") == "disabled":
            return cls()
        try:
            import swanlab
            directory = Path(directory or "swanlog")
            if directory.name == "wandb":
                directory = directory.with_name("swanlog")
            directory.mkdir(parents=True, exist_ok=True)
            requested = os.environ.get("SWANLAB_MODE", "online")
            settings = swanlab.Settings(interactive=False)
            run = swanlab.init(project=os.environ.get("TRACE_SWANLAB_PROJECT", "trace-vl-core"),
                workspace=os.environ.get("TRACE_SWANLAB_WORKSPACE") or None, experiment_name=name,
                group=os.environ.get("TRACE_SWANLAB_GROUP") or group, job_type=job_type,
                config=config or {}, tags=tags, logdir=str(directory), mode=requested,
                on_online_error="offline", settings=settings, reinit=True,
                **({"id": run_id, "resume": "allow"} if run_id else {}))
            tracker = cls(run, run.mode)
            url = swanlab.get_url() if tracker.mode == "online" else None
            print(f"SWANLAB mode={tracker.mode} experiment={name} url={url}", flush=True)
            if tracker.mode != requested:
                warnings.warn(f"SwanLab requested {requested}, actual {tracker.mode}; local data retained for sync")
            return tracker
        except Exception as error:  # noqa: BLE001 - local training logs remain authoritative
            warnings.warn(f"SwanLab init failed ({type(error).__name__}); tracking inactive, local logs retained")
            return cls()

    @property
    def active(self):
        return self.run is not None

    def _guard(self, what, fn):
        if self.run is None:
            return None
        try:
            return fn()
        except Exception as error:  # noqa: BLE001
            self.error = f"{what}: {type(error).__name__}"
            warnings.warn(f"SwanLab {what} failed ({type(error).__name__}); local logs retained")
            return None

    def log(self, values, step=None):
        clean = scalars(values)
        if clean:
            self._guard("log", lambda: self.run.log(clean, step=step))

    def log_media(self, values, step=None):
        self._guard("media", lambda: self.run.log(values, step=step))

    def text(self, key, value, *, caption=None, step=None):
        def write():
            import swanlab
            self.run.log({key: swanlab.Text(value, caption=caption)}, step=step)
        self._guard("text", write)

    def summary(self, values):
        self.log(values)
        self.text("summary/report", json.dumps(values, ensure_ascii=False, indent=2))

    def table(self, key, columns, rows):
        self.text(key, json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False, indent=2))

    def upload(self, name, kind, paths, metadata=None, base=None, names=None):
        if not self.active:
            return None
        result = {}
        used = {}
        def write():
            for path in map(Path, paths):
                if not path.exists():
                    continue
                # Callers upload explicit text files; recurse only when explicitly supplied.
                for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
                    if not file.is_file():
                        continue
                    target = (names or {}).get(str(file)) or artifact_name(file, base)
                    if target in used and used[target] != file:
                        raise ValueError("report path collision: " + target)
                    used[target] = file
                    data = file.read_bytes()
                    content = data.decode("utf-8")
                    parts = [content[i:i + 12000] for i in range(0, len(content), 12000)] or [""]
                    key = "files/" + hashlib.sha256(target.encode()).hexdigest()[:16]
                    for index, part in enumerate(parts):
                        self.text(key, part, caption=f"{target} | part {index + 1}/{len(parts)}", step=index)
                    result[target] = {"key": key, "parts": len(parts), "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()}
            self.text("files/manifest-" + hashlib.sha256(name.encode()).hexdigest()[:12],
                json.dumps({"name": name, "kind": kind, "metadata": metadata or {},
                            "files": result}, ensure_ascii=False, indent=2))
            return result
        return self._guard("text-file upload", write)

    def finish(self):
        if self.run is not None:
            self._guard("finish", lambda: self.run.finish())
            self.run = None
