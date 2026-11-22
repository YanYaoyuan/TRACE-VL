"""Share exact frame probing across evaluation shards and stage slow reads on demand."""
from __future__ import annotations

import fcntl
import json
import subprocess
from pathlib import Path

from trace_vl.core_vlm.data import decode_observation, probe_video, sha256
from trace_vl.core_vlm.prepare import digest, write_json

DECODE_POLICY = "shared-pts-cache-local-retry-v1"


def media_stat(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


class VideoProbeCache:
    def __init__(self, directory, fast_timeout=180, retry_timeout=900):
        if min(fast_timeout, retry_timeout) <= 0:
            raise ValueError("video probe timeouts must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fast_timeout = fast_timeout
        self.retry_timeout = retry_timeout

    def _copy_local(self, source, signature, local, expected_hash):
        # Only the timed-out source is copied. A successful copy is published
        # under its original file suffix for FFmpeg's format detection.
        temporary = local.with_suffix(".partial")
        try:
            subprocess.run(["cp", "--", str(source), str(temporary)], check=True, capture_output=True,
                           timeout=self.retry_timeout)
            if media_stat(source) != signature or temporary.stat().st_size != signature[2]:
                raise ValueError(f"source changed while copying: {source}")
            actual_hash = sha256(temporary)
            if expected_hash is not None and expected_hash != actual_hash:
                raise ValueError(f"local copy has different bytes: {source}")
            temporary.replace(local)
            return actual_hash
        finally:
            temporary.unlink(missing_ok=True)

    def source(self, original, *, stage=False):
        source = Path(original).resolve(strict=True)
        signature = media_stat(source)
        key = digest([str(source), signature])
        index = self.directory / f"{key}.json"
        local = self.directory / f"{key}.media{source.suffix}"
        with (self.directory / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = json.loads(index.read_text()) if index.exists() else None
            if cached is None and not stage:
                try:
                    metadata = probe_video(source, self.fast_timeout)
                except subprocess.TimeoutExpired:
                    stage = True
                else:
                    cached = {"metadata": metadata, "sha256": sha256(source), "local": False}
            if stage and (cached is None or not cached["local"] or not local.is_file()):
                print(f"MEDIA local retry {source}", flush=True)
                checksum = self._copy_local(source, signature, local, cached["sha256"] if cached else None)
                metadata = cached["metadata"] if cached else probe_video(local, self.retry_timeout)
                cached = {"metadata": metadata, "sha256": checksum, "local": True}
            if cached["local"] and not local.is_file():
                # The index still describes the original media if local scratch
                # was cleared between evaluations.
                cached["local"] = False
            if media_stat(source) != signature:
                raise ValueError(f"source changed during video probing: {source}")
            if not index.exists() or stage:
                write_json(index, cached)
            return local if cached["local"] else source, cached["metadata"], cached["sha256"]

    def decode(self, row, **kwargs):
        source = Path(row["video"]).resolve(strict=True)
        signature = media_stat(source)
        media = self.source(source)
        try:
            result = decode_observation(row, **kwargs, media=media,
                                        timeout=self.fast_timeout if media[0] == source else self.retry_timeout)
        except subprocess.TimeoutExpired:
            if media[0] != source:
                raise
            result = decode_observation(row, **kwargs, media=self.source(source, stage=True),
                                        timeout=self.retry_timeout)
        if media_stat(source) != signature:
            raise ValueError(f"source changed during decoding: {source}")
        return result
