"""Share frozen visual observations across training questions and experiment arms."""
from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .data import decode_observation
from .native import NativeLayout, check_native_forward, encode_native, prepare_native
from .prepare import digest
from .qa import VisualStep


def observation_id(row):
    return digest([row["video"], row.get("media_sha256"), row.get("observation", {}),
                   row.get("query_time"), row.get("protocol")])


class SharedVisionSamples:
    """Build one dense feature tensor and native prompt layouts per observed video window."""

    def __init__(self, backbone, args, rows, run):
        self.bb, self.args = backbone, args
        config = run["config"]
        keys = ("steps", "pixels", "timestamp_policy")
        identity = {"manifest": run["manifest_sha256"], "ids": run["ids"], "videos": run["videos"],
                    "model": run["model"], "torch": run["torch"], "transformers": run["transformers"],
                    "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_capability(self.bb.device) if self.bb.device.type == "cuda" else "cpu",
                    "dtype": str(backbone.dtype), "observation": {k: config[k] for k in keys},
                    "code": {k: run["code"][k] for k in ("native.py", "data.py", "feature_cache.py",
                                                            "../v3/backbone.py")}}
        self.root = Path(args.feature_cache) / digest(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_by_id, self.groups = {}, defaultdict(list)
        for row in rows:
            key = observation_id(row)
            self.key_by_id[row["id"]] = key
            self.groups[key].append(row)
        self.hits = self.writes = 0
        self.check_pending = args.native_check

    def _path(self, key):
        return self.root / key[:2] / f"{key}.pt"

    def _decode(self, row):
        return decode_observation(row, steps=self.args.steps, pixels=self.args.pixels)

    def _read_or_decode(self, key):
        path = self._path(key)
        if path.is_file():
            entry = torch.load(path, map_location="cpu", weights_only=True)
            if entry["key"] != key:
                raise ValueError("cached feature entry belongs to another observation")
            return entry, None
        return None, self._decode(self.groups[key][0])

    def _layout(self, decoded, row):
        images, times, resolved, audit = decoded
        record = {**row, "query_time": resolved["query_time"]}
        layout = prepare_native(self.bb.processor, images, times, record, audit,
                                allow_native_timestamp_approximation=self.args.timestamp_policy == "native-index-fps")
        return layout, record

    def _encode(self, layout):
        self.bb.visual.eval()
        return encode_native(self.bb, layout)

    def _make_entry(self, key, decoded):
        rows = self.groups[key]
        first = rows[0]
        layout, _ = self._layout(decoded, first)
        features = self._encode(layout)
        layouts = {first["id"]: layout}
        for row in rows[1:]:
            other, _ = self._layout(decoded, row)
            if not torch.equal(other.inputs["pixel_values_videos"], layout.inputs["pixel_values_videos"]):
                raise ValueError("questions from one observation produced different vision input tensors")
            layouts[row["id"]] = other
        return {"key": key, "features": torch.stack([s.tokens.cpu() for s in features]),
                "grid": features[0].grid, "frame_times": layout.frame_times,
                "query_time": decoded[2]["query_time"],
                "layouts": {
                    qid: {"inputs": {k: v for k, v in x.inputs.items() if k != "pixel_values_videos"},
                          "question": x.question, "audit": x.audit}
                    for qid, x in layouts.items()}}

    def _publish(self, key, entry):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        try:
            torch.save(entry, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _sample(self, row, entry):
        saved = entry["layouts"][row["id"]]
        times = entry["frame_times"]
        layout = NativeLayout(saved["inputs"], times, saved["question"], saved["audit"])
        features = [VisualStep(tokens, entry["grid"], pair[-1], tuple(pair))
                    for tokens, pair in zip(entry["features"], times, strict=True)]
        return features, {**row, "query_time": entry["query_time"], "native_layout": layout}

    def _verify(self, row, entry, decoded):
        layout, record = self._layout(self._decode(row) if decoded is None else decoded, row)
        visual = self._encode(layout)
        cached, cached_row = self._sample(row, entry)
        expected = {k: v for k, v in layout.inputs.items() if k != "pixel_values_videos"}
        saved = cached_row["native_layout"]
        if (expected.keys() != saved.inputs.keys()
                or any(not torch.equal(v, saved.inputs[k]) for k, v in expected.items())
                or layout.frame_times != saved.frame_times or layout.audit != saved.audit
                or any(not torch.equal(a.tokens.cpu(), b.tokens) for a, b in zip(visual, cached, strict=True))):
            raise ValueError("cached visual features or prompt layout differ from fresh encoding")
        self.bb.language_model.eval()
        check_native_forward(self.bb, layout, visual, record)
        self.check_pending = False
        print("VERIFIED SHARED VISION CACHE PARITY", flush=True)

    def batch(self, rows):
        representatives = {self.key_by_id[row["id"]]: row for row in rows}
        with ThreadPoolExecutor(min(len(representatives), self.args.decode_workers)) as pool:
            prepared = list(pool.map(self._read_or_decode, representatives))
        entries = {}
        for (key, row), (entry, decoded) in zip(representatives.items(), prepared, strict=True):
            new = entry is None
            if new:
                entry = self._make_entry(key, decoded)
            if self.check_pending:
                self._verify(row, entry, decoded)
            if new:
                self._publish(key, entry)
                self.writes += 1
            else:
                self.hits += 1
            entries[key] = entry
        return [self._sample(row, entries[self.key_by_id[row["id"]]]) for row in rows]
