"""Prepare the reviewed state-attention smoke clips from existing ActivityNet media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from trace_vl.v3.ovo_transfer import ensure_media
from trace_vl.v3.run_io import atomic_text

from .data import sha256


def prepare_media(pairs_path, inventory_path, output, media_root, archive_path):
    inventory = json.loads(inventory_path.read_text())
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    locked = inventory["videos"]
    names = {Path(row["video"]).name for row in rows}
    if names != set(locked) or not names:
        raise ValueError("reviewed pairs and media inventory must cover the same clips")
    for name, entry in locked.items():
        path = PurePosixPath(entry["member"])
        if path.is_absolute() or ".." in path.parts or path.name != name:
            raise ValueError(f"invalid archive member for {name}")

    def verify(path):
        entry = locked.get(path.name)
        return entry is not None and sha256(path) == entry["sha256"]

    output.mkdir(parents=True, exist_ok=True)
    archive = archive_path.resolve()
    catalog = [{"name": str(archive), "parts": [archive]}] if archive.is_file() else []
    report = ensure_media(
        sorted(entry["member"] for entry in locked.values()), media_root, [], output,
        probe=lambda path: 1 if verify(path) else None, archive_catalog=catalog,
    )
    atomic_text(output / "media.json", json.dumps(report, indent=2) + "\n")
    if report["still_missing"] or report["ambiguous"] or report["extracted_but_unreadable"]:
        raise ValueError(f"reviewed media not recovered from {media_root} or {archive}. "
                         f"Inspect {output / 'media.json'} and set TRACE_ACTIVITYNET_ROOT or TRACE_ACTIVITYNET_ARCHIVE.")
    overlay = Path(report["overlay"])
    for row in rows:
        row["video"] = str(overlay / locked[Path(row["video"]).name]["member"])
    atomic_text(output / "pairs.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    print(f"MEDIA VERIFIED {len(locked)} clips / {len(rows)} pairs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "inventory", "out", "media-root", "archive"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    prepare_media(args.pairs, args.inventory, args.out, args.media_root, args.archive)


if __name__ == "__main__":
    main()
