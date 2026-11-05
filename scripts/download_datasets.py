"""Pinned dataset acquisition. Default: offline preview; never extracts archives."""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import quote

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT / "configs/datasets/downloads.json"


def load_catalog(path=DEFAULT_CATALOG):
    path = Path(path).resolve()
    catalog = json.loads(path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported download catalog schema")
    audit = json.loads((path.parent / catalog["audit"]).resolve().read_text())
    return catalog, {r["requested_id"]: r for r in audit["datasets"]}


def safe_destination(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"destination escapes dataset root: {relative}")
    return path


def make_plan(names, root, catalog_path=DEFAULT_CATALOG):
    catalog, sources = load_catalog(catalog_path)
    selected, profiles = {}, []
    for name in names:
        if name not in catalog["profiles"]:
            raise ValueError(f"unknown download profile: {name}")
        profile = catalog["profiles"][name]
        if profile["provider"] == "hf":
            source = sources[profile["source_id"]]
            files = {}
            for pattern in profile["patterns"]:
                matches = [f for f in source["files"] if fnmatch.fnmatchcase(f["path"], pattern)]
                if not matches:
                    raise ValueError(f"profile {name}: pattern matches no audited files: {pattern}")
                files.update({f["path"]: f for f in matches})
            revision = source["revision"]
        elif profile["provider"] == "http":
            files = {f["path"]: f for f in profile["files"]}
            revision = profile["revision"]
        else:
            raise ValueError("unsupported source provider")
        count_bytes = 0
        for source_file in files.values():
            if not isinstance(source_file.get("bytes"), int) or source_file["bytes"] <= 0:
                raise ValueError("download profiles must name nonempty files with verified byte sizes")
            destination = safe_destination(root, Path(profile["directory"]) / source_file["path"])
            item = {
                **source_file,
                "provider": profile["provider"],
                "source_id": profile["source_id"],
                "revision": revision,
                "destination": str(destination),
                "local_dir": str(safe_destination(root, profile["directory"])),
            }
            existing = selected.get(str(destination))
            if existing is not None and existing != item:
                raise ValueError(f"conflicting source revisions target the same file: {destination}")
            selected[str(destination)] = item
            count_bytes += item["bytes"]
        profiles.append(
            {
                "name": name,
                "bytes": count_bytes,
                "role": profile["role"],
                "notes": profile["notes"],
                "suggested_data_space_gb_estimate": profile["suggested_data_space_gb_estimate"],
            }
        )
    return {
        "profiles": profiles,
        "files": list(selected.values()),
        "total_bytes": sum(f["bytes"] for f in selected.values()),
        "root": str(Path(root).resolve()),
    }


def validate_budget(plan, max_gb):
    if not math.isfinite(max_gb) or max_gb <= 0:
        raise ValueError("max-gb must be finite and positive")
    if plan["total_bytes"] > max_gb * 1e9:
        raise ValueError(
            f"selected {plan['total_bytes'] / 1e9:.3f} GB exceeds --max-gb {max_gb}; "
            "choose a pilot or set an explicit larger budget"
        )


def digests(path, size):
    sha256, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f"blob {size}\0".encode())
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024**2), b""):
            sha256.update(data)
            blob.update(data)
    return sha256.hexdigest(), blob.hexdigest()


def verify_file(path, item, expected_local_sha256=None):
    path = Path(path)
    if path.stat().st_size != item["bytes"]:
        raise ValueError(f"file size mismatch: {path}")
    sha256, blob = digests(path, item["bytes"])
    expected = item.get("sha256") or expected_local_sha256
    if expected is not None and sha256 != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    if item.get("git_blob_sha1") and blob != item["git_blob_sha1"]:
        raise ValueError(f"Git blob hash mismatch: {path}")
    return sha256


def receipt_path(path):
    return Path(str(path) + ".trace-vl.json")


def existing_is_verified(item):
    path = Path(item["destination"])
    if not path.exists():
        return False
    expected = None
    receipt = receipt_path(path)
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if any(record.get(k) != item[k] for k in ("provider", "source_id", "revision", "path")):
            raise ValueError(f"existing receipt belongs to a different source revision: {path}")
        expected = record["local_sha256"]
    if not (item.get("sha256") or item.get("git_blob_sha1") or expected):
        raise ValueError(f"existing file has no verifiable provenance: {path}; use a new output directory")
    verify_file(path, item, expected)
    return True


def disk_free(root):
    path = Path(root).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def transfer(item):
    destination = Path(item["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if item["provider"] == "hf":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("Install data support with uv sync --extra data") from e
        downloaded = Path(
            hf_hub_download(
                repo_id=item["source_id"],
                repo_type="dataset",
                revision=item["revision"],
                filename=item["path"],
                local_dir=item["local_dir"],
            )
        )
        if downloaded.resolve() != destination.resolve():
            raise ValueError("HF download returned an unexpected path")
        return verify_file(destination, item)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", delete=False
        ) as f:
            temporary = Path(f.name)
            with urllib.request.urlopen(item["url"], timeout=120) as response:
                shutil.copyfileobj(response, f)
        sha256 = verify_file(temporary, item)
        temporary.replace(destination)
        return sha256
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def transfer_curl(item):
    """Public-file fallback for environments where the HF HTTP/Xet clients fail."""
    destination = Path(item["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".download")
    if temporary.exists() and temporary.stat().st_size > item["bytes"]:
        raise ValueError(f"partial download exceeds expected size: {temporary}")
    if item["provider"] == "hf":
        url = (
            "https://huggingface.co/datasets/"
            + quote(item["source_id"], safe="/")
            + "/resolve/"
            + quote(item["revision"], safe="")
            + "/"
            + quote(item["path"], safe="/")
            + "?download=true"
        )
    else:
        url = item["url"]
    if not temporary.exists() or temporary.stat().st_size != item["bytes"]:
        subprocess.run(
            [
                "curl",
                "--http1.1",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "20",
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "--max-time",
                "1800",
                "--continue-at",
                "-",
                "--max-filesize",
                str(item["bytes"]),
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--silent",
                "--show-error",
                "--output",
                str(temporary),
                url,
            ],
            check=True,
        )
    sha256 = verify_file(temporary, item)
    temporary.replace(destination)
    return sha256


def download(plan, max_gb=10, reserve_gb=5, transport="hf"):
    if transport not in {"hf", "curl"}:
        raise ValueError("transport must be hf or curl")
    validate_budget(plan, max_gb)
    if not math.isfinite(reserve_gb) or reserve_gb < 0:
        raise ValueError("reserve-gb must be finite and nonnegative")
    # Verify every existing target before the first network call; never silently overwrite it.
    pending = [f for f in plan["files"] if not existing_is_verified(f)]
    needed = sum(f["bytes"] for f in pending)
    if needed + reserve_gb * 1e9 > disk_free(plan["root"]):
        raise ValueError("insufficient free disk for downloads plus reserve; extraction is not included")
    for item in pending:
        print(f"Downloading {item['source_id']} / {item['path']}", flush=True)
        sha256 = transfer(item) if transport == "hf" else transfer_curl(item)
        record = {
            **item,
            "local_sha256": sha256,
            "downloaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        receipt_path(item["destination"]).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"Verified {len(plan['files'])} files; downloaded {len(pending)}; archives were not extracted.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--root", type=Path, default=PROJECT / "data/raw")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download", action="store_true", help="actually transfer selected dataset files")
    parser.add_argument(
        "--transport",
        choices=["hf", "curl"],
        default="hf",
        help="curl is an optional resumable transport for public files",
    )
    parser.add_argument(
        "--max-gb", type=float, default=10, help="selected download bytes ceiling, decimal GB"
    )
    parser.add_argument(
        "--reserve-gb", type=float, default=5, help="leave this much free disk after transfers"
    )
    parser.add_argument("--show-files", action="store_true")
    args = parser.parse_args(argv)
    catalog, _ = load_catalog(args.catalog)
    if args.list or not args.profiles:
        for name in catalog["profiles"]:
            p = make_plan([name], args.root, args.catalog)
            print(f"{name:34} {p['total_bytes'] / 1e9:10.4f} GB  {p['profiles'][0]['role']}")
        return
    plan = make_plan(args.profiles, args.root, args.catalog)
    validate_budget(plan, args.max_gb)
    print("DOWNLOAD" if args.download else "OFFLINE PREVIEW: no network requests, no dataset files created")
    print(
        f"{len(plan['files'])} unique files; {plan['total_bytes'] / 1e9:.3f} GB "
        f"({plan['total_bytes'] / 2**30:.3f} GiB); root={plan['root']}"
    )
    for p in plan["profiles"]:
        print(f"{p['name']}: {p['notes']}")
    if args.show_files:
        for f in plan["files"]:
            print(f"  {f['source_id']} @ {f['revision']} / {f['path']} ({f['bytes']} bytes)")
    if args.download:
        download(plan, args.max_gb, args.reserve_gb, args.transport)


if __name__ == "__main__":
    main()
