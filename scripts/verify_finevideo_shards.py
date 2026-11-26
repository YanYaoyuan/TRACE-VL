"""Match selected local shards to pinned public Git-LFS pointer metadata.

No network, parquet decoding, QA projection, model imports, or subprocesses.
Use an external process timeout for a hard wall-time limit.
"""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pointer_identity(sha256, size):
    """Standard v1 LFS pointer and its ordinary Git blob ID."""
    pointer = (f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n").encode()
    return hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer).hexdigest(), len(pointer)


def load_tree(path, expected_sha256, revision):
    raw = Path(path).read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("public tree file SHA256 does not match its declared pin")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase Git object ID")
    value = json.loads(raw)
    receipt = value["receipt"]
    if receipt.get("completed") is not True or receipt.get("status") != 200 or receipt.get("revision") != revision:
        raise ValueError("public tree receipt is incomplete or belongs to another revision")
    rows = {}
    for row in value["files"]:
        name = row["path"]
        if not re.fullmatch(r"data/train-\d{5}-of-\d{5}\.parquet", name) or name in rows:
            raise ValueError("invalid or duplicate shard path in tree")
        size = row["size"]
        pointer_size = row["lfs"]["pointerSize"]
        if (type(size) is not int or size <= 0 or size != row["lfs"]["size"]
                or type(pointer_size) is not int or not 0 < pointer_size < 1024
                or not re.fullmatch(r"[0-9a-f]{40}", row["oid"])):
            raise ValueError("invalid shard size or Git pointer metadata")
        # Public gated-repository metadata can mask lfs.oid. It is not required:
        # compute our local digest and compare the standard pointer's blob ID.
        rows[name] = {"path": name, "bytes": size, "pointer_blob_oid": row["oid"],
                      "pointer_bytes": pointer_size}
    if not rows or receipt.get("files") != len(rows):
        raise ValueError("public tree file count differs from its receipt")
    return {"revision": revision, "tree_sha256": expected_sha256, "rows": rows}


def local_path(root, relative):
    path = root / relative
    if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("selected shard must not be a symlink or escape the dataset root")
    return path


def make_plan(tree, selection, root, max_bytes):
    if (not isinstance(selection, list) or not selection or any(type(v) is not str for v in selection)
            or len(set(selection)) != len(selection)):
        raise ValueError("selection must be a nonempty unique list of exact data/shard paths")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if any(name not in tree["rows"] for name in selection):
        raise ValueError("selection contains a path absent from the pinned tree")
    selected = [tree["rows"][name] for name in sorted(selection)]
    total = sum(row["bytes"] for row in selected)
    if total > max_bytes:
        raise ValueError("selected shard bytes exceed the declared read budget")
    root = Path(root).resolve(strict=True)
    for row in selected:
        info = local_path(root, row["path"]).stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != row["bytes"]:
            raise ValueError("selected local shard is not a regular file of the expected size")
    plan = {"protocol": "finevideo-selected-shard-identity-v1", "root": str(root),
            "revision": tree["revision"], "tree_sha256": tree["tree_sha256"], "shards": selected,
            "expected_read_bytes": total, "max_bytes": max_bytes, "source_selection_sha256": digest(selection),
            "scope": "Declared local shard/revision membership only; not QA/media/split validity."}
    return {**plan, "id": digest(plan)}


def verify_file(root, row):
    path = local_path(root, row["path"])
    before = path.stat()
    sha256, count = hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("local shard changed before opening")
        while chunk := source.read(8 * 1024 * 1024):
            count += len(chunk)
            if count > row["bytes"]:
                raise ValueError("local shard grew beyond its declared size")
            sha256.update(chunk)
        final = os.fstat(source.fileno())
    after = path.stat()
    signatures = {(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                  for s in (before, opened, final, after)}
    if len(signatures) != 1 or count != row["bytes"]:
        raise ValueError("local shard changed during verification")
    oid, pointer_bytes = pointer_identity(sha256.hexdigest(), count)
    if oid != row["pointer_blob_oid"] or pointer_bytes != row["pointer_bytes"]:
        raise ValueError("local shard content does not match the pinned revision pointer")
    return {"path": row["path"], "bytes": count, "sha256": sha256.hexdigest(),
            "pointer_blob_oid": oid, "pointer_bytes": pointer_bytes,
            "observed_stat": {"device": after.st_dev, "inode": after.st_ino,
                              "mtime_ns": after.st_mtime_ns, "ctime_ns": after.st_ctime_ns}}


def write_json(path, value):
    temp = path.with_suffix(".partial.json")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def verify_plan(plan, out):
    if plan["id"] != digest({k: v for k, v in plan.items() if k != "id"}):
        raise ValueError("plan identity changed")
    records = []
    for row in plan["shards"]:
        records.append(verify_file(Path(plan["root"]), row))
        write_json(out / "progress.json", {"verified_shards": len(records), "total_shards": len(plan["shards"]),
                                           "verified_bytes": sum(r["bytes"] for r in records)})
    return {"verified": True, "plan_id": plan["id"], "revision": plan["revision"],
            "verified_shards": len(records), "verified_bytes": sum(r["bytes"] for r in records),
            "files": records, "network_requests": 0, "QA_values_read": False, "model_calls": 0,
            "scope": "All selected shard bytes were reread and matched declared revision pointers. "
                     "Unselected shards, QA/evidence correctness, frame PTS and source independence are unverified. "
                     "Receipts attest to the observed files; they are not a persistent cache-validity guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--tree-sha256", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True, help="JSON list of exact data/*.parquet paths")
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new, nonexisting receipt directory")
    parser.add_argument("--plan-only", action="store_true", help="stat selected files without hashing contents")
    args = parser.parse_args()
    tree = load_tree(args.tree, args.tree_sha256, args.revision)
    plan = make_plan(tree, json.loads(args.selection.read_text()), args.root, args.max_bytes)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "plan.json", plan)
    write_json(args.out / "status.json", {"stage": "planned" if args.plan_only else "verifying", "verified": False})
    if args.plan_only:
        print(json.dumps({"planned_shards": len(plan["shards"]), "read_bytes": plan["expected_read_bytes"],
                          "plan_id": plan["id"], "verified": False}))
        return
    try:
        result = verify_plan(plan, args.out)
        write_json(args.out / "report.json", result)
        write_json(args.out / "status.json", {"stage": "completed", "verified": True, "plan_id": plan["id"]})
    except BaseException as exc:
        write_json(args.out / "status.json", {"stage": "failed_or_interrupted", "verified": False,
                                              "error_type": type(exc).__name__, "plan_id": plan["id"]})
        raise
    print(json.dumps({k: result[k] for k in ("verified", "verified_shards", "verified_bytes", "plan_id")}))


if __name__ == "__main__":
    main()
