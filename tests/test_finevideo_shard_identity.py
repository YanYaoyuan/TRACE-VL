import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "finevideo_shards", Path(__file__).parents[1] / "scripts/verify_finevideo_shards.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
REVISION = "84c74091e1c6ee7a5dffabfafb5c9033e4718883"


def fixture(tmp_path):
    root = tmp_path / "dataset"
    (root / "data").mkdir(parents=True)
    payload = b"local fixture bytes; not a parquet or QA record"
    name = "data/train-00000-of-01357.parquet"
    (root / name).write_bytes(payload)
    oid, pointer_bytes = audit.pointer_identity(hashlib.sha256(payload).hexdigest(), len(payload))
    tree = {"receipt": {"completed": True, "status": 200, "revision": REVISION, "files": 1},
            "files": [{"path": name, "size": len(payload), "oid": oid,
                       "lfs": {"oid": "*" * 64, "size": len(payload), "pointerSize": pointer_bytes}}]}
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(tree))
    return root, name, payload, tree, path


def load(path):
    return audit.load_tree(path, hashlib.sha256(path.read_bytes()).hexdigest(), REVISION)


def test_pointer_matches_independently_verified_public_revision_sample():
    # The 493,919,884-byte NAS shard was independently reread on A100; this
    # ordinary pointer blob ID is from the public HF tree at REVISION.
    assert audit.pointer_identity(
        "607bc6f45f6c4dcf27950abaf81eea58d7d50e2defd077efd0740647cf88baed", 493919884
    ) == ("e5fe2203ee33069a46a43661d510c9532678d437", 134)


def test_masked_remote_lfs_digest_still_verifies_all_selected_local_bytes(tmp_path):
    root, name, payload, _, tree = fixture(tmp_path)
    plan = audit.make_plan(load(tree), [name], root, len(payload))
    out = tmp_path / "receipts"
    out.mkdir()
    report = audit.verify_plan(plan, out)
    assert report["verified"] and report["verified_bytes"] == len(payload)
    assert report["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["network_requests"] == report["model_calls"] == 0 and not report["QA_values_read"]


def test_same_size_content_replacement_is_rejected(tmp_path):
    root, name, payload, _, tree = fixture(tmp_path)
    plan = audit.make_plan(load(tree), [name], root, len(payload))
    (root / name).write_bytes(b"X" * len(payload))
    with pytest.raises(ValueError, match="pinned revision pointer"):
        audit.verify_plan(plan, tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_tree_pin_is_checked_before_accepting_revision_claim(tmp_path):
    _, _, _, _, tree = fixture(tmp_path)
    expected = hashlib.sha256(tree.read_bytes()).hexdigest()
    tree.write_bytes(tree.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        audit.load_tree(tree, expected, REVISION)


@pytest.mark.parametrize("change", ["revision", "incomplete", "count"])
def test_other_revision_incomplete_or_truncated_tree_is_rejected(tmp_path, change):
    _, _, _, data, tree = fixture(tmp_path)
    key, value = {"revision": ("revision", "0" * 40), "incomplete": ("completed", False),
                  "count": ("files", 2)}[change]
    data["receipt"][key] = value
    tree.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load(tree)


@pytest.mark.parametrize("selection", [["../outside.parquet"], [], ["data/missing.parquet"]])
def test_selection_cannot_expand_beyond_explicit_pinned_shards(tmp_path, selection):
    root, _, _, _, tree = fixture(tmp_path)
    with pytest.raises(ValueError):
        audit.make_plan(load(tree), selection, root, 100000)


def test_duplicate_selection_and_excess_read_budget_are_rejected(tmp_path):
    root, name, payload, _, tree = fixture(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        audit.make_plan(load(tree), [name, name], root, 100000)
    with pytest.raises(ValueError, match="budget"):
        audit.make_plan(load(tree), [name], root, len(payload) - 1)


def test_external_symlink_is_not_admitted_even_with_identical_bytes(tmp_path):
    root, name, payload, _, tree = fixture(tmp_path)
    external = tmp_path / "external.parquet"
    external.write_bytes(payload)
    (root / name).unlink()
    (root / name).symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        audit.make_plan(load(tree), [name], root, 100000)


def test_tampered_plan_is_rejected_before_reading(tmp_path):
    root, name, payload, _, tree = fixture(tmp_path)
    plan = audit.make_plan(load(tree), [name], root, len(payload))
    plan["max_bytes"] += 1
    with pytest.raises(ValueError, match="identity"):
        audit.verify_plan(plan, tmp_path)


def test_plan_only_never_hashes_or_claims_verified_content(tmp_path, monkeypatch):
    root, name, _, _, tree = fixture(tmp_path)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps([name]))
    out = tmp_path / "plan-output"
    monkeypatch.setattr(sys, "argv", ["verify", "--tree", str(tree), "--tree-sha256",
                                     hashlib.sha256(tree.read_bytes()).hexdigest(), "--revision", REVISION,
                                     "--root", str(root), "--selection", str(selection), "--max-bytes", "100000",
                                     "--out", str(out), "--plan-only"])
    def forbidden(*args):
        raise AssertionError("plan-only must not read selected shard content")
    monkeypatch.setattr(audit, "verify_file", forbidden)
    audit.main()
    assert json.loads((out / "status.json").read_text()) == {"stage": "planned", "verified": False}
    assert not (out / "report.json").exists()
    with pytest.raises(FileExistsError):
        audit.main()


def test_cli_failure_retains_negative_receipt_without_success_report(tmp_path, monkeypatch):
    root, name, payload, _, tree = fixture(tmp_path)
    (root / name).write_bytes(b"X" * len(payload))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps([name]))
    out = tmp_path / "failed-output"
    monkeypatch.setattr(sys, "argv", ["verify", "--tree", str(tree), "--tree-sha256",
                                     hashlib.sha256(tree.read_bytes()).hexdigest(), "--revision", REVISION,
                                     "--root", str(root), "--selection", str(selection), "--max-bytes", "100000",
                                     "--out", str(out)])
    with pytest.raises(ValueError, match="pointer"):
        audit.main()
    assert not json.loads((out / "status.json").read_text())["verified"]
    assert not (out / "report.json").exists()
