"""Offline checks of the dataset plan, transfer limits, provenance and safe paths."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_datasets.py"
spec = importlib.util.spec_from_file_location("dataset_downloads", SCRIPT)
downloads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloads)


def test_all_download_profiles_resolve_audited_files(tmp_path):
    catalog, _ = downloads.load_catalog()
    for name in catalog["profiles"]:
        plan = downloads.make_plan([name], tmp_path)
        assert plan["files"] and plan["total_bytes"] > 0
        assert all(f["revision"] for f in plan["files"])
        if name.startswith("ovo-"):
            parts = [f for f in plan["files"] if ".tar.part" in f["path"]]
            assert len(parts) == (5 if name == "ovo-source" else 15)
        if name == "fastbench":
            assert sum(f["path"].endswith(".mp4") for f in plan["files"]) == 300
        if name == "llava-activitynet-pilot":
            assert len(plan["files"]) == 2
            assert plan["total_bytes"] == 928035564


def test_overlapping_profiles_do_not_double_count(tmp_path):
    alone = downloads.make_plan(["llava-short-pilot"], tmp_path)
    combined = downloads.make_plan(["llava-short-pilot", "llava-annotations"], tmp_path)
    assert combined["total_bytes"] == alone["total_bytes"]
    assert len(combined["files"]) == len(alone["files"])


def test_preview_does_not_download_or_create_data_directory(tmp_path, monkeypatch, capsys):
    def blocked(*args, **kwargs):
        raise AssertionError("network is forbidden in preview")

    monkeypatch.setattr(downloads.urllib.request, "urlopen", blocked)
    root = tmp_path / "not_created"
    downloads.main(["uca-annotations", "--root", str(root)])
    assert not root.exists()
    assert "OFFLINE PREVIEW" in capsys.readouterr().out


def test_large_plan_is_rejected_before_transfer(tmp_path, monkeypatch):
    plan = downloads.make_plan(["ovo-chunked"], tmp_path)

    def blocked(*args, **kwargs):
        raise AssertionError("transfer happened before budget check")

    monkeypatch.setattr(downloads, "transfer", blocked)
    with pytest.raises(ValueError, match="exceeds"):
        downloads.download(plan, max_gb=10)


def test_insufficient_disk_is_rejected_before_transfer(tmp_path, monkeypatch):
    plan = downloads.make_plan(["uca-annotations"], tmp_path)
    monkeypatch.setattr(downloads, "disk_free", lambda _: 0)
    with pytest.raises(ValueError, match="insufficient free disk"):
        downloads.download(plan)


def test_destination_cannot_escape_dataset_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        downloads.safe_destination(tmp_path, "../outside.mp4")


def test_integrity_receipt_and_resume_without_network(tmp_path, monkeypatch):
    payload = b"local software fixture, not a downloaded dataset"
    item = {
        "destination": str(tmp_path / "clip.bin"),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "provider": "http",
        "source_id": "local-fixture",
        "revision": "fixture-v1",
        "path": "clip.bin",
    }
    plan = {"files": [item], "total_bytes": len(payload), "root": str(tmp_path)}
    calls = []

    def fake_transfer(record):
        calls.append(record)
        Path(record["destination"]).write_bytes(payload)
        return downloads.verify_file(record["destination"], record)

    monkeypatch.setattr(downloads, "transfer", fake_transfer)
    downloads.download(plan, max_gb=1, reserve_gb=0)
    downloads.download(plan, max_gb=1, reserve_gb=0)
    assert len(calls) == 1
    receipt = json.loads(downloads.receipt_path(item["destination"]).read_text())
    assert receipt["local_sha256"] == item["sha256"]
    Path(item["destination"]).write_bytes(b"X" * len(payload))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        downloads.download(plan, max_gb=1, reserve_gb=0)
    assert len(calls) == 1


def test_git_blob_hash_verification(tmp_path):
    payload = b"annotation example"
    path = tmp_path / "annotation.json"
    path.write_bytes(payload)
    item = {
        "bytes": len(payload),
        "git_blob_sha1": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
    }
    assert downloads.verify_file(path, item) == hashlib.sha256(payload).hexdigest()


def test_curl_transport_verifies_before_promoting_partial(tmp_path, monkeypatch):
    payload = b"public-file software fixture"
    destination = tmp_path / "clip.tar.gz"
    item = {
        "destination": str(destination),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "provider": "hf",
        "source_id": "fixture/dataset",
        "revision": "fixed-revision",
        "path": "clips/clip.tar.gz",
    }
    commands = []

    def fake_run(command, check):
        commands.append(command)
        assert check and "--continue-at" in command
        Path(command[command.index("--output") + 1]).write_bytes(payload)

    monkeypatch.setattr(downloads.subprocess, "run", fake_run)
    plan = {"files": [item], "total_bytes": len(payload), "root": str(tmp_path)}
    downloads.download(plan, max_gb=1, reserve_gb=0, transport="curl")
    assert destination.read_bytes() == payload
    assert "/resolve/fixed-revision/clips/clip.tar.gz" in commands[0][-1]
    assert downloads.receipt_path(destination).exists()
    destination.unlink()
    Path(str(destination) + ".download").write_bytes(b"X" * len(payload))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        downloads.transfer_curl(item)
    assert not destination.exists()
    assert len(commands) == 1
