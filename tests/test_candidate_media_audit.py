import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("candidate_audit", Path(__file__).parents[1] / "scripts/audit_streaming_candidates.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def archive_candidate(tmp_path, payload=b"candidate-media", sample=251):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("nested/sample_251/video.mp4", payload)
    with zipfile.ZipFile(archive) as handle:
        item = handle.infolist()[0]
    return {"sample": sample, "archive": str(archive), "member": item.filename,
            "bytes": len(payload), "crc32": item.CRC}


def test_stage_has_fixed_destination_hash_crc_and_does_not_overwrite(tmp_path):
    candidate = archive_candidate(tmp_path)
    out = tmp_path / "audit"
    row = audit.stage_one(candidate, out, lambda _: {"valid": True, "duration_seconds": 400.})
    assert Path(row["staged_path"]) == out / "media/sample_251.mp4"
    assert row["sha256"] == hashlib.sha256(b"candidate-media").hexdigest()
    assert Path(row["staged_path"]).read_bytes() == b"candidate-media" and row["zip_crc_verified"]
    with pytest.raises(RuntimeError, match="overwrite"):
        audit.stage_one(candidate, out)


def test_corrupt_zip_payload_fails_without_admitting_partial_media(tmp_path):
    candidate = archive_candidate(tmp_path)
    archive = Path(candidate["archive"])
    archive.write_bytes(archive.read_bytes().replace(b"candidate-media", b"Candidate-media", 1))
    with pytest.raises(zipfile.BadZipFile):
        audit.stage_one(candidate, tmp_path / "audit")
    assert list((tmp_path / "audit/media").iterdir()) == []


def test_probe_failure_is_an_ineligible_data_finding(tmp_path):
    candidate = archive_candidate(tmp_path)
    def invalid(_):
        raise ValueError("header is not usable")
    row = audit.stage_one(candidate, tmp_path / "audit", invalid)
    assert row["header"]["valid"] is False and row["zip_crc_verified"]
    assert Path(row["staged_path"]).exists()


def test_summary_excludes_overlaps_entire_duplicate_groups_and_bad_headers(tmp_path, monkeypatch):
    candidates, old_media, hashes = [], {}, {}
    for sample in range(251, 501):
        h = hashlib.sha256(str(sample).encode()).hexdigest()
        if sample == 254:
            h = hashes[253]
        hashes[sample] = h
        candidates.append({"sample": sample, "bytes": 1, "queries": [
            {"task": "Action Perception", "clue": "Concurrent", "cutoff": 60.}]})
        path = tmp_path / "media" / f"{sample}.mp4"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
        audit.atomic_json(tmp_path / "candidates" / f"{sample}.json", {
            "sample": sample, "bytes": 1, "sha256": h, "zip_crc_verified": True, "staged_path": str(path),
            "header": {"valid": sample != 255, "duration_seconds": 400.}})
    for sample in range(1, 251):
        old_media[str(sample)] = "unused"
        h = hashes[252] if sample == 1 else hashlib.sha256(f"old{sample}".encode()).hexdigest()
        audit.atomic_json(tmp_path / "old_hashes" / f"{sample}.json", {"sample": sample, "sha256": h})
    monkeypatch.setattr(audit, "load", lambda _: {"id": "fixture", "candidates": candidates,
                        "old_media": old_media, "training_hashes": [hashes[251]], "staged_bytes_planned": 250})
    result = audit.summarize(tmp_path)
    assert result["byte_overlap"] == {"training": [251], "old_development": [252]}
    assert result["duplicate_candidate_groups"] == [[253, 254]]
    assert result["byte_disjoint_unique_candidates"] == 246 and result["invalid_header_samples"] == [255]
    assert result["header_eligibility"]["strict_action_object"]["sources_header_eligible"] == 245
    assert not result["full_pts_verified"] and not result["underlying_source_independence_verified"]
    assert json.loads((tmp_path / "report.json").read_text())["model_calls"] == 0
