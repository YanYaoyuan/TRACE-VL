import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from trace_vl.core_vlm.data import read_split, sha256
from trace_vl.core_vlm.prepare import (
    alias_groups,
    aliases,
    annotation_rows,
    parse_pair,
    prepare,
    publish_manifest,
    source_group,
    split_for,
    unique_records,
    verify_video,
)


@pytest.fixture(scope="module")
def video_bytes(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    root = tmp_path_factory.mktemp("core-media")
    result = []
    for i in range(8):
        p = root / f"{i}.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        f"color=c=0x{i * 25:02x}4080:s=64x64:r=4:d=2", "-threads", "1",
                        "-c:v", "ffv1", str(p)], check=True)
        result.append(p.read_bytes())
    return result


def item(video, question="What is visible?", answer="blue"):
    return {"id": video, "video": video, "conversations": [
        {"from": "human", "value": "<image>\n" + question}, {"from": "gpt", "value": answer}]}


@pytest.fixture
def dataset(tmp_path, video_bytes):
    llava, benchmark = tmp_path / "llava", tmp_path / "benchmark"
    subset = llava / "0_30_s_youtube_v0_1"
    subset.mkdir(parents=True)
    benchmark.mkdir()
    (benchmark / "annotations.json").write_text(json.dumps([{"video_id": "blocked1234"}]))
    rows = [item(f"source/v_{i:011d}.mkv") for i in range(8)]
    # Two renamed exact-byte copies must not inflate evaluation questions/videos.
    rows.extend([item("source/copy.mkv"), item("source/v_blocked1234.mp4")])
    (subset / "sample_oe_qa_processed.json").write_text(json.dumps(rows))
    with zipfile.ZipFile(subset / "videos-1.zip", "w") as z:
        for i in range(4):
            z.writestr(rows[i]["video"], video_bytes[i])
        z.writestr("source/copy.mkv", video_bytes[0])
    with zipfile.ZipFile(subset / "videos-2.zip", "w") as z:
        for i in range(4, 8):
            z.writestr(rows[i]["video"], video_bytes[i])
    cfg = {"llava_root": str(llava), "sources": [subset.name], "benchmark_roots": [str(benchmark)],
           "exclude_source_patterns": [], "seed": 7, "val_fraction": .5, "max_videos": 50,
           "max_questions_per_video": 6, "max_question_chars": 2048, "max_answer_chars": 512,
           "minimum_frames": 8, "workers": 2, "archive_workers": 2, "reserve_gb": 0,
           "min_videos": {"train": 1, "val": 1}, "fail_on_malformed": True}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    return config, tmp_path / "prepared", subset, benchmark


def test_real_archive_decode_disjoint_manifest_and_idempotent_resume(dataset, monkeypatch):
    config, out, subset, benchmark = dataset
    originals = {str(p): sha256(p) for root in (subset, benchmark) for p in root.rglob("*") if p.is_file()}
    report = prepare(config, out)
    assert report["prepared"] and report["verified_media_paths"] == 9
    assert report["selected_source_groups"] == 8
    assert sum(c["questions"] for c in report["counts"].values()) == 8
    train, val = read_split(out / "manifest.jsonl", "train"), read_split(out / "manifest.jsonl", "val")
    assert {r["source_group"] for r in train}.isdisjoint({r["source_group"] for r in val})
    assert {sha256(r["video"]) for r in train}.isdisjoint({sha256(r["video"]) for r in val})
    assert all("blocked1234" not in r["relative_video"] for r in train + val)
    assert all(Path(r["video"]).is_file() for r in train + val)
    fingerprint = sha256(out / "manifest.jsonl")
    import trace_vl.core_vlm.prepare as module

    def no_decode(*args):
        raise AssertionError("a verified unchanged video should reuse its full-decode cache")

    monkeypatch.setattr(module, "verify_video", no_decode)
    again = prepare(config, out)
    assert again["prepared"] and sha256(out / "manifest.jsonl") == fingerprint
    assert originals == {p: sha256(p) for p in originals}


def test_missing_selected_media_does_not_refill_or_claim_complete(dataset):
    config, out, subset, _ = dataset
    annotation = subset / "sample_oe_qa_processed.json"
    annotation.write_text(json.dumps(json.loads(annotation.read_text()) + [item("source/unavailable.mp4")]))
    with pytest.raises(ValueError, match="unavailable"):
        prepare(config, out)
    assert not (out / "complete.json").exists()
    assert "unavailable.mp4" in (out / "missing.json").read_text()


def test_duplicate_archive_paths_are_not_arbitrarily_selected(dataset, video_bytes):
    config, out, subset, _ = dataset
    with zipfile.ZipFile(subset / "collision.zip", "w") as z:
        z.writestr("source/v_00000000001.mkv", video_bytes[3])
    with pytest.raises(ValueError, match="ambiguous archive members"):
        prepare(config, out)
    assert not (out / "complete.json").exists()


def test_decode_gate_refuses_short_or_corrupt_video(tmp_path, video_bytes):
    path = tmp_path / "short.mkv"
    path.write_bytes(video_bytes[0])
    assert verify_video(path, 8)["frames"] == 8
    with pytest.raises(ValueError, match="insufficient frames"):
        verify_video(path, 9)
    path.write_bytes(b"not a video")
    with pytest.raises(subprocess.CalledProcessError):
        verify_video(path, 1)


def test_manifest_is_published_before_completion_and_never_silently_repaired(tmp_path, monkeypatch):
    import trace_vl.core_vlm.prepare as module
    original = module.ensure_run_manifest
    manifest = '{"id":"one"}\n'
    import hashlib
    completion = {"id": "one", "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}

    def check_before_marker(path, value):
        assert (tmp_path / "manifest.jsonl").read_text() == manifest
        original(path, value)

    monkeypatch.setattr(module, "ensure_run_manifest", check_before_marker)
    publish_manifest(tmp_path, manifest, completion)
    publish_manifest(tmp_path, manifest, completion)
    (tmp_path / "manifest.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="missing or changed"):
        publish_manifest(tmp_path, manifest, completion)
    assert (tmp_path / "manifest.jsonl").read_text() == "corrupted"


def test_qa_parser_first_pair_has_no_gold_history_and_rejects_ambiguous_mc(tmp_path):
    mc = parse_pair("<video>\nWhat happens first?\nA. open\nB. close\nAnswer with a letter.", "B", "mc")
    assert mc["answer"] == 1 and mc["options"] == ["open", "close"]
    with pytest.raises(ValueError):
        parse_pair("Which?\nA. x\nC. y", "A", "mc")
    row = item("x.mp4")
    row["conversations"] += [{"from": "human", "value": "What about that?"},
                             {"from": "gpt", "value": "secret later answer"}]
    path = tmp_path / "qa.json"
    path.write_text(json.dumps([row]))
    rows, rejected = annotation_rows(path, "source", "oe", set(), {
        "exclude_source_patterns": [], "max_question_chars": 4096, "max_answer_chars": 512})
    assert len(rows) == 1 and rejected["later_dialogue_turns_not_used"] == 1
    assert "secret" not in json.dumps(rows) and "What about" not in json.dumps(rows)


@pytest.mark.parametrize("footer,answer", [
    ("Please respond with only the letter of the correct answer.", "B."),
    ("Please provide your answer by stating the letter followed by the full option.", "B. The door closes."),
])
def test_official_youtube_mc_response_formats_reach_annotation_manifest(tmp_path, footer, answer):
    question = "What happens?\nA. The door opens\nB. The door closes\n" + footer
    path = tmp_path / "youtube_mc_qa.json"
    path.write_text(json.dumps([item("sample.mp4", question, answer)]))
    rows, rejected = annotation_rows(path, "youtube", "mc", set(), {
        "exclude_source_patterns": [], "max_question_chars": 4096, "max_answer_chars": 512})
    assert len(rows) == 1 and not rejected.get("malformed")
    assert rows[0]["question"] == "What happens?"
    assert rows[0]["options"] == ["The door opens", "The door closes"]
    assert rows[0]["answer"] == 1


@pytest.mark.parametrize("answer", ["B. opens.", "A. closes.", "C.", "B because it closes", "B. closes. Explanation"])
def test_full_option_answers_must_agree_with_the_letter(answer):
    with pytest.raises(ValueError):
        parse_pair("What?\nA. opens\nB. closes", answer, "mc")


def test_unknown_footer_and_options_after_instructions_remain_errors():
    with pytest.raises(ValueError, match="ambiguous text"):
        parse_pair("What?\nA. opens\nB. closes\nPlease provide the secret answer: B", "B", "mc")
    with pytest.raises(ValueError, match="after response instruction"):
        parse_pair("What?\nA. opens\nAnswer with a letter.\nB. closes", "B", "mc")
    assert parse_pair("What?\nA. opens?\nB. closes", "A. opens?.", "mc")["answer"] == 0


def test_aliases_and_conflicting_gold_do_not_cross_split():
    assert "abc12345678" in aliases("https://youtu.be/abc12345678?t=3")
    assert "abc12345678" in aliases("clips/abc12345678-Scene-001.mp4")
    assert source_group("v_abc12345678.mp4")[0] == source_group("abc12345678-Scene-002.mp4")[0]
    rows = [{"id": str(i), "source_group": group, "source_aliases": keys,
             "question": "What?", "answer_text": answer} for i, (group, keys, answer) in enumerate([
                 ("a", ["a", "b"], "one"), ("c", ["b", "c"], "two")])]
    grouped = alias_groups(rows)
    assert len({r["source_group"] for r in grouped}) == 1
    with pytest.raises(ValueError, match="conflicting answers"):
        unique_records(grouped)
    # Split ownership depends on source identity, never the selected pool length.
    short = {str(i): split_for(str(i), 7, .2) for i in range(10)}
    long = {str(i): split_for(str(i), 7, .2) for i in range(100)}
    assert all(long[k] == v for k, v in short.items())


def test_conflicting_copies_all_quarantined_but_different_scene_answers_retained():
    from trace_vl.core_vlm.prepare import with_group
    def row(clip, answer):
        return with_group({"relative_video": clip, "question": "What color?", "answer_text": answer}, "parent")

    conflicts = []
    kept = unique_records([row("scene-1.mp4", "blue"), row("scene-1.mp4", "red"),
                           row("scene-2.mp4", "red"), row("scene-3.mp4", "blue")], conflicts)
    assert {r["relative_video"] for r in kept} == {"scene-2.mp4", "scene-3.mp4"}
    assert len(conflicts) == 1 and len(conflicts[0]["copies"]) == 2


def test_changed_completed_dataset_and_insufficient_counts_cannot_authorize_training(dataset):
    config, out, _, _ = dataset
    cfg = json.loads(config.read_text())
    cfg["min_videos"] = {"train": 100, "val": 100}
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="insufficient split counts"):
        prepare(config, out)
    assert not (out / "complete.json").exists()
    assert json.loads((out / "report.json").read_text())["prepared"] is False
