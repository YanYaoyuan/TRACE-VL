"""CPU test of the read-only data audit on a synthetic A100-like layout."""

import json

import numpy as np

from trace_vl.v3.audit import digest, run_audit


def write_video(path, frames=20, fps=4):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24))
    for i in range(frames):
        writer.write(np.full((24, 32, 3), i, dtype=np.uint8))
    writer.release()


def build(tmp_path):
    media, data, llava, root = tmp_path / "llava" / "media", tmp_path / "manifests", tmp_path / "llava", tmp_path / "data"
    for name in ("a", "b", "c", "unused1", "unused2"):
        write_video(media / "yt" / f"{name}.avi")
    data.mkdir()
    rows = {"train": [("t1", "yt/a.avi"), ("t2", "yt/a.avi"), ("t3", "yt/b.avi")], "val": [("v1", "yt/c.avi"), ("v2", "yt/gone.avi")]}
    for split, items in rows.items():
        (data / f"{split}.jsonl").write_text("".join(json.dumps({"id": i, "video": v, "split": split, "task": "mc",
                                                                 "options": ["x", "y"]}) + "\n" for i, v in items))
    sub = llava / "0_30_s_youtube_v0_1"
    sub.mkdir(parents=True)
    (sub / "videos_1.tar.gz").write_bytes(b"x" * 10)
    mc = [{"id": "1", "video": "yt/a.avi", "conversations": [{}, {}]},
          {"id": "2", "video": "yt/unused1.avi", "conversations": [{}, {}, {}, {}]},
          {"id": "3", "video": "yt/not_extracted.avi", "conversations": [{}, {}]}]
    (sub / "0_30_s_youtube_mc_v0_1_qa_processed.json").write_text(json.dumps(mc))
    nq = llava / "0_30_s_nextqa"
    nq.mkdir()
    (nq / "0_30_s_nextqa_oe_qa_processed.json").write_text(json.dumps([{"video": "NextQA/NExTVideo/1164/3238737531.mp4"}]))
    (root / "NExTQA").mkdir(parents=True)
    (root / "NExTQA" / "test.csv").write_text("video,question\n3238737531,q\n1111111111,q\n")
    write_video(root / "ucf-crime" / "Abuse001_x264.mp4")
    write_video(root / "XD-Violence" / "Abuse001.mp4")
    (root / "ShanghaiTech").mkdir()
    return data, llava, media, root


def test_audit_counts_unused_media_overlaps_and_converters(tmp_path):
    data, llava, media, root = build(tmp_path)
    report = run_audit(data, llava, media, root, tmp_path / "report", probe_limit=50, max_files=1000)
    m = report["manifests"]["files"]
    assert m["train"]["questions"] == 3 and m["train"]["unique_videos"] == 2 and m["val"]["missing_media"] == 1
    assert report["manifests"]["video_overlap"] == {"train&val": 0}
    mr = report["media_root"]
    assert mr["video_files"] == 5 and mr["referenced_by_manifests"] == 3 and mr["unused_video_files"] == 2
    ann = {a["name"]: a for a in report["llava"]["0_30_s_youtube_v0_1"]["annotations"]}
    entry = ann["0_30_s_youtube_mc_v0_1_qa_processed.json"]
    assert entry["kind"] == "mc" and entry["qa_turns"] == 4 and entry["unique_videos"] == 3
    assert entry["videos_extracted"] == 2 and entry["extracted_videos_not_in_manifests"] == 1
    assert report["llava"]["0_30_s_youtube_v0_1"]["archives"][0]["name"] == "videos_1.tar.gz"
    assert report["overlap"]["nextqa"]["overlap"] == {"0_30_s_nextqa & test.csv": 1}
    assert report["overlap"]["monitoring"]["name_overlap"]["XD-Violence & ucf-crime"] == 1
    assert "not_implemented" in report["converters"] and report["errors"] == {}
    text = digest(report)
    assert "UNUSED 2" in text and "nextqa" in text and (tmp_path / "report" / "audit.json").is_file()
