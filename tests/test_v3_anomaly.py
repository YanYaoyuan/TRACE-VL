"""CPU tests for the alert-head anomaly data and metrics (no weights, no real datasets)."""

import itertools
import json

import numpy as np
import pytest

from trace_vl.v3.anomaly import (
    CANONICAL,
    average_precision,
    convert_anomaly,
    frame_spans,
    roc_auc,
    score_alerts,
    step_label,  # fmt: skip
)
from trace_vl.v3.config import V3Config, load_v3_config
from trace_vl.v3.data import StepReader, read_jsonl


def test_frame_spans_and_step_labels():
    labels = [0, 1, 1, 0, 0, 1, 0, 1]
    spans = frame_spans(labels, 2.0)
    assert spans == [[0.5, 1.0], [2.5, 2.5], [3.5, 3.5]]
    assert frame_spans([0, 0], 25.0) == [] and frame_spans([1, 1], 1.0) == [[0.0, 1.0]]
    assert [step_label(spans, t) for t in (0.25, 0.75, 1.0, 2.0, 3.5)] == [0, 1, 1, 0, 1]


def brute_auc(y, s):
    pos = [b for a, b in zip(y, s) if a]
    neg = [b for a, b in zip(y, s) if not a]
    wins = sum((p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def test_roc_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 40)
        s = rng.integers(0, 5, 40).astype(float)  # many ties
        if 0 < y.sum() < len(y):
            assert roc_auc(y, s) == pytest.approx(brute_auc(y, s))
    assert roc_auc([1, 1], [0.1, 0.2]) is None and roc_auc([0, 1], [0.2, 0.9]) == 1.0


def test_average_precision():
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([0, 1], [0.5, 0.5]) == pytest.approx(0.5)  # a tie is one threshold
    assert average_precision([0, 0], [0.1, 0.2]) is None


def write_frames(folder, n, size=(40, 32)):
    import cv2

    folder.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"{i:03d}.jpg"), np.full((size[1], size[0], 3), i * 10 % 255, dtype=np.uint8))


def write_video(path, n, fps=24, size=(40, 32)):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    writer.release()


def make_shanghaitech(root):
    masks = root / "testing" / "test_frame_mask"
    masks.mkdir(parents=True)
    for name, n, abnormal in (("01_0014", 48, (10, 30)), ("02_0128", 36, None), ("03_0031", 24, (0, 5))):
        write_frames(root / "testing" / "frames" / name, n)
        labels = np.zeros(n, dtype=int)
        if abnormal:
            labels[abnormal[0] : abnormal[1]] = 1
        np.save(masks / f"{name}.npy", labels)
    write_frames(root / "testing" / "frames" / "04_0001", 10)  # no mask -> reported missing
    for i in range(3):
        write_video(root / "training" / "videos" / f"01_00{i}.avi", 30)


def test_convert_shanghaitech_split_list_and_negatives(tmp_path):
    make_shanghaitech(tmp_path)
    (tmp_path / "train.txt").write_text("01_0014.npy\n02_0128\n")
    out = tmp_path / "records.jsonl"
    report = convert_anomaly("shanghaitech", str(tmp_path), str(out), train_list=str(tmp_path / "train.txt"),
                             normal_train=2)  # fmt: skip
    rows = {r["id"]: r for r in read_jsonl(out)}
    assert report["missing_count"] == 1 and report["train"]["videos"] == 4 and report["test"]["videos"] == 1
    test = rows["shanghaitech-03_0031"]
    assert test["split"] == "test" and test["alerts"][0]["text"] == CANONICAL
    assert test["alerts"][0]["spans"] == [[0.0, round(4 / 24, 3)]] and test["fps"] == 24.0
    assert rows["shanghaitech-01_0014"]["alerts"][0]["spans"] == [[round(10 / 24, 3), round(29 / 24, 3)]]
    negatives = [r for r in rows.values() if r["id"].startswith("shanghaitech-train")]
    assert len(negatives) == 2 and all(r["alerts"][0]["spans"] == [] and r["split"] == "train" for r in negatives)
    assert all("question" not in r and r["alert_only"] for r in rows.values())
    # Frame folders stream through StepReader with the record's fps.
    reader = StepReader(test, str(tmp_path))
    try:
        assert reader.duration == pytest.approx(23 / 24)
        frames = reader.step(0.5, 0.5)
        assert frames.shape == (2, 32, 40, 3)
    finally:
        reader.close()
    with pytest.raises(ValueError, match="dataset"):
        convert_anomaly("ucf", str(tmp_path), str(out))


def test_convert_hash_split_is_video_disjoint_and_stable(tmp_path):
    make_shanghaitech(tmp_path)
    a = convert_anomaly("shanghaitech", str(tmp_path), str(tmp_path / "a.jsonl"), normal_train=0, seed=3)
    b = convert_anomaly("shanghaitech", str(tmp_path), str(tmp_path / "b.jsonl"), normal_train=0, seed=3)
    assert a == b and (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()
    ids = [r["id"] for r in read_jsonl(tmp_path / "a.jsonl")]
    assert len(ids) == len(set(ids)) == 3


def test_avenue_npy_labels(tmp_path):
    write_video(tmp_path / "testing_videos" / "01.avi", 50, fps=25)
    write_video(tmp_path / "training_videos" / "01.avi", 20, fps=25)
    (tmp_path / "labels").mkdir()
    np.save(tmp_path / "labels" / "01.npy", np.r_[np.zeros(25), np.ones(25)])
    convert_anomaly("avenue", str(tmp_path), str(tmp_path / "av.jsonl"), labels_dir=str(tmp_path / "labels"),
                    train_fraction=0.0)  # fmt: skip
    rows = {r["id"]: r for r in read_jsonl(tmp_path / "av.jsonl")}
    assert rows["avenue-test01"]["split"] == "test" and rows["avenue-test01"]["alerts"][0]["spans"] == [[1.0, 1.96]]
    assert rows["avenue-train01"]["split"] == "train"


def test_score_alerts_and_mixed_runs(tmp_path):
    rows = []
    for v in range(4):
        for k in range(10):
            label = int(k >= 6 and v % 2 == 0)
            rows.append({"id": f"v{v}", "t": k + 0.25, "label": label, "p": 0.2 + 0.6 * label + 0.01 * k, "ms": 2.0,
                         "run": "r1"})  # fmt: skip
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = score_alerts(str(path), replicates=50)
    assert result["auc"] == 1.0 and result["videos"] == 4 and result["ms_per_decision_mean"] == 2.0
    assert result["auc_ci95"][0] <= 1.0
    rows[0]["run"] = "r2"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="mixes runs"):
        score_alerts(str(path))


def test_alert_configs():
    cfg, _ = load_v3_config("configs/v3/stage1_alert.yaml")
    assert cfg.train.alert_fraction == 0.25
    cfg, _ = load_v3_config("configs/v3/stage1_a100_alert.yaml")
    assert cfg.train.grad_accum == 4 and cfg.train.alert_fraction == 0.25
    with pytest.raises(ValueError, match="alert_fraction"):
        V3Config.from_dict({"train": {"alert_fraction": 1.5}})


def test_prefix_resolves_against_a_shared_root(tmp_path):
    make_shanghaitech(tmp_path / "sht")
    convert_anomaly("shanghaitech", str(tmp_path / "sht"), str(tmp_path / "r.jsonl"), normal_train=1, prefix="sht/")
    for record in read_jsonl(tmp_path / "r.jsonl"):
        assert record["video"].startswith("sht/") and (tmp_path / record["video"]).exists()
