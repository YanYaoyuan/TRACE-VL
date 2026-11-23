import json
import shutil
import subprocess

import numpy as np
import pytest

from trace_vl.core_vlm.data import decode_observation, presentation_frames, read_split


@pytest.fixture
def video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    path = tmp_path / "pts.mkv"
    # Distinct pixel values let us verify decoded indices, not just timestamps.
    frames = np.stack([np.full((32, 32, 3), 20 * i, np.uint8) for i in range(8)])
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "32x32", "-r", "2", "-i", "pipe:0", "-c:v", "ffv1", str(path)],
                   input=frames.tobytes(), check=True)
    return path


def test_exact_frames_are_causal_and_reused_for_pooling_modes(video):
    row = {"video": str(video), "protocol": "online", "query_time": 1.6}
    images, times, resolved, audit = decode_observation(row, steps=2, pixels=1024)
    assert audit["frame_indices"] == [0, 1, 2, 3]
    np.testing.assert_allclose(times, [[0, .5], [1, 1.5]])
    np.testing.assert_allclose(images.mean(axis=(2, 3, 4)), [[0, 20], [40, 60]])
    assert resolved["query_time"] == 1.6


def test_offline_end_is_resolved_and_observation_start_respected(video):
    row = {"video": str(video), "protocol": "offline", "observation": {"start": 2., "end": "video_end"}}
    _, times, resolved, audit = decode_observation(row, steps=1, pixels=1024)
    assert resolved["query_time"] == 3.5
    assert audit["frame_indices"] == [4, 7]
    np.testing.assert_allclose(times, [[2., 3.5]])


def test_short_prefix_cannot_silently_read_future_frames(video):
    with pytest.raises(ValueError, match="distinct causal frames"):
        decode_observation({"video": str(video), "query_time": .5}, steps=2, pixels=1024)
    with pytest.raises(ValueError, match="online examples require"):
        decode_observation({"video": str(video)}, steps=1, pixels=1024)


def test_timestamp_filter_preserves_clean_observations(video):
    row = {"video": str(video), "protocol": "online", "query_time": 1.6}
    options = dict(steps=2, pixels=1024, recent_hz=2.)
    strict = decode_observation(row, **options)
    repaired = decode_observation(row, **options, drop_bad_pts=True)
    np.testing.assert_array_equal(strict[0], repaired[0])
    np.testing.assert_array_equal(strict[1], repaired[1])
    assert strict[3]["frame_indices"] == repaired[3]["frame_indices"]
    assert repaired[3]["timestamp_filter"]["dropped_frames"] == 0


def test_duplicate_timestamps_preserve_decoder_frame_numbers(tmp_path, video):
    duplicate = tmp_path / "duplicate-pts.mkv"
    frames = np.stack([np.full((32, 32, 3), 20 * i, np.uint8) for i in range(8)])
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32", "-r", "4",
         "-i", "pipe:0", "-vf", "setpts=floor(N/2)/(2*TB)", "-vsync", "0", "-c:v", "ffv1", str(duplicate)],
        input=frames.tobytes(), check=True,
    )
    row = {"video": str(duplicate), "query_time": 1.6}
    options = dict(steps=2, pixels=1024, recent_hz=2.)
    with pytest.raises(ValueError, match="strictly increasing"):
        decode_observation(row, **options)
    images, times, _, audit = decode_observation(row, **options, drop_bad_pts=True)
    np.testing.assert_allclose(times, [[0, .5], [1., 1.5]])
    np.testing.assert_allclose(images.mean(axis=(2, 3, 4)), [[0, 40], [80, 120]])
    assert audit["frame_indices"] == [0, 2, 4, 6]
    assert audit["timestamp_filter"]["dropped_frames"] == 4
    with pytest.raises(ValueError, match="distinct causal frames"):
        decode_observation({**row, "query_time": .1}, **options, drop_bad_pts=True)


def test_timestamp_filter_drops_backwards_and_missing_values_in_decode_order():
    frames = [{"best_effort_timestamp_time": x} for x in ("1.0", "2.0", "1.5", None, "2.0", "3.0")]
    _, mask = presentation_frames(frames, drop_bad_pts=True)
    assert np.flatnonzero(mask).tolist() == [0, 1, 5]
    with pytest.raises(ValueError, match="finite first"):
        presentation_frames(frames[3:], drop_bad_pts=True)


def test_manifest_preserves_split_and_rejects_cross_split_sources(tmp_path):
    path = tmp_path / "qa.jsonl"
    row = {"id": "a", "video": "a.mp4", "question": "Action?", "answer": "running",
           "answer_type": "text", "split": "train", "source_group": "original-a"}
    path.write_text(json.dumps(row) + "\n")
    result = read_split(path, "train")[0]
    assert result["answer_text"] == "running"
    assert result["video"] == str(tmp_path / "a.mp4")
    other = {**row, "id": "b", "video": "alternate-transcode.mp4", "split": "val"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(other))
    with pytest.raises(ValueError, match="crosses splits"):
        read_split(path, "train")
