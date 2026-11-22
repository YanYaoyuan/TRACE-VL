"""Exercise exact full refresh, causal sparse writes and correction gradients."""
from __future__ import annotations

import csv
import zipfile

import numpy as np
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from trace_vl.core_vlm.data import recent_indices
from trace_vl.streaming_vision.data import streaming_population
from trace_vl.streaming_vision.encoder import ContextCorrection, SpatialRefresh


def tiny_visual():
    torch.manual_seed(7)
    config = Qwen3_5VisionConfig(depth=3, hidden_size=32, num_heads=4, intermediate_size=48,
                                 out_hidden_size=48, num_position_embeddings=16)
    return Qwen3_5VisionModel(config).eval()


def test_full_refresh_matches_native_and_geometry_changes_reset_state():
    visual = tiny_visual()
    state = SpatialRefresh(visual, 1, 1., 99, 100.)
    for h, w in ((4, 4), (4, 4), (2, 4)):
        patches = torch.randn(h * w, 3 * 2 * 16**2)
        expected = visual(patches, grid_thw=torch.tensor([[1, h, w]])).pooler_output
        result = state.update(patches, h, w)
        torch.testing.assert_close(result.features, expected, atol=0, rtol=0)
        assert result.refreshed.all()
    assert state.step == 1 and result.reason == "initial"


def test_sparse_refresh_reads_bounded_cache_and_does_not_mutate_prior_output():
    visual = tiny_visual()
    state = SpatialRefresh(visual, 1, .25, 4, 100.)
    patches = torch.randn(16, 3 * 2 * 16**2)
    initial = state.update(patches, 4, 4)
    saved = initial.features.clone()
    cache = [(k.data_ptr(), v.data_ptr()) for k, v in state.kv]
    second = state.update(patches + .3 * torch.randn_like(patches), 4, 4)
    assert second.reason == "selective" and second.refreshed.sum() == 1
    assert cache == [(k.data_ptr(), v.data_ptr()) for k, v in state.kv]
    assert torch.equal(initial.features, saved)
    assert torch.equal(second.features[~second.refreshed], saved[~second.refreshed])
    assert all(k.size(0) == v.size(0) == 16 for k, v in state.kv)
    for _ in range(3):
        later = state.update(patches, 4, 4)
    assert later.reason == "periodic"


def test_scene_cut_forces_native_refresh():
    visual = tiny_visual()
    state = SpatialRefresh(visual, 1, .25, 99, 1e-8)
    patches = torch.randn(16, 3 * 2 * 16**2)
    state.update(patches, 4, 4)
    result = state.update(-patches, 4, 4)
    assert result.reason == "scene-cut" and result.refreshed.all()


def test_correction_gradients_and_zero_for_refreshed_context():
    corrector = ContextCorrection(32, 48, 8)
    scene, anchors = torch.randn(32), torch.randn(4, 32)
    assert torch.count_nonzero(corrector(scene, anchors, anchors)) == 0
    target = torch.randn(4, 48)
    optim = torch.optim.AdamW(corrector.parameters(), lr=.03)
    for _ in range(3):
        optim.zero_grad()
        (corrector(scene, anchors, anchors) - target).square().mean().backward()
        optim.step()
    assert corrector.context.weight.grad.norm() > 0
    assert corrector.location.weight.grad.norm() > 0
    assert torch.count_nonzero(corrector(scene, anchors, anchors)) > 0
    assert torch.count_nonzero(corrector(scene, scene.expand_as(anchors), anchors)) == 0


def test_fixed_cadence_indices_are_causal_and_distinct():
    pts = np.arange(20) / 4
    assert recent_indices(pts, np.arange(17), 2, 2, 4).tolist() == [8, 10, 12, 14]
    assert recent_indices(pts, np.arange(16), 2, 2, 3.75).tolist() == [8, 10, 12, 14]
    assert recent_indices(pts, np.arange(2), 5, 2, .25).tolist() == [0, 1]


def test_streamingbench_reads_official_csv_and_extracts_only_selected_zip_member(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    keys = ["question_id", "time_stamp", "task_type", "question", "options", "answer"]
    with (root / "Real_Time_Visual_Understanding.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for i in (1, 2):
            writer.writerow((f"Real-Time Visual Understanding_sample_{i}_1", "00:00:08", "Counting",
                             "How many?", "['A. One', 'B. Two']", "B"))
    with zipfile.ZipFile(root / "Real-Time Visual Understanding_1-50.zip", "w") as archive:
        archive.writestr("Real-Time Visual Understanding/sample_1/v_abcdefghijk.mp4", b"a")
        archive.writestr("Real-Time Visual Understanding/sample_2/v_ABCDEFGHIJK.mp4", b"b")
        archive.writestr("__MACOSX/Real-Time Visual Understanding/sample_1/._v_abcdefghijk.mp4", b"c")
    output = tmp_path / "selection"
    rows, _ = streaming_population(root, output, 1)
    assert len(rows) == 1 and rows[0]["options"] == ["One", "Two"] and rows[0]["answer"] == 1
    assert rows[0]["query_time"] == 8
    assert {p.name for p in output.iterdir()} == {"sample_1.mp4"}


def test_probe_cache_preserves_frames_and_shares_index_across_workers(tmp_path, monkeypatch):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    from trace_vl.core_vlm.data import decode_observation, probe_video
    from trace_vl.streaming_vision import media

    video = tmp_path / "motion.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=8:duration=4",
         "-c:v", "libx264", "-threads", "1", str(video)],
        check=True,
    )
    row = {"video": str(video), "query_time": 2.75}
    observation = {"steps": 3, "pixels": 6144, "recent_hz": 2}
    reference = decode_observation(row, **observation)
    probed = []

    def delayed(path, timeout):
        probed.append(path)
        if path == video:
            raise subprocess.TimeoutExpired("ffprobe", timeout)
        return probe_video(path, timeout)

    monkeypatch.setattr(media, "probe_video", delayed)
    scratch = tmp_path / "cache"
    caches = [media.VideoProbeCache(scratch, fast_timeout=1) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(lambda cache: cache.decode(row, **observation), caches))
    assert len(probed) == 2
    assert probed[0] == video and probed[1].parent == scratch
    assert len(list(scratch.glob("*.mp4"))) == 1
    for images, times, resolved, audit in results:
        np.testing.assert_array_equal(images, reference[0])
        np.testing.assert_array_equal(times, reference[1])
        assert (resolved, audit) == reference[2:]
