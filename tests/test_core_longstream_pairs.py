import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from trace_vl.core_vlm.longstream import schedule, stream_pixels
from trace_vl.core_vlm.longstream_pairs import pair_queries, select_events, write_views
from trace_vl.streaming_vision.media import VideoProbeCache


def test_frozen_all_source_annotations_have_nonoverlapping_exact_reviewed_events():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'docs/core_vlm/longstream_pairs_v1.json').read_text())
    assert len(plan['rows']) == len({r['source_group'] for r in plan['rows']}) == 8
    for a in plan['rows']:
        ab, ba = [pair_queries(a, swap, plan['question'], plan['delays']) for swap in (False, True)]
        for p, q in zip(ab, ba, strict=True):
            assert p['question'] == q['question'] and p['options'] == q['options']
            assert p['answer'] != q['answer'] and p['query_time'] == q['query_time']
            assert p['options'][p['answer']] == a['labels'][0]
            assert q['options'][q['answer']] == a['labels'][1]


def test_reviewed_scene_selection_rejects_boundary_frame_reuse_and_bad_steps():
    review = {'source_group': 's', 'groups': [
        {'indices': list(range(8)), 'times': [x/2 for x in range(8)]},
        {'indices': list(range(8, 16)), 'times': [4+x/2 for x in range(8)]}]}
    a = {'source_group': 's', 'labels': ['left', 'right'], 'event_groups': [0, 1], 'event_steps': [[0, 1], [1, 2]]}
    selected = select_events(review, a)
    assert selected[1]['indices'] == [[10, 11], [12, 13]]
    bad = copy.deepcopy(a)
    bad['event_groups'] = [0, 0]
    with pytest.raises(ValueError, match='distinct source frames'):
        select_events(review, bad)
    bad['event_groups'] = [0, 1]
    bad['event_steps'][0] = [-1, 0]
    with pytest.raises(ValueError, match='outside'):
        select_events(review, bad)
    bad['event_steps'][0] = [1, 0]
    with pytest.raises(ValueError, match='internal order'):
        select_events(review, bad)


def test_lossless_materialized_pair_changes_only_history_and_preserves_decoder_clock(tmp_path):
    # Real FFmpeg encode/probe/decode, not a mock of the generator: include colors,
    # high-frequency noise and noncontiguous source indices to catch RGB/PTS drift.
    rng = np.random.default_rng(4)
    filler_frames = rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    filler = tmp_path / 'filler.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '32x32',
                    '-r', '2', '-i', 'pipe:0', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(filler)],
                   input=filler_frames.tobytes(), capture_output=True, check=True)
    events = [rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8) for _ in range(2)]
    outputs = [tmp_path / 'ab.mkv', tmp_path / 'ba.mkv']
    chosen = [0, 1, 4, 5, 8, 9, 14, 15]
    hashes, suffix = write_views(outputs, events, filler, chosen, (32, 32))
    assert suffix == hashlib.sha256(filler_frames[chosen].tobytes()).hexdigest()
    decoded = []
    for i, path in enumerate(outputs):
        _, meta, _ = VideoProbeCache(tmp_path / 'cache', stage_media=False).source(str(path), stage=False)
        indices, times, _ = schedule(meta, 7.5)
        np.testing.assert_array_equal(indices, np.arange(16))
        np.testing.assert_array_equal(times.ravel(), np.arange(16) / 2)
        rgb = np.concatenate(list(stream_pixels(path, indices, 32, 32)))
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == hashes[i]
        decoded.append(rgb)
    np.testing.assert_array_equal(decoded[0][:4], decoded[1][4:8])
    np.testing.assert_array_equal(decoded[0][4:8], decoded[1][:4])
    np.testing.assert_array_equal(decoded[0][8:], decoded[1][8:])
    assert not np.array_equal(decoded[0][:8], decoded[1][:8])
