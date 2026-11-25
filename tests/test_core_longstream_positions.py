import hashlib
import subprocess

import numpy as np

from trace_vl.core_vlm.longstream import stream_pixels
from trace_vl.core_vlm.longstream_pairs import write_views
from trace_vl.core_vlm.longstream_positions import DELAYS, OFFSETS, shifted_queries


def test_event_age_and_position_are_crossed_without_changing_question_options_or_labels():
    a = {'source_group': 'source', 'labels': ['visible scene A', 'visible scene B']}
    ids = []
    for offset in OFFSETS:
        ab, ba = [shifted_queries(a, swapped, offset) for swapped in (False, True)]
        for delay, x, y in zip(DELAYS, ab, ba, strict=True):
            assert x['options'] == y['options'] and x['question'] == y['question']
            assert x['answer'] != y['answer'] and x['query_time'] == y['query_time']
            assert x['evidence'] == y['evidence'] == [offset, offset+3.5]
            assert x['query_time'] - x['evidence'][1] == delay+4
            ids += [x['id'], y['id']]
    assert len(set(ids)) == 18


def test_lossless_common_prefix_moves_history_without_changing_other_pixels(tmp_path):
    rng = np.random.default_rng(11)
    video = rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    path = tmp_path / 'common.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '32x32',
                    '-r', '2', '-i', 'pipe:0', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(path)],
                   input=video.tobytes(), capture_output=True, check=True)
    events = [np.full((4, 32, 32, 3), 30, dtype=np.uint8), np.full((4, 32, 32, 3), 210, dtype=np.uint8)]
    outputs = [tmp_path / 'ab.mkv', tmp_path / 'ba.mkv']
    hashes, suffix = write_views(outputs, events, path, np.arange(8, 16), (32, 32),
                                  shared_prefix=(path, np.arange(6)))
    decoded = [np.concatenate(list(stream_pixels(p, np.arange(22), 32, 32))) for p in outputs]
    for i, rgb in enumerate(decoded):
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == hashes[i]
        np.testing.assert_array_equal(rgb[:6], video[:6])
        np.testing.assert_array_equal(rgb[14:], video[8:16])
    np.testing.assert_array_equal(decoded[0][6:10], decoded[1][10:14])
    np.testing.assert_array_equal(decoded[0][10:14], decoded[1][6:10])
    assert suffix == hashlib.sha256(video[8:16].tobytes()).hexdigest()
