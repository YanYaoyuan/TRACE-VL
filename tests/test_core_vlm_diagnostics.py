import copy
import json

import pytest
import torch

from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.interventions import block_order, disable_temporal, shuffle_blocks
from trace_vl.core_vlm.mixed_diagnostics import (
    VARIANTS,
    audit_variants,
    check_replay,
    hld_analysis,
    interaction,
    load_base,
    make_report,
)
from trace_vl.core_vlm.prepare import write_json
from trace_vl.core_vlm.qa import VisualStep


def fixture():
    rows = [dict(id=str(i), video=f'video{i//2}', source_group=f'video{i//2}', task='HLD',
                 split='val', question='which?', options=['a', 'b'], answer=i % 2) for i in range(4)]
    maps = {}
    for name in ['original_grid', 'plain', 'temporal', *VARIANTS]:
        maps[name] = {}
        for row in rows:
            obs = dict(video_sha256=row['video'], frame_indices=[0, 1, 2, 3], decoded_rgb_sha256='rgb',
                       input_ids_sha256='prompt', timestamp_policy='native-index-fps',
                       timestamp_text_seconds=[1., 2.], frame_times=[[0., 1.], [1.5, 2.]])
            shuffle = name.endswith('_shuffle')
            maps[name][row['id']] = dict(id=row['id'], run=name, choice=row['answer'],
                answer=chr(65+row['answer']), token_ids=[row['answer']], observation=obs, visual_tokens=40,
                intervention={'temporal_off': name == 'temporal_off',
                              'block_shuffle_seed': 17 if shuffle else None,
                              'block_order': block_order(2, row['video']) if shuffle else [0, 1]})
    return rows, maps


def test_shuffle_preserves_content_and_time_slots_without_mutation():
    steps = [VisualStep(torch.full((4, 8), float(i)), (2, 2), float(i+1), (i+.5, i+1.))
             for i in range(8)]
    changed, order = shuffle_blocks(steps, 'video-sha')
    assert sorted(order) == list(range(8)) and order != list(range(8))
    assert shuffle_blocks(steps, 'video-sha')[1] == order
    assert [s.frame_times for s in changed] == [s.frame_times for s in steps]
    assert [s.end for s in changed] == [s.end for s in steps]
    for slot, source in enumerate(order):
        assert changed[slot].tokens is steps[source].tokens
        assert steps[slot].tokens[0, 0] == slot
    with pytest.raises(ValueError, match='two temporal blocks'):
        shuffle_blocks(steps[:1], 'video-sha')
    steps[0].grid = (1, 4)
    with pytest.raises(ValueError, match='spatial grids'):
        shuffle_blocks(steps, 'video-sha')


def test_temporal_off_changes_native_visual_path_keeps_language_weights():
    from test_core_vlm_native import tiny_example

    from trace_vl.core_vlm.native import NativeVideoQALoss, encode_native
    from trace_vl.core_vlm.temporal import attach_temporal
    # Uses the same tiny native backbone as the existing native forward tests.
    bb, layout, row = tiny_example()
    net = NativeVideoQALoss(bb, mode='grid', pool_grid=(1, 1))
    steps = encode_native(bb, layout)
    row = {**row, 'native_layout': layout}
    original = net.prefix(steps, row)
    before = {k: v.detach().clone() for k, v in bb.model.state_dict().items()}
    attach_temporal(net, 'causal')
    with torch.no_grad():
        net.temporal.up.weight.fill_(.2)
    active = net.prefix(steps, row)
    disable_temporal(net)
    off = net.prefix(steps, row)
    assert torch.equal(original[0], off[0])
    assert not torch.equal(active[0], off[0])
    assert all(torch.equal(v, bb.model.state_dict()[k]) for k, v in before.items())
    with pytest.raises(ValueError, match='trained temporal'):
        disable_temporal(net)


def test_audit_rejects_changed_frames_budget_or_intervention():
    rows, maps = fixture()
    audit_variants(rows, maps)
    for field, value in [('visual_tokens', 39), ('intervention', {}),
                         ('observation', {**maps['temporal_off']['0']['observation'], 'frame_indices': [9]})]:
        bad = copy.deepcopy(maps)
        bad['temporal_off']['0'][field] = value
        with pytest.raises(ValueError):
            audit_variants(rows, bad)
    rows[0]['query_time'] = 1.
    with pytest.raises(ValueError, match='future frames'):
        audit_variants(rows, maps)


def test_replay_requires_same_choice_text_tokens_and_observations():
    rows, maps = fixture()
    check_replay(rows, maps['plain'], maps['plain'])
    for field, value in [('choice', 1), ('token_ids', [99]), ('visual_tokens', 20)]:
        bad = copy.deepcopy(maps['plain'])
        bad['0'][field] = value
        with pytest.raises(ValueError, match='replay differs'):
            check_replay(rows, maps['plain'], bad)


def test_direct_interaction_pairs_correlated_arms_and_clusters():
    rows, maps = fixture()
    # Both arms lose exactly the same answers: interaction is identically zero,
    # although each ordered-minus-shuffled contrast is positive.
    for name in ('temporal_shuffle', 'plain_shuffle'):
        maps[name]['0']['choice'] = 1
        maps[name]['2']['choice'] = 1
    result = interaction(rows, maps)
    assert result['delta_pp'] == 0 and result['ci95_pp'] == [0., 0.]
    assert result['source_videos'] == 2
    maps['temporal_shuffle']['1']['choice'] = 0
    assert interaction(rows, maps)['delta_pp'] == 25
    assert interaction([], maps) is None
    assert interaction(rows[:1], maps)['ci95_pp'] is None


def test_report_keeps_all_hld_cases_and_no_training_claim():
    rows, maps = fixture()
    report = make_report({'ovo': rows}, {'ovo': maps})
    assert report['completed'] and report['training_launched'] is False
    assert report['results']['ovo']['order_interaction']['primary']['delta_pp'] == 0
    hld = hld_analysis(rows, maps)
    assert hld['n'] == 4 and len(hld['cases']) == 4
    assert hld['gold_option_counts'] == {'0': 2, '1': 2}


def test_load_base_fails_on_population_protocol_or_prediction_corruption(tmp_path):
    rows, maps = fixture()
    base, manifests = tmp_path / 'base', tmp_path / 'data'
    manifests.mkdir()
    cfg = dict(model='model', steps=2, pixels=65536, pool_grid=[4, 5],
               timestamp_policy='native-index-fps', max_new_tokens=64)
    write_json(base / 'report.json', {'completed': True})
    write_json(base / 'training-audit.json', {'passed': True})
    for ds in ('ovo', 'nextqa'):
        m = manifests / f'{ds}.jsonl'
        m.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for arm in ('original_grid', 'plain', 'temporal'):
            folder = base / 'eval' / ds / arm
            write_json(folder / 'run.json', {'id': arm, 'manifest_sha256': sha256(m),
                       'config': {**cfg, 'mode': 'grid', 'limit': 0, 'split': 'val', 'mc_choice_score': True}})
            (folder / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in maps[arm].values()))
    populations, _, hashes = load_base(base, manifests, cfg)
    assert len(populations['ovo']) == 4 and len(hashes) == 14
    with pytest.raises(ValueError, match='protocol changed'):
        load_base(base, manifests, {**cfg, 'steps': 8})
    manifest = manifests / 'ovo.jsonl'
    content = manifest.read_text()
    manifest.write_text(content+'\n')
    with pytest.raises(ValueError, match='population changed'):
        load_base(base, manifests, cfg)
    manifest.write_text(content)
    predictions = base / 'eval/ovo/plain/predictions.jsonl'
    predictions.write_text(predictions.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError, match='coverage'):
        load_base(base, manifests, cfg)
