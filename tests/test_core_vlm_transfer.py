import copy
import csv
import json
from types import SimpleNamespace

import pytest
import torch

from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.native import NativeVideoQALoss, encode_native
from trace_vl.core_vlm.temporal import CausalTemporalAdapter, attach_temporal, load_temporal
from trace_vl.core_vlm.transfer import audit_observations, continuation_gate, stats
from trace_vl.core_vlm.transfer_data import check_overlap, nextqa_rows, ovo_rows
from trace_vl.v3.ovo_transfer import video_half


def test_temporal_identity_initialization_and_future_isolation():
    model = CausalTemporalAdapter(32, 16).eval()
    x = torch.randn(5, 3, 32)
    torch.testing.assert_close(model(x, range(5)), x, atol=0, rtol=0)
    torch.nn.init.normal_(model.up.weight, std=.1)
    original = model(x, range(5))
    changed = x.clone()
    changed[3:] += 100 * torch.randn_like(changed[3:])
    torch.testing.assert_close(model(changed, range(5))[:3], original[:3], atol=1e-6, rtol=1e-6)
    # Prefix equivalence after nonzero initialization tests the actual attention mask.
    torch.testing.assert_close(model(x[:3], range(3)), original[:3], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='increase'):
        model(x, [0, 1, 1, 3, 4])


def test_real_qa_loss_trains_temporal_inner_layers_and_reload(tmp_path):
    from test_core_vlm_native import tiny_example
    bb, layout, row = tiny_example()
    bb.model.requires_grad_(False)
    net = NativeVideoQALoss(bb, mode='grid', pool_grid=(1, 1), answer_end_token_id=2)
    steps = encode_native(bb, layout)
    attach_temporal(net, 'causal')
    optimizer = torch.optim.AdamW(net.temporal.parameters(), lr=.001, weight_decay=0)
    net.train()
    for i in range(2):
        optimizer.zero_grad()
        loss, _ = net(steps, row)
        loss.backward()
        assert net.temporal.up.weight.grad.abs().sum() > 0
        if i:
            assert net.temporal.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
        optimizer.step()
    net.eval()
    expected = net.prefix(steps, row)[0].detach().clone()
    torch.save({'parameters': {n: p.detach() for n, p in net.named_parameters() if n.startswith('temporal.')}},
               tmp_path / 'training.pt')
    attach_temporal(net, 'causal')
    load_temporal(net, tmp_path)
    net.eval()
    torch.testing.assert_close(net.prefix(steps, row)[0], expected, atol=0, rtol=0)


def test_first_token_choice_is_gold_blind_and_survives_eos():
    from test_core_vlm_qa import TinyBackbone, sample

    from trace_vl.core_vlm.qa import VideoQALoss
    bb = TinyBackbone()
    net = VideoQALoss(bb, answer_end_token_id=2)
    steps, row = sample()
    row.update(options=['left', 'right'], answer=0)
    def logits(hidden):
        out = torch.zeros(1, 1, 64)
        out[..., 2] = 100  # Free generation produces nothing.
        out[..., bb.ids('B')[0]] = 10
        return out
    bb.logits = logits
    assert net.generate(steps, row, return_choice=True) == ([], 1)
    row['answer'] = 1
    assert net.generate(steps, row, return_choice=True) == ([], 1)


def test_formats_are_reported_separately_from_choice_accuracy():
    rows = [{'id': 'q', 'question': 'x', 'options': ['a', 'b'], 'answer': 1}]
    result = stats(rows, {'q': {'answer': 'The answer is B', 'choice': 1}})
    assert result['choice_accuracy'] == 1
    assert result['strict_accuracy'] == 0 and result['format_valid_fraction'] == 0


def test_transfer_gate_requires_both_domains_gain_and_no_harm():
    rule = {'gain_pp': 2, 'noninferiority_pp': 2}
    good = {'primary': {'choice': {'delta_pp': 3, 'ci95_pp': [1, 5]}},
            'no_harm': {'choice': {'delta_pp': 0, 'ci95_pp': [-1, 1]}}}
    values = {d: copy.deepcopy(good) for d in ('ovo', 'nextqa')}
    assert continuation_gate(values, rule)['passed']
    for dataset, key, change in [('ovo', 'primary', {'delta_pp': 1.9}),
                                  ('nextqa', 'primary', {'ci95_pp': [-.1, 9]}),
                                  ('ovo', 'no_harm', {'ci95_pp': [-2, 2]}),
                                  ('nextqa', 'primary', {'ci95_pp': None})]:
        changed = copy.deepcopy(values)
        changed[dataset][key]['choice'].update(change)
        assert not continuation_gate(changed, rule)['passed']


def test_observation_audit_catches_future_and_frame_mismatch():
    observation = dict(video_sha256='x', frame_indices=[0, 1], decoded_rgb_sha256='rgb',
                       input_ids_sha256='prompt', timestamp_policy='native', timestamp_text_seconds=[1],
                       frame_times=[[0, 1]])
    maps = {arm: {'q': {'observation': copy.deepcopy(observation)}} for arm in ('a', 'b')}
    rows = [{'id': 'q', 'query_time': 1}]
    audit_observations(rows, maps)
    maps['b']['q']['observation']['decoded_rgb_sha256'] = 'other'
    with pytest.raises(ValueError, match='unmatched'):
        audit_observations(rows, maps)
    maps['b']['q']['observation'] = copy.deepcopy(observation)
    with pytest.raises(ValueError, match='future'):
        audit_observations([{'id': 'q', 'query_time': .9}], maps)


def test_nextqa_selection_is_by_video_before_outcomes_and_keeps_all_questions(tmp_path):
    path = tmp_path / 'val.csv'
    fields = ['video', 'qid', 'question', 'answer', 'type', 'a0', 'a1', 'a2', 'a3', 'a4']
    with path.open('w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for video in range(10):
            for q in range(2):
                writer.writerow(dict(video=video, qid=q, question='when?', answer=q, type='TN',
                                     **{f'a{i}': str(i) for i in range(5)}))
    rows = nextqa_rows(path, videos=4)
    assert len(rows) == 8 and len({r['video'] for r in rows}) == 4
    assert nextqa_rows(path, videos=4) == rows
    assert all(r['split'] == 'val' and r['protocol'] == 'offline' for r in rows)


def test_ovo_preserves_manifest_hash_and_rejects_final_half(tmp_path):
    def write(video):
        row = dict(id='q', video=video, time=5, task='ASI', question='order?', options=['a', 'b'], answer=0)
        path = tmp_path / 'questions.jsonl'
        path.write_text(json.dumps(row) + '\n')
        (tmp_path / 'inventory.json').write_text(json.dumps(dict(
            questions_sha256=sha256(path), hash_split={'seed': 7, 'fraction': .5},
            sources={'video_root': '/media'})))
    diag = next(str(i) + '.mp4' for i in range(100) if video_half(str(i) + '.mp4', 7, .5) == 'diagnostic')
    final = next(str(i) + '.mp4' for i in range(100) if video_half(str(i) + '.mp4', 7, .5) != 'diagnostic')
    write(diag)
    assert ovo_rows(tmp_path)[0]['query_time'] == 5
    write(final)
    with pytest.raises(ValueError, match='outside diagnostic'):
        ovo_rows(tmp_path)


@pytest.mark.parametrize('alias,content', [(True, False), (False, True)])
def test_training_overlap_fails_for_alias_or_renamed_exact_bytes(alias, content):
    train = [{'video': '123.mp4', 'source_aliases': ['123']}]
    rows = [{'id': 'q', 'video': 'renamed.mp4', 'source_aliases': ['123' if alias else 'other']}]
    with pytest.raises(ValueError, match='overlaps'):
        check_overlap(rows, train, {'renamed.mp4': 'same' if content else 'different'}, {'same'})


def test_continuation_refuses_incompatible_base_or_protocol(monkeypatch, tmp_path):
    from trace_vl.core_vlm import train
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    args = SimpleNamespace(init_checkpoint=str(checkpoint), model='m', mode='grid', pool_grid=[4, 5],
                           rank=8, steps=8, pixels=65536, timestamp_policy='native-index-fps')
    config = {k: v for k, v in vars(args).items() if k not in ('model', 'init_checkpoint')}
    (tmp_path / 'run.json').write_text(json.dumps({'id': 'old', 'model': {'sha': 'x'}, 'config': config}))
    monkeypatch.setattr(train, 'checked_checkpoint', lambda *a: {'update': 1000, 'run': 'old'})
    monkeypatch.setattr(train, 'checkpoint_identity', lambda *a: {'sha': 'x'})
    assert train.initial_checkpoint(args)['update'] == 1000
    args.steps = 16
    with pytest.raises(ValueError, match='steps'):
        train.initial_checkpoint(args)


def test_failed_external_gate_stops_queue_before_continuation(monkeypatch, tmp_path):
    from trace_vl.core_vlm import transfer
    base = tmp_path / 'baseline'
    base.mkdir()
    cfg = dict(model='model', steps=8, pixels=65536, pool_grid=[4, 5], timestamp_policy='native-index-fps',
               rank=8, updates=300, baseline_run=str(base), data_out=str(tmp_path / 'data'),
               gpus=[0, 1], minimum_free_gib=30, max_new_tokens=64,
               gate={'gain_pp': 2, 'noninferiority_pp': 2})
    (base / 'experiment-config.json').write_text(json.dumps({**cfg, 'updates': 1000}))
    (base / 'summary.json').write_text(json.dumps({'completed': True}))
    for mode in ('full', 'grid'):
        path = base / 'train' / mode / 'checkpoint-001000'
        path.mkdir(parents=True)
        (path / 'complete.json').write_text('{}')
        (path.parent / 'run.json').write_text(json.dumps({'id': mode}))
    monkeypatch.setattr(transfer, 'checked_checkpoint', lambda *args: {'update': 1000})
    manifest = tmp_path / 'external.jsonl'
    manifest.write_text(json.dumps(dict(id='q', video='v.mp4', question='x', options=['a', 'b'],
                                        answer=0, split='val', protocol='offline'))+'\n')
    monkeypatch.setattr(transfer, 'prepare', lambda *args: {'ovo': manifest, 'nextqa': manifest})
    jobs = []
    monkeypatch.setattr(transfer, 'run_jobs', lambda tasks, *args: jobs.extend(tasks))
    negative = {'primary': {'choice': {'delta_pp': -1, 'ci95_pp': [-3, 1]}},
                'no_harm': {'choice': {'delta_pp': 0, 'ci95_pp': [-1, 1]}}}
    monkeypatch.setattr(transfer, 'external_report', lambda *args: {
        d: {'paired': {'trained_grid_minus_original_grid': negative}} for d in ('ovo', 'nextqa')})
    output = tmp_path / 'run'
    transfer.run(cfg, output)
    assert json.loads((output / 'queue-status.json').read_text())['training_started'] is False
    assert not any('trace_vl.core_vlm.train' in cmd for _, cmd, _ in jobs)


def test_external_prepare_real_media_and_frozen_resume(monkeypatch, tmp_path):
    import subprocess

    from trace_vl.core_vlm import transfer_data
    # Tiny synthetic clips do not need the production 30 GiB reserve on /tmp.
    # Keep the real media/manifest checks; test the disk gate independently below.
    space_checks = []
    monkeypatch.setattr(transfer_data, 'check_space',
                        lambda out, reserve: space_checks.append((out, reserve)))
    media = tmp_path / 'media'
    media.mkdir()
    for i in range(10):
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        f'color=c=0x{i * 24:02x}4080:s=192x160:r=16:d=2', '-threads', '1',
                        '-c:v', 'libx264', str(media / f'{i}.mp4')], check=True)
    annotation = tmp_path / 'val.csv'
    with annotation.open('w') as handle:
        fields = ['video', 'qid', 'question', 'answer', 'type', 'a0', 'a1', 'a2', 'a3', 'a4']
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for i in range(10):
            for q, kind in enumerate(('TN', 'DC')):
                writer.writerow(dict(video=i, qid=q, question='what?', answer=0, type=kind,
                                     **{f'a{j}': str(j) for j in range(5)}))
    monkeypatch.setattr(transfer_data, 'NEXTQA_VAL_SHA256', sha256(annotation))
    ovo = tmp_path / 'ovo'
    ovo.mkdir()
    diag = [f'v{i}.mp4' for i in range(100) if video_half(f'v{i}.mp4', 7, .5) == 'diagnostic'][:10]
    rows = []
    for i, v in enumerate(diag):
        (media / v).symlink_to(media / f'{i}.mp4')
        for task in ('ASI', 'OCR'):
            rows.append(dict(id=f'{v}-{task}', video=v, time=1, task=task, question='what?', options=['a','b'], answer=0))
    # Low-FPS/early-query equivalent: fewer than 4 frames in the causal prefix.
    # A query exactly at the fourth frame is eligible; the earlier query is not.
    rows.extend([{**rows[0], 'id': 'short-prefix', 'time': .1},
                 {**rows[1], 'id': 'boundary-prefix', 'time': 3 / 16}])
    qpath = ovo / 'questions.jsonl'
    qpath.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (ovo / 'inventory.json').write_text(json.dumps(dict(questions_sha256=sha256(qpath),
        hash_split={'seed': 7, 'fraction': .5}, sources={'video_root': str(media)})))
    base = tmp_path / 'baseline'
    (base / 'train/grid').mkdir(parents=True)
    train = tmp_path / 'training'
    train.mkdir()
    manifest = train / 'manifest.jsonl'
    manifest.write_text(json.dumps(dict(id='tr', video='unrelated.mp4', source_aliases=['unrelated'],
        question='what?', options=['a','b'], answer=0, split='train'))+'\n')
    (base / 'experiment-config.json').write_text(json.dumps({'data_out': str(train)}))
    (base / 'train/grid/run.json').write_text(json.dumps(dict(manifest_sha256=sha256(manifest), ids=['tr'],
                                                            videos={'unrelated.mp4': 'not-a-benchmark-hash'})))
    cfg=dict(baseline_run=str(base), nextqa_annotation=str(annotation), ovo_transfer=str(ovo),
             nextqa_videos=10, population_seed=17, nextqa_roots=[str(media)], archive_workers=2,
             steps=2, workers=2, pixels=65536, pool_grid=[4,5],
             minimums={d: {'questions':10,'videos':10} for d in ('ovo','nextqa')})
    result = transfer_data.prepare(cfg, tmp_path / 'prepared')
    assert len(transfer_data.jsonl(result['ovo'])) == 21
    assert len(transfer_data.jsonl(result['nextqa'])) == 20
    excluded = json.loads((tmp_path / 'prepared/exclusions.json').read_text())['excluded']
    assert [r['id'] for r in excluded] == ['short-prefix']
    assert excluded[0]['required_distinct_frames'] == 4
    assert excluded[0]['minimum_frames_end'] == 3 / 16
    inv = json.loads((tmp_path / 'prepared/inventory.json').read_text())
    assert inv['ovo']['candidate_questions'] == 22
    assert inv['ovo']['excluded_by_task'] == {'ASI': 1}
    assert inv['ovo']['primary_questions'] == 10
    # Reuse media and validated full-decode caches in a NEW versioned directory.
    monkeypatch.setattr(transfer_data, 'verify_video', lambda *a: pytest.fail('repeated full decode'))
    reused_cfg = {**cfg, 'reuse_preparation': str(tmp_path / 'prepared')}
    reused = transfer_data.prepare(reused_cfg, tmp_path / 'reused')
    assert transfer_data.jsonl(reused['ovo']) == transfer_data.jsonl(result['ovo'])
    with pytest.raises(ValueError, match='different experiment/data settings'):
        transfer_data.prepare({**reused_cfg, 'steps': 3}, tmp_path / 'bad-reuse')
    monkeypatch.setattr(transfer_data, 'ensure_media', lambda *a, **kw: pytest.fail('resume rescanned archives'))
    assert transfer_data.prepare(cfg, tmp_path / 'prepared') == result
    complete = json.loads((tmp_path / 'prepared/complete.json').read_text())
    assert complete['training_overlap'] == 0
    # An edited population cannot silently reuse the previous prepared directory.
    changed = {**cfg, 'population_seed': 18}
    with pytest.raises(ValueError):
        transfer_data.prepare(changed, tmp_path / 'prepared')
    assert space_checks == [(tmp_path / 'prepared', 30), (tmp_path / 'reused', 30),
                            (tmp_path / 'bad-reuse', 30)] + [(tmp_path / 'prepared', 30)] * 2


@pytest.mark.parametrize('free_gib', [14.7, 29.9, 30, 40])
def test_external_prepare_preserves_production_disk_reserve(monkeypatch, tmp_path, free_gib):
    from types import SimpleNamespace

    from trace_vl.core_vlm import prepare, transfer_data

    monkeypatch.setattr(prepare.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=int(free_gib * 2**30)))
    if free_gib < 30:
        # The real transfer entry point must stop before reading any dataset.
        with pytest.raises(ValueError, match='reserve is 30 GiB'):
            transfer_data.prepare({}, tmp_path / 'prepared')
    else:
        prepare.check_space(tmp_path, 30)
