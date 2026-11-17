import copy
import csv
import json
import subprocess
import zipfile

import pytest
import torch

from trace_vl.core_vlm import mixed_data, mixed_experiment
from trace_vl.core_vlm.data import read_split, sha256
from trace_vl.core_vlm.prepare import verify_video, write_json
from trace_vl.core_vlm.train import tensor_digest
from trace_vl.v3.run_io import atomic_text


def row(i, video, **kw):
    return dict(id=str(i), video=str(video), source_group=str(video), source_aliases=[str(video)],
                question='what?', options=['a', 'b'], answer=0, protocol='offline', split='train', **kw)


def test_general_data_follows_recorded_versioned_path_and_checks_provenance(tmp_path):
    general = tmp_path / 'data/core-llava-youtube-v1-8c5e1b7'
    atomic_text(general / 'manifest.jsonl', json.dumps(row('g', 'g.mp4')) + '\n')
    expected = sha256(general / 'manifest.jsonl')
    write_json(general / 'complete.json', {'manifest_sha256': expected})
    write_json(general / 'verification.json', {'videos': {}})
    baseline, external = tmp_path / 'baseline', tmp_path / 'external'
    write_json(baseline / 'experiment-config.json', {'data_out': str(general)})
    write_json(baseline / 'train/grid/run.json', {'manifest_sha256': expected})
    write_json(external / 'inputs.json', {'config': {'baseline_run': str(baseline)},
                                        'training_manifest': expected})
    cfg = {'external_data': str(external), 'general_data': None}
    assert mixed_data.resolve_general_data(cfg) == general
    # Explicit overrides are respected; a missing override must not fall back.
    assert mixed_data.resolve_general_data({'general_data': str(general)}) == general
    with pytest.raises(ValueError, match='prepared general QA data missing'):
        mixed_data.resolve_general_data({**cfg, 'general_data': str(tmp_path / 'wrong')})
    (general / 'manifest.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='recorded provenance'):
        mixed_data.resolve_general_data(cfg)
    write_json(baseline / 'train/grid/run.json', {'manifest_sha256': 'different'})
    with pytest.raises(ValueError, match='baseline and frozen transfer'):
        mixed_data.resolve_general_data(cfg)


def test_selection_order_independent_equal_families_and_fail_closed():
    temporal = [row(i, f't{i//3}') for i in range(12)]
    general = [row(f'g{i}', f'g{i//2}') for i in range(12)]
    a = mixed_data.balanced_selection(temporal, general, 4, 17)
    b = mixed_data.balanced_selection(temporal[::-1], general[::-1], 4, 17)
    assert a == b
    assert len(a) == 8
    assert len({r['video'] for r in a if r['mixture_family'] == 'temporal'}) == 4
    with pytest.raises(ValueError, match='insufficient'):
        mixed_data.balanced_selection(temporal, general[:2], 4, 17)


def test_training_source_id_collision_preserves_distinct_prompts_and_strict_eval(tmp_path):
    from trace_vl.core_vlm.transfer_data import nextqa_rows
    path = tmp_path / 'train.csv'
    base = dict(video='3617262562', qid='2', type='TC', question='how did the man react?',
                answer='2', **{f'a{i}': f'option {i}' for i in range(5)})
    other = {**base, 'type': 'TN', 'question': 'how did the man react in the middle?', 'answer': '4'}
    def save(rows):
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(base))
            writer.writeheader()
            writer.writerows(rows)
    save([base, other])
    with pytest.raises(ValueError, match='duplicate NExT-QA questions'):
        nextqa_rows(path, videos=1)
    rows = nextqa_rows(path, videos=1, training_id_collisions=True)
    assert len(rows) == len({r['id'] for r in rows}) == 2
    assert {r['annotation_id'] for r in rows} == {'nextqa-3617262562-2'}
    assert {(r['question'], r['answer']) for r in rows} == {(base['question'], 2), (other['question'], 4)}
    save([other, base])
    assert nextqa_rows(path, videos=1, training_id_collisions=True) == rows
    for conflict in [base, {**base, 'answer': '4'}]:
        save([base, conflict])
        with pytest.raises(ValueError, match='duplicate or conflicting'):
            nextqa_rows(path, videos=1, training_id_collisions=True)
    save([base])
    assert nextqa_rows(path, videos=1, training_id_collisions=True) == nextqa_rows(path, videos=1)


def test_source_wrappers_and_exact_bytes_are_excluded():
    rows = [row('1', 'v_abcdefghijk.mp4'), row('2', 'other.mp4'), row('3', 'safe.mp4')]
    kept, rejected = mixed_data.disjoint_filter(rows, {'abcdefghijk'}, {'duplicate'},
        {'v_abcdefghijk.mp4': 'one', 'other.mp4': 'duplicate', 'safe.mp4': 'safe'})
    assert [r['id'] for r in kept] == ['3']
    assert {r['reason'] for r in rejected} == {'heldout_source_overlap', 'heldout_exact_bytes'}


def test_gate_requires_gain_and_retention_against_both_references():
    good = {'choice': {'delta_pp': 3, 'ci95_pp': [1, 5]}}
    results = {d: {'paired': {f'temporal_minus_{r}': {'primary': copy.deepcopy(good),
               'no_harm': copy.deepcopy(good)} for r in ('plain', 'original_grid')}} for d in ('ovo', 'nextqa')}
    assert mixed_experiment.decision(results)['passed']
    for dataset, ref, endpoint, field, value in [
        ('nextqa', 'plain', 'primary', 'delta_pp', 1.9),
        ('nextqa', 'original_grid', 'primary', 'ci95_pp', [-.1, 5]),
        ('ovo', 'plain', 'primary', 'ci95_pp', [-2, 5]),
        ('ovo', 'original_grid', 'no_harm', 'ci95_pp', [-2.1, 5]),
        ('nextqa', 'plain', 'no_harm', 'ci95_pp', None)]:
        bad = copy.deepcopy(results)
        bad[dataset]['paired'][f'temporal_minus_{ref}'][endpoint]['choice'][field] = value
        assert not mixed_experiment.decision(bad)['passed']


def make_training(root, name):
    from safetensors.torch import save_file
    folder = root / name
    checkpoint = folder / 'checkpoint-000002'
    (checkpoint / 'adapter').mkdir(parents=True)
    params = {'lora_A': torch.ones(2)}
    if name == 'temporal':
        params['temporal.up.weight'] = torch.ones(2, 2)
    metrics = [dict(update=i, loss=.5, gradient_norm=1., temporal_gradient_norm_after_clip=1.,
                    question_ids_by_rank=[['q']], lr=.001, answer_tokens=2) for i in (1, 2)]
    state = dict(run=name, update=2, parameters=params, optimizer={}, rng=[{}], metrics=metrics)
    torch.save(state, checkpoint / 'training.pt')
    save_file({'lora_A': params['lora_A']}, checkpoint / 'adapter/adapter_model.safetensors')
    write_json(checkpoint / 'adapter/adapter_config.json', {})
    write_json(checkpoint / 'complete.json', dict(run=name, update=2, adapter_sha256=tensor_digest(params),
        files={p: sha256(checkpoint / p) for p in ['training.pt', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']}))
    write_json(folder / 'run.json', dict(id=name, model='original', manifest_sha256='fixed', ids=['q'],
        videos={'v':'h'}, world_size=1, objective='QA', initial_checkpoint=None,
        config={'temporal': mixed_experiment.ARMS[name], 'seed':17, 'updates':2}))
    write_json(folder / 'initialization.json', dict(run=name, lora_sha256=tensor_digest({'lora_A':torch.zeros(2)}),
                                                  trainable_parameters=sum(p.numel() for p in params.values())))
    atomic_text(folder / 'metrics.jsonl', ''.join(json.dumps(r)+'\n' for r in metrics))


def test_real_checkpoint_audit_rejects_schedule_drift_and_corruption(tmp_path):
    for name in mixed_experiment.ARMS:
        make_training(tmp_path, name)
    assert mixed_experiment.audit_training(tmp_path, 2)['passed']
    path = tmp_path / 'temporal/metrics.jsonl'
    original = path.read_text()
    changed = [json.loads(x) for x in original.splitlines()]
    changed[1]['question_ids_by_rank'] = [['another']]
    path.write_text(''.join(json.dumps(r)+'\n' for r in changed))
    with pytest.raises(ValueError, match='sample'):
        mixed_experiment.audit_training(tmp_path, 2)
    path.write_text(original)
    (tmp_path / 'temporal/checkpoint-000002/adapter/adapter_model.safetensors').write_bytes(b'bad')
    with pytest.raises(ValueError, match='file differs'):
        mixed_experiment.audit_training(tmp_path, 2)


@pytest.mark.parametrize('repair_missing', [False, True])
def test_prepare_real_archived_media_frozen_resume_and_split_exclusion(tmp_path, monkeypatch, repair_missing):
    monkeypatch.setattr(mixed_data, 'check_space', lambda *a: None)
    monkeypatch.setattr(mixed_data, 'benchmark_inventory', lambda *a, **k: ({'99'}, []))
    videos = tmp_path / 'videos'
    videos.mkdir()
    for i in range(1, 8):
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'color=c=0x{i*25:02x}4080:s=192x160:r=16:d=2',
                        '-threads','1','-c:v','libx264',str(videos/f'{i}.mp4')],check=True)
    archives = tmp_path/'archives'
    archives.mkdir()
    with zipfile.ZipFile(archives/'videos.zip','w') as z:
        for i in ((1,) if repair_missing else (1, 2)):
            z.write(videos/f'{i}.mp4',f'NExTVideo/{i}.mp4')
    def annotation(path, ids):
        with path.open('w') as f:
            w=csv.DictWriter(f, fieldnames=['video','qid','type','answer','question',*[f'a{i}' for i in range(5)]])
            w.writeheader()
            for i in ids:
                w.writerow(dict(video=i,qid=0,type='TN',answer=0,question='before?',**{f'a{j}':str(j) for j in range(5)}))
    train_csv, val_csv = tmp_path/'train.csv',tmp_path/'val.csv'
    annotation(train_csv,[1,2,99])  # benchmark source 99 must never be selected
    annotation(val_csv,[7])
    monkeypatch.setattr(mixed_data,'NEXTQA_TRAIN_SHA256',sha256(train_csv))
    monkeypatch.setattr(mixed_data,'NEXTQA_VAL_SHA256',sha256(val_csv))
    general = tmp_path/'general'
    general.mkdir()
    rows = [row(f'g{i}',videos/f'{i}.mp4') for i in (3,4,5)]
    rows[-1]['split']='val'
    atomic_text(general/'manifest.jsonl',''.join(json.dumps(r)+'\n' for r in rows))
    write_json(general/'complete.json',{'manifest_sha256':sha256(general/'manifest.jsonl')})
    write_json(general/'verification.json',{'videos':{r['video']:verify_video(r['video'],4) for r in rows}})
    external=tmp_path/'external'
    external.mkdir()
    for dataset,i in [('ovo',6),('nextqa',7)]:
        atomic_text(external/f'{dataset}.jsonl',json.dumps({**row(dataset,videos/f'{i}.mp4'), 'split':'val'})+'\n')
    write_json(external/'inventory.json',{})
    write_json(external/'complete.json',{'manifests':{d:sha256(external/f'{d}.jsonl') for d in ('ovo','nextqa')},
        'audit_files':{'inventory.json':sha256(external/'inventory.json')},
        'videos':{str(videos/f'{i}.mp4'):sha256(videos/f'{i}.mp4') for i in (6,7)}})
    cfg=dict(external_data=str(external),general_data=str(general),nextqa_train=str(train_csv),nextqa_val=str(val_csv),
        benchmark_roots=[],nextqa_roots=[str(archives)],seed=17,questions_per_family=2,minimum_videos_per_family=2,
        workers=2,archive_workers=2,steps=2,pixels=65536,pool_grid=[4,5])
    if repair_missing:
        from trace_vl.core_vlm.prepare import digest
        from trace_vl.core_vlm.scale_repair import reuse_failed_preparation
        cfg['minimum_videos_per_family'] = 1
        prior = tmp_path / 'prior'
        # Real archive extraction must fail with the default strict policy.
        with pytest.raises(ValueError, match='selected training media unavailable'):
            mixed_data.prepare(cfg, prior / 'mixture')
        import shutil
        shutil.copytree(general, prior / 'general')
        manifest = (general / 'manifest.jsonl').read_text()
        verification = json.loads((general / 'verification.json').read_text())['videos']
        write_json(prior / 'general/run.json', {'id': 'general-run'})
        write_json(prior / 'general/complete.json', {
            'run': 'general-run', 'manifest_sha256': sha256(general / 'manifest.jsonl'),
            'id': digest(['general-run', {p:v['sha256'] for p,v in verification.items()}, manifest])})
        write_json(prior / 'preparation.json', {'config': cfg})
        repair_cfg = {**cfg, 'repair_from': str(prior), 'exclude_training_videos': ['2.mp4']}
        record = reuse_failed_preparation(repair_cfg, tmp_path / 'repaired')
        assert reuse_failed_preparation(repair_cfg, tmp_path / 'repaired') == record
        assert (tmp_path / 'repaired/general/manifest.jsonl').read_bytes() == (general / 'manifest.jsonl').read_bytes()
        with pytest.raises(ValueError, match='original scale configuration'):
            reuse_failed_preparation({**repair_cfg, 'seed': 18}, tmp_path / 'wrong')
        with pytest.raises(ValueError, match='prior missing-only'):
            reuse_failed_preparation({**repair_cfg, 'exclude_training_videos': ['1.mp4']}, tmp_path / 'wrong')
        cfg = {**cfg, 'media_repair': record, 'general_data': str(tmp_path / 'repaired/general'),
               'nextqa_roots': [str(prior / 'mixture/nextqa-media/media')]}
        assert not (prior / 'mixture/complete.json').exists()
    path, _ = mixed_data.prepare(cfg,tmp_path/'prepared')
    result=read_split(path,'train')
    if repair_missing:
        audit = json.loads((path.parent / 'media-exclusion.json').read_text())
        assert audit['selected_before'] == 4 and audit['retained_after'] == 3
        assert audit['replacement_questions'] == 0
        assert [r['video'] for r in audit['excluded_questions']] == ['2.mp4']
        assert json.loads((path.parent / 'media.json').read_text())['archives_scanned'] == []
        assert 'media-exclusion.json' in json.loads((path.parent / 'complete.json').read_text())['files']
    assert len(result)==(3 if repair_missing else 4) and all('99' not in r['source_aliases'] for r in result)
    assert {r['mixture_family'] for r in result}=={'general','temporal'}
    monkeypatch.setattr(mixed_data,'verify_video',lambda *a:pytest.fail('resume decoded again'))
    monkeypatch.setattr(mixed_data,'ensure_media',lambda *a,**k:pytest.fail('resume rescanned archives'))
    assert mixed_data.prepare(cfg,tmp_path/'prepared')[0]==path
    with pytest.raises(ValueError):
        mixed_data.prepare({**cfg,'seed':18},tmp_path/'prepared')
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='mixture changed'):
        mixed_data.prepare(cfg,tmp_path/'prepared')


@pytest.mark.parametrize('fail_smoke', [False, True])
def test_queue_smoke_before_full_training_and_fresh_initialization(tmp_path, monkeypatch, fail_smoke):
    ext = tmp_path/'external'
    ext.mkdir()
    cfg = dict(updates=4, flow_updates=2, gpus=[0,1], model='model', steps=2, pixels=65536,
        pool_grid=[4,5], timestamp_policy='native-index-fps', data_out=str(tmp_path/'data'),
        external_data=str(ext), minimum_free_gib=30, accumulation=2, rank=8, lr=.00002,
        seed=17, save_every=2, cache_mb=0, max_new_tokens=4)
    write_json(ext/'inputs.json',{'config':cfg})
    manifest=tmp_path/'train.jsonl'
    rows=[{**row(i,f'{i}.mp4'), 'split':'val'} for i in range(4)]
    atomic_text(manifest,''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(mixed_experiment,'prepare',lambda *a:(manifest,{'ovo':manifest,'nextqa':manifest}))
    jobs=[]
    def run_jobs(batch, env, logs):
        from pathlib import Path
        for name,cmd,devices in batch:
            jobs.append((name,cmd))
            if 'trace_vl.core_vlm.train' in cmd:
                folder=Path(cmd[cmd.index('--out')+1])
                write_json(folder/'run.json',{'id':'test'})
    monkeypatch.setattr(mixed_experiment,'run_jobs',run_jobs)
    monkeypatch.setattr(mixed_experiment,'checked_checkpoint',lambda *a: {})
    def audit(path, updates):
        if fail_smoke:
            raise ValueError('smoke failure')
        assert updates in (2,4)
        return {'passed':True}
    monkeypatch.setattr(mixed_experiment,'audit_training',audit)
    monkeypatch.setattr(mixed_experiment,'matched_report',lambda *a:{})
    monkeypatch.setattr(mixed_experiment,'decision',lambda *a:{'passed':False})
    if fail_smoke:
        with pytest.raises(ValueError,match='smoke failure'):
            mixed_experiment.run(cfg,tmp_path/'out')
        assert not any(n.startswith('train-') for n,c in jobs)
    else:
        mixed_experiment.run(cfg,tmp_path/'out')
        train_jobs=[(n,c) for n,c in jobs if 'trace_vl.core_vlm.train' in c]
        assert [n for n,c in train_jobs]==['smoke-train-plain','smoke-train-temporal','train-plain','train-temporal']
        assert all('--init-checkpoint' not in c for n,c in train_jobs)
        assert [c[c.index('--temporal')+1] for n,c in train_jobs]==['none','causal','none','causal']
        assert (tmp_path/'out/report.json').is_file()
