import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trace_vl.core_vlm import longstream_delayed as core
from trace_vl.core_vlm import longstream_writer_controls as module
from trace_vl.core_vlm.data import sha256
from trace_vl.core_vlm.longstream import BoundedBank, StoredStep, schedule
from trace_vl.core_vlm.prepare import digest, write_json


def step(i):
    return StoredStep(torch.full((16,4),float(i),dtype=torch.bfloat16),(4,4),(float(i),i+.5),
                      (2*i,2*i+1),hashlib.sha256(str(i).encode()).hexdigest())


def test_added_reservoir_is_exact_algorithm_r_with_same_seed_and_four_recent():
    for seed in (17, module.extra_seed('used-source',29), module.extra_seed('used-source',43)):
        expected=BoundedBank(32,seed)
        actual=module.ReservoirWriter(seed)
        for i in range(500):
            expected.append(step(i))
            actual.append(step(i))
            assert [s.indices for s in actual.history]==[s.indices for s in expected.reservoir]
            assert [s.indices for s in actual.recent]==[s.indices for s in list(expected.recent)[-4:]]
        assert actual.audit()['old_seen']==496
        assert actual.audit()['memory_seed']==seed
    with pytest.raises(ValueError):
        module.extra_seed('used-source',99)


def test_new_writers_do_not_change_encoder_pass_or_shared_predictions(tmp_path,monkeypatch):
    metadata=tmp_path/'pts.json'
    write_json(metadata,{'streams':[{'height':128,'width':128,'avg_frame_rate':'2/1'}],
                         'frames':[{'best_effort_timestamp_time':str(i/2)} for i in range(723)]})
    video=tmp_path/'video.bytes'
    video.write_bytes(b'fixed-source')
    queries=core.delayed_queries([{'id':'q','query_time':60.,'question':'What is held?',
                                  'options':['a','b'],'answer':1}],361.)
    indices,times,dropped=schedule(json.loads(metadata.read_text()),360.)
    for q in queries:
        q['planned_encoder_calls']=int(sum(t[-1]<=q['query_time'] for t in times))
        q['target_encoder_calls']=int(sum(t[-1]<=q['target_time'] for t in times))
    source={'source_group':'test-source','memory_seed':17,'video':str(video),'video_sha256':sha256(video),
            'metadata':str(metadata),'metadata_sha256':sha256(metadata),'sampled_frame_indices':indices.tolist(),
            'sampled_frame_times':times.tolist(),'queries':queries}
    inputs={'id':'fixed-test-input','pixels':16384}

    class Backbone:
        def __init__(self):
            self.calls=0

        def preprocess(self,frames):
            return frames,None

        def encode(self,values,grid):
            self.calls+=1
            return (torch.arange(64).reshape(16,4)+float(values[0,0,0,0])).to(torch.bfloat16)

        def merged_grid(self,grid):
            return (4,4)

    def pixels(*args):
        for i in range(len(times)):
            yield np.full((2,2,2,3),i%255,dtype=np.uint8)

    def answer(bb,qa,selected,q,meta,hw,checksum):
        choose=sum(s.indices[0] for s in selected)%2
        return {'read_frame_indices':[i for s in selected for i in s.indices],
                'read_times':[list(s.times) for s in selected],'real_feature_rgb_hashes':[s.rgb_sha256 for s in selected],
                'visual_tokens':16*len(selected),'query_time_encoder_calls':0,'source_video_sha256':checksum,
                'option_logits':[float(choose==0),float(choose==1)],'option_logits_fp32':[float(choose==0),float(choose==1)],
                'choice':choose,'choice_fp32':choose}

    monkeypatch.setattr(core.longstream,'stream_pixels',pixels)
    monkeypatch.setattr(core.longstream,'answer',answer)
    monkeypatch.setattr(core.longstream,'text_answer',lambda bb,q,sha:answer(bb,None,[],q,None,None,sha))
    monkeypatch.setattr(core,'NativeVideoQALoss',lambda *args,**kwargs:object())
    first,second=tmp_path/'baseline',tmp_path/'extended'
    first.mkdir()
    second.mkdir()
    bb,extended_bb=Backbone(),Backbone()
    core.evaluate_source(bb,inputs,source,first,False)
    extension=module.WriterExtension(source['source_group'],second)
    core.evaluate_source(extended_bb,inputs,source,second,False,extension=extension)
    assert bb.calls==extended_bb.calls==len(times)
    for q in queries:
        old=json.loads((first/'questions'/f"{digest(q['id'])}.json").read_text())
        new=json.loads((second/'questions'/f"{digest(q['id'])}.json").read_text())
        assert {a:new['predictions'][a] for a in core.ARMS}==old['predictions']
        assert len(new['predictions'])==10
        core.validate_predictions(new,q,arms=module.ARMS)
        module.check_new_reads(new,q,bytes_per_step=16*4*2)
    receipt=extension.receipt()
    assert sha256(receipt['path'])==receipt['sha256'] and receipt['events']>0
    for line in open(receipt['path']):
        event=json.loads(line)
        index=min(range(1,len(event['candidates'])-1),key=lambda i:(event['interior_costs'][i-1],event['interior_spans'][i-1],event['candidates'][i]['times'][-1]))
        assert index==event['removed_position']
        assert event['removed_indices']==event['candidates'][index]['indices']


def test_prepare_seals_full385_population_stable_resume_and_rejects_reference_change(tmp_path,monkeypatch):
    reference=tmp_path/'old'
    reference.mkdir()
    data=tmp_path/'video.bytes'
    data.write_bytes(b'unchanged')
    metadata=tmp_path/'metadata.json'
    write_json(metadata,{'frames':[]})
    sources=[{'source_group':f'g{i}','video':str(data),'video_sha256':sha256(data),
              'metadata':str(metadata),'metadata_sha256':sha256(metadata),'queries':[{'id':f'q{i}/{d}'} for d in core.DELAYS]}
             for i in range(77)]
    base={'sources':sources,'plain':'/fake/checkpoint','checkpoint':{'run':'fixed'},'model':'/fake/model',
          'model_identity':{'fixed':True},'state_source_code_sha256':'prior','questions':385}
    base['id']=digest(base)
    write_json(reference/'inputs.json',base)
    write_json(reference/'report.json',{'completed':True,'questions':385,'source_videos':77})
    (reference/'reports.tar.gz').write_bytes(b'archive')
    monkeypatch.setattr(module,'REFERENCE_ID',base['id'])
    monkeypatch.setattr(module,'REFERENCE_REPORT_SHA256',sha256(reference/'report.json'))
    monkeypatch.setattr(module,'REFERENCE_ARCHIVE_SHA256',sha256(reference/'reports.tar.gz'))
    monkeypatch.setattr(module,'checked_checkpoint',lambda *args:base['checkpoint'])
    monkeypatch.setattr(module,'checkpoint_identity',lambda *args:base['model_identity'])
    args=SimpleNamespace(out=tmp_path/'new',reference=reference)
    module.prepare(args)
    sealed=json.loads((args.out/'inputs.json').read_text())
    assert sealed['id']==digest({k:v for k,v in sealed.items() if k!='id'}) and sealed['id']!=base['id']
    assert sealed['sources']==base['sources'] and sealed['arms']==list(module.ARMS)
    module.prepare(args)
    assert json.loads((args.out/'inputs.json').read_text())==sealed
    changed=copy.deepcopy(base)
    changed['sources'].pop()
    write_json(reference/'inputs.json',changed)
    with pytest.raises(ValueError,match='reference changed'):
        module.prepare(args)


def test_failed_shared_baseline_replay_never_publishes_success_report(tmp_path,monkeypatch):
    out,reference=tmp_path/'new',tmp_path/'old'
    (out/'sources').mkdir(parents=True)
    (out/'questions').mkdir()
    (reference/'questions').mkdir(parents=True)
    query={'id':'q'}
    trace=tmp_path/'trace.jsonl'
    trace.write_text('')
    write_json(out/'inputs.json',{'writer_protocol_source_sha256':sha256(module.__file__),
                                'writer_class_source_sha256':sha256(module.writers.__file__),
                                'sources':[{'source_group':'g','queries':[query]}]})
    write_json(out/'sources'/f"{digest('g')}.json",{'extension_trace':{'path':str(trace),'sha256':sha256(trace)},
               'final_memory':{'writer_controls':{'writer_append_wall_seconds':{a:0. for a in module.EXTRA_ARMS}}}})
    write_json(out/'questions'/f"{digest('q')}.json",{'predictions':{a:{'choice':0} for a in core.ARMS}})
    write_json(reference/'questions'/f"{digest('q')}.json",{'predictions':{a:{'choice':1} for a in core.ARMS}})

    def base_report(args,arms,output_path):
        write_json(output_path,{'completed':True})

    monkeypatch.setattr(core,'report',base_report)
    args=SimpleNamespace(out=out,reference=reference,smoke=False)
    with pytest.raises(ValueError,match='exactly replay'):
        module.report(args)
    assert not (out/'report.json').exists() and not (out/'smoke-ok.json').exists()
    assert (out/'writer-report.pending.json').exists()
