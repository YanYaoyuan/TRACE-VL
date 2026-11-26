"""Post hoc hash-cluster bootstrap of saved old diagnostics; JSON/NumPy math only."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
for name in ('test-a','b0','protected','capacity'):
 parser.add_argument('--'+name+'-root',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
names=('test_a','b0','protected','capacity')
folders={name:getattr(args,name+'_root') for name in names}
read=lambda p:json.loads(Path(p).read_text())
base=read(folders['capacity']/'inputs.json')['reference']
media={s['source_group']:s['video_sha256'] for s in base['sources']}
source_ids=sorted(media);assert len(source_ids)==77 and len(set(media.values()))==73
clusters=sorted(set(media.values()));groups=[[i for i,s in enumerate(source_ids) if media[s]==c] for c in clusters]
sizes=np.array([len(g) for g in groups]);assert sorted(sizes.tolist())==[1]*69+[2]*4
rng=np.random.default_rng(7);picks=rng.integers(0,73,(10000,73));denominators=sizes[picks].sum(1)
reports={k:read(v/'report.json') for k,v in folders.items()}
raw={};input_digests={}
for name,p in folders.items():
 paths=sorted((p/'questions').glob('*.json'));rows=[read(f) for f in paths];assert len(rows)==385
 raw[name]={(r['source_group'],r['delay_seconds']):r for r in rows}
 assert len(raw[name])==385
 value=[(str(f.relative_to(p)),hashlib.sha256(f.read_bytes()).hexdigest()) for f in [p/'inputs.json',p/'report.json',*paths]]
 input_digests[name]=hashlib.sha256(json.dumps(value).encode()).hexdigest()

def metric(row):
 result={}
 for name,prediction in row['predictions'].items():
  v=prediction['option_logits_fp32'];top=max(v)
  result[name]={'nll':top+math.log(sum(math.exp(x-top) for x in v))-v[row['gold']],
                'acc':float(max(range(len(v)),key=v.__getitem__)==row['gold'])}
 return result

def average(values,names):
 return {m:float(np.mean([values[n][m] for n in names])) for m in ('nll','acc')}

weights={
 'carried_state':{'window_stream':1,'recent_replay':-1},
 'history_kv_given_state':{'S1K1':1,'window_stream':-1},
 'history_kv_fork_S1':{'S1K1':1,'S1K0':-1},
 'history_kv_fork_S0':{'S0K1':1,'S0K0':-1},
 'state_fork_K0':{'S1K0':1,'S0K0':-1},
 'state_fork_K1':{'S1K1':1,'S0K1':-1},
 'interaction':{'S1K1':1,'S1K0':-1,'S0K1':-1,'S0K0':1},
 'full_kv_headroom':{'full_stream':1,'window_stream':-1},
 'reservoir_gap_to_full':{'full_stream':1,'S1K1':-1},
 'snapshot_headroom':{'snapshot_replay':1,'window_stream':-1}}

def reading(ci,metric):
 if metric=='nll':return 'favors_first' if ci[1]<0 else 'favors_second' if ci[0]>0 else 'inconclusive'
 return 'favors_first' if ci[0]>0 else 'favors_second' if ci[1]<0 else 'inconclusive'

def boot(values,old,metric_name):
 values=np.asarray(values,dtype=float);assert np.isfinite(values).all() and len(values)==77
 assert abs(float(values.mean())-old['mean'])<1e-12
 sums=np.array([values[g].sum() for g in groups]);means=sums/sizes
 ratio=sums[picks].sum(1)/denominators
 equal=means[picks].mean(1)
 for pick,observed,balanced in zip(picks[:100],ratio[:100],equal[:100],strict=True):
  expanded=[values[i] for cluster in pick for i in groups[cluster]]
  assert abs(float(np.mean(expanded))-observed)<1e-12
  assert abs(float(np.mean([np.mean(values[groups[cluster]]) for cluster in pick]))-balanced)<1e-12
 ci=np.quantile(ratio,[.025,.975]).tolist();ec=np.quantile(equal,[.025,.975]).tolist()
 return {'original_source_id_bootstrap':old,'same_source_id_weighted_estimand_hash_cluster_bootstrap':{
          'mean':float(values.mean()),'ci95':ci,'bootstrap_sd':float(ratio.std(ddof=1)),'reading':reading(ci,metric_name)},
         'equal_video_weighted_posthoc_estimand':{'mean':float(means.mean()),'ci95':ec,'bootstrap_sd':float(equal.std(ddof=1)),'reading':reading(ec,metric_name)}}

analysis={k:{} for k in names}
for delay in (0,15,60,180,300):
 vectors={k:defaultdict(lambda:defaultdict(list)) for k in names}
 for source in source_ids:
  rows={k:raw[k][source,delay] for k in names}
  assert len({r['id'] for r in rows.values()})==1 and len({r['gold'] for r in rows.values()})==1
  v={k:metric(r) for k,r in rows.items()};a=v['test_a']
  for f in ('S1K1','S1K0','S0K1','S0K0'):a[f]=average(a,[f'reservoir{s}:{f}' for s in (17,29,43)])
  for name,w in weights.items():
   for m in ('nll','acc'):vectors['test_a'][name][m].append(sum(sign*a[arm][m] for arm,sign in w.items()))
  b={'reservoir':a['S1K1'],**v['b0']}
  p={'protected':average(v['protected'],[f'protected{s}' for s in (17,29,43)]),'reservoir':a['S1K1']}
  c={f'R{cap}':average(v['capacity'],[f'r{cap}s{s}' for s in (17,29,43)]) for cap in (32,128)}
  for family,arms,pairs in [('b0',b,[('reservoir','window40'),('window40','window8'),('reservoir','window8')]),('protected',p,[('protected','reservoir')]),('capacity',c,[('R128','R32')])]:
   for first,second in pairs:
    for m in ('nll','acc'):vectors[family][first+'_minus_'+second][m].append(arms[first][m]-arms[second][m])
 for family,contrasts in vectors.items():
  analysis[family][str(delay)]={name:{m:boot(values,reports[family]['analysis'][str(delay)]['contrasts'][name][m],m) for m,values in metrics.items()} for name,metrics in contrasts.items()}

for delay,contrasts in analysis['test_a'].items():
 for metric_name,result in contrasts['interaction'].items():
  for name in ('same_source_id_weighted_estimand_hash_cluster_bootstrap','equal_video_weighted_posthoc_estimand'):
   value=result[name];ci=value['ci95'];value.pop('reading',None);value['signed_reading']='negative' if ci[1]<0 else 'positive' if ci[0]>0 else 'inconclusive'
qmap={s['source_group']:{q['delay_seconds']:q for q in s['queries']} for s in base['sources']}
duplicate_details=[]
for g in groups:
 if len(g)<2:continue
 s1,s2=[source_ids[i] for i in g];q1,q2=qmap[s1][300],qmap[s2][300]
 duplicate_details.append({'source_groups':[s1,s2],'same_question_options_gold':all(q1[k]==q2[k] for k in ('question','options','answer')),'same_target_time':q1['target_time']==q2['target_time']})
output={'posthoc':True,'source_ids':77,'byte_distinct_media_clusters':73,'draws':10000,'seed':7,'explicit_resample_expansion_checks':15000,
        'cluster_order':'sorted video SHA256','duplicate_details':duplicate_details,'input_record_set_digests':input_digests,
        'method':'Resample73 media-hash clusters with replacement, retaining all77 source facts and within-source memory-seed averaging. Ratio of cluster sums/counts preserves the original source-ID-weighted point estimand. Equal-video cluster means are a separately labelled changed estimand.',
        'scope':'Post hoc pointwise sensitivity, no multiplicity adjustment, no new inference or question selection. Exact-byte hashes are only known dependency clusters; shared underlying/re-encoded/overlapping sources remain unverified. Frozen reports are unchanged.','analysis':analysis}
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'source_ids':77,'media_hash_clusters':73,'draws':10000,'explicit_expansion_checks':15000,'output':str(args.out)}))
