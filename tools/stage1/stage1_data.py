"""Convert LLaVA MC safely: content IDs, deduplication, conflicts, media audit, split manifests."""
import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

from trace_vl.v3.data import convert_llava_mc


def canonicalize(rows):
    result=[]
    seen={}
    duplicates=0
    for row in rows:
        assert row['question'].strip() and 2<=len(row['options'])<=8
        assert isinstance(row['answer'],int) and 0<=row['answer']<len(row['options'])
        identity={'video':row['video'],'question':' '.join(row['question'].split()),
                  'options':[' '.join(x.split()) for x in row['options']]}
        key='llava-mc-q-'+hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if key in seen:
            if seen[key]!=row['answer']:
                raise ValueError('Conflicting gold answers for '+key)
            duplicates+=1
            continue
        seen[key]=row['answer']
        fixed=dict(row)
        fixed['upstream_record_id']=row['id']
        fixed['id']=key
        result.append(fixed)
    return result,duplicates


def limit_video_groups(rows,maximum,seed=7):
    if not maximum:
        return rows
    grouped={}
    for row in rows:
        grouped.setdefault(row['video'],[]).append(row)
    keys=sorted(grouped)
    random.Random(seed).shuffle(keys)
    selected=[]
    for video in keys:
        if len(selected)+len(grouped[video])<=maximum:
            selected.extend(grouped[video])
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True,help='Fresh output directory')
    p.add_argument('--video-root',type=Path)
    p.add_argument('--max-seconds',type=float,default=180)
    p.add_argument('--val-fraction',type=float,default=.1)
    p.add_argument('--min-records',type=int,default=1)
    p.add_argument('--max-train-questions',type=int,default=0,help='0 = all; select whole video groups')
    p.add_argument('--max-val-questions',type=int,default=0,help='0 = all; select whole video groups')
    p.add_argument('--annotations-only',action='store_true',help='Schema validation only; not training-ready')
    args=p.parse_args()
    assert 0<args.val_fraction<1
    assert args.max_train_questions>=0 and args.max_val_questions>=0
    assert args.annotations_only or args.video_root is not None,'Training preparation needs --video-root'
    assert not args.output.exists(),'Use a fresh output directory'
    annotation_paths=[Path(x).resolve(strict=True) for x in args.annotations]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        parsed_path=Path(scratch)/'parsed.jsonl'
        conversion=convert_llava_mc([str(x) for x in annotation_paths],str(parsed_path),
            video_root=None if args.annotations_only else str(args.video_root),val_fraction=args.val_fraction)
        parsed=[json.loads(x) for x in parsed_path.read_text().splitlines()]
        rows,duplicates=canonicalize(parsed)
    rows=limit_video_groups([r for r in rows if r['split']=='train'],args.max_train_questions)+\
         limit_video_groups([r for r in rows if r['split']=='val'],args.max_val_questions)
    media={}
    excluded={}
    if not args.annotations_only:
        import cv2
        root=args.video_root.resolve(strict=True)
        for video in sorted({r['video'] for r in rows}):
            path=(root/video).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError('Video outside --video-root: '+video)
            cap=cv2.VideoCapture(str(path))
            fps,count=cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            checks=[]
            if fps>0 and count>0:
                for index in [0,max(0,count-2)]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                    ok,frame=cap.read()
                    checks.append(bool(ok))
            cap.release()
            if not checks or not all(checks):
                excluded[video]='failed metadata/first-tail decode; repair a derivative before retry'
            elif count/fps>args.max_seconds+.01:
                excluded[video]=f'duration {count/fps:.3f} exceeds whole-clip training limit {args.max_seconds}'
            else:
                stat=path.stat()
                media[video]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                              'seconds':count/fps,'fps':fps,'frames':count,'decode_checks':checks}
        rows=[r for r in rows if r['video'] in media]
    assert len(rows)>=args.min_records,(f'Only {len(rows)} usable QA (need {args.min_records}); '
        f'missing annotation items={conversion.get("missing_videos_skipped",0)}, '
        f'excluded videos={len(excluded)}, examples={list(excluded.items())[:3]}')
    assert len(rows)==len({r['id'] for r in rows})
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='val']
    assert train and val,'Both train and val must contain videos'
    assert not ({r['video'] for r in train}&{r['video'] for r in val})
    report={'training_ready':not args.annotations_only,'conversion':conversion,
        'parsed_questions':len(parsed),'original_distinct_ids':len({r['id'] for r in parsed}),
        'usable_questions':len(rows),'unique_question_ids':len({r['id'] for r in rows}),
        'exact_duplicates_skipped':duplicates,'train_questions':len(train),'val_questions':len(val),
        'question_limits':{'train':args.max_train_questions,'val':args.max_val_questions,'seed':7,
                           'unit':'whole video groups'},
        'train_videos':len({r['video'] for r in train}),'val_videos':len({r['video'] for r in val}),
        'answer_counts':dict(Counter(r['answer'] for r in rows)),'video_disjoint':True,
        'excluded_videos':excluded,'media':media,'max_seconds':args.max_seconds,
        'annotations_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in annotation_paths},
        'identity':'SHA256(video, whitespace-normalized question, ordered options); gold conflicts rejected',
        'note':'Use val.jsonl for evaluation. all.jsonl intentionally includes both splits.'}
    args.output.mkdir()
    for name,items in [('all',rows),('train',train),('val',val)]:
        (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'media','excluded_videos'}},indent=2))


if __name__=='__main__':
    main()
