"""Reject incomplete, duplicate or mixed train/eval predictions before reporting a score."""
import argparse
import json
from pathlib import Path

from trace_vl.v3.harness import score

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--records',type=Path,required=True)
p.add_argument('--predictions',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
records=[json.loads(s) for s in args.records.read_text().splitlines() if s.strip()]
assert records and all(r.get('split') in {'val','test'} for r in records),'Expected held-out records only'
expected={r['id'] for r in records}
assert len(expected)==len(records),'Duplicate input question IDs'
errors=list(args.predictions.glob('errors.*.json'))
assert errors,'Evaluation completion/error files are absent'
assert all(not json.loads(p.read_text()) for p in errors),'Decode/inference errors; inspect errors files'
rows=[json.loads(s) for f in args.predictions.glob('predictions.*.jsonl') for s in f.read_text().splitlines() if s.strip()]
assert len(rows)==len({r['id'] for r in rows})==len(expected),'Duplicate or incomplete predictions'
assert {r['id'] for r in rows}==expected,'Missing/unexpected prediction IDs'
gold={r['id']:r['answer'] for r in records}
assert all(r['answer']==gold[r['id']] for r in rows),'Gold labels differ from held-out manifest'
report=score(str(args.predictions),str(args.output))
assert report['n']==len(expected)
print(json.dumps(report,indent=2))
