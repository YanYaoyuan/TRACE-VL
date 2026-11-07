"""Exercise the shell workflow with a fake UV/GPU CLI; no weights or network required."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from trace_vl.v3.config import V3Config

ROOT = Path(__file__).parents[1]


@pytest.fixture
def pipeline(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/evaluate_a100_pilot.sh", project / "scripts/evaluate_a100_pilot.sh")
    (project / "src/trace_vl/v3").mkdir(parents=True)
    (project / "src/trace_vl/v3/identity.py").write_text("# fixture source fingerprint\n")
    data, runs, videos = [tmp_path / name for name in ("data", "runs", "videos")]
    for path in (data, runs, videos):
        path.mkdir()
    records = []
    for index, split in enumerate(("train", "val", "val")):
        video = f"v{index}.mp4"
        (videos / video).write_bytes(b"fixture media, not decoded by fake CLI")
        records.append({"id": str(index), "video": video, "split": split, "answer": 0,
                        "source": "fixture", "task": "qa", "query_time": None})
    train_text = json.dumps(records[0]) + "\n"
    (data / "train.jsonl").write_text(train_text)
    (data / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records[1:]))
    import hashlib

    train_sha = hashlib.sha256(train_text.encode()).hexdigest()
    config_dir = project / "configs/v3"
    config_dir.mkdir(parents=True)
    for arm in ("transition", "b4"):
        cfg = V3Config()
        cfg.transition.p_step_mode = "transition" if arm == "transition" else "pooled"
        cfg.train.steps = 2
        (config_dir / f"a100-pilot-{arm}.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
        run = runs / f"pilot-{arm}"
        target = run / "checkpoints/step-00002"
        (target / "lora").mkdir(parents=True)
        (target / "modules.pt").write_bytes(b"fixture checkpoint")
        (target / "lora/adapter_config.json").write_text("{}")
        (target / "lora/adapter_model.safetensors").write_bytes(b"fixture adapter")
        (target / "checkpoint.json").write_text(json.dumps({"step": 2, "paired_from_same_optimizer_state": True}))
        (run / "run.json").write_text(json.dumps({"config": cfg.to_dict(), "world": 1}))
        (run / "rank-0-resources.json").write_text(json.dumps({"status": "completed", "records_sha256": train_sha}))
        (run / "metrics.jsonl").write_text(''.join(json.dumps({"step": s, "p_step_tokens_mean": 8.0}) + '\n' for s in (1, 2)))
    fake = tmp_path / "fake_uv.py"
    fake.write_text(f"#!{sys.executable}\n" + r'''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
assert args[:4]==['run','--locked','--no-sync','python'],args
args=args[4:]
if args[0]=='-':
    import torch
    torch.cuda.is_available=lambda:True
    sys.argv=['-']+args[1:]
    exec(compile(sys.stdin.read(),'<pipeline-python>','exec'))
elif args[:4]==['-m','trace_vl','v3','budget']:
    print(json.dumps({'arms':{'transition':{'steps':2,'total_visual_tokens':16},'b4':{'steps':2,'total_visual_tokens':16}},'matched':True,'budget_gap':0.0}))
elif args[:4]==['-m','trace_vl','v3','eval']:
    def get(k): return args[args.index(k)+1]
    out=Path(get('--output'));out.mkdir(parents=True,exist_ok=True)
    identity=json.loads((out.parent/'workflow.json').read_text())['eval_runs'][out.name]
    (out/'run.json').write_text(json.dumps(identity))
    rows=[json.loads(s) for s in Path(get('--records')).read_text().splitlines() if s]
    broken=os.environ.get('FAKE_B4_ERROR')=='1' and out.name=='b4'
    if broken: rows=rows[:1]
    target=out/'predictions.0of1.jsonl'
    prior=[json.loads(s) for s in target.read_text().splitlines()] if target.exists() else []
    done={r['id'] for r in prior}
    with target.open('a') as f:
        for r in rows:
            if r['id'] not in done:
                f.write(json.dumps({'run':identity['id'],'id':r['id'],'answer':r['answer'],'prediction':0,'video_key':r['video'],'source':'fixture','task':'qa','lm_tokens_per_step':8,'kv_mb':1,'answer_seconds':0.1})+'\n')
    (out/'errors.0of1.json').write_text(json.dumps({'v2':'simulated decoder failure'} if broken else {}))
else: raise AssertionError(args)
''')
    fake.chmod(0o755)
    env = {**os.environ, "TRACE_RUNS": str(runs), "TRACE_DATA_DIR": str(data), "TRACE_VIDEO_ROOT": str(videos),
           "TRACE_UV_BIN": str(fake), "TRACE_EVAL_STEP": "2", "PYTHONPATH": str(ROOT / "src"),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return project, runs, data, env


def call(pipeline, *args, **extra):
    project, _, _, env = pipeline
    return subprocess.run(["bash", str(project / "scripts/evaluate_a100_pilot.sh"), *args],
                          cwd=project, env={**env, **extra}, text=True, capture_output=True, timeout=90)


def test_pipeline_check_only_and_complete_summary(pipeline):
    _, runs, _, _ = pipeline
    result = call(pipeline, "--check-only")
    assert result.returncode == 0, result.stderr
    assert not (runs / "pilot-eval").exists()
    result = call(pipeline)
    assert result.returncode == 0, result.stderr
    summary = json.loads((runs / "pilot-eval/summary.json").read_text())
    assert summary["completed"] and summary["budget_matched"]
    assert summary["scores"]["transition"]["n"] == 2
    assert summary["paired"]["transition_minus_b4"]["delta_accuracy_pp"] == 0
    assert summary["gate_summary"]["transition"]["gist_only_records"] == 2


def test_pipeline_failed_arm_has_no_success_summary_and_resumes(pipeline):
    _, runs, _, _ = pipeline
    result = call(pipeline, FAKE_B4_ERROR="1")
    assert result.returncode != 0
    assert not (runs / "pilot-eval/summary.json").exists()
    result = call(pipeline)
    assert result.returncode == 0, result.stderr
    rows = (runs / "pilot-eval/b4/predictions.0of1.jsonl").read_text().splitlines()
    assert len(rows) == 2  # retry appends the missing record rather than duplicating it


def test_pipeline_rejects_changed_checkpoint_before_reusing_results(pipeline):
    _, runs, _, _ = pipeline
    result = call(pipeline)
    assert result.returncode == 0, result.stderr
    previous = (runs / "pilot-eval/summary.json").read_bytes()
    (runs / "pilot-transition/checkpoints/step-00002/modules.pt").write_bytes(b"different checkpoint")
    result = call(pipeline)
    assert result.returncode != 0
    assert 'different existing evaluation' in result.stderr
    assert (runs / "pilot-eval/summary.json").read_bytes() == previous
