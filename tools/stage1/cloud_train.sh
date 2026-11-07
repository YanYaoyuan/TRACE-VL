#!/usr/bin/env bash
set -euo pipefail
# Run from the verified TRACE-VL checkout. No model/data downloads or cloud actions are automatic.
: "${TRACE_CONFIG:?Set TRACE_CONFIG to a resolved YAML with a local model path}"
: "${TRACE_RECORDS:?Set TRACE_RECORDS to train.jsonl}"
: "${TRACE_VIDEO_ROOT:?Set TRACE_VIDEO_ROOT to the base of annotation-relative video paths}"
: "${TRACE_RUN:?Set TRACE_RUN to a NEW output directory}"
TRACE_GPUS="${TRACE_GPUS:-4}"
TRACE_CLIP_SECONDS="${TRACE_CLIP_SECONDS:-30}"
TRACE_STEPS="${TRACE_STEPS:-100}"
if [[ -e "$TRACE_RUN" ]]; then
  echo "Use a new TRACE_RUN: $TRACE_RUN" >&2
  exit 2
fi
uv run --locked --no-sync python - "$TRACE_CONFIG" "$TRACE_RECORDS" "$TRACE_VIDEO_ROOT" "$TRACE_GPUS" "$TRACE_CLIP_SECONDS" <<'PY'
import json,sys
from pathlib import Path
import torch
from trace_vl.v3.config import load_v3_config
cfg,_=load_v3_config(sys.argv[1])
rows=[json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines() if x.strip()]
assert rows and all(r.get('split')=='train' for r in rows),'Pass train.jsonl, not a mixed/train+val manifest'
assert len(rows)==len({r['id'] for r in rows}),'Duplicate question IDs; run safe data preparation'
assert torch.cuda.device_count()>=int(sys.argv[4])
root=Path(sys.argv[3])
assert all((root/r['video']).is_file() for r in rows),'Missing training videos'
audit=Path(sys.argv[2]).parent/'audit.json'
assert audit.is_file(),'Expected adjacent data audit.json'
data=json.loads(audit.read_text())
assert data['training_ready'] and data['video_disjoint']
assert data['max_seconds']<=float(sys.argv[5]),'Clip window must cover every audited training video'
print({'gpus':torch.cuda.device_count(),'effective_batch':int(sys.argv[4])*cfg.train.grad_accum,
       'records':len(rows),'clip_seconds':float(sys.argv[5]),'model':cfg.backbone.name})
PY
tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname -- "$TRACE_RUN")"
uv run --locked --no-sync torchrun --standalone --nproc_per_node="$TRACE_GPUS" \
  "$tool_dir/train_with_checkpoints.py" --config "$TRACE_CONFIG" --records "$TRACE_RECORDS" \
  --video-root "$TRACE_VIDEO_ROOT" --output "$TRACE_RUN" --clip-seconds "$TRACE_CLIP_SECONDS" \
  --max-steps "$TRACE_STEPS" 2>&1 | tee "$TRACE_RUN.log"
