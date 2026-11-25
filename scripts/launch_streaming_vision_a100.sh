#!/usr/bin/env bash
# Run the matched streaming vision experiment or resume its evaluation.
set -Eeuo pipefail

repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
cd "$repo"
rev=$(git rev-parse --verify "${TRACE_REV:-HEAD}^{commit}")
runroot=/nas-data/vlm/yyy/tracevl-runs
resume="${TRACE_RESUME_RUN:-}"
out="${resume:-${TRACE_OUT:-$runroot/streaming-vision-${rev:0:7}-001}}"
gpus="${TRACE_CUDA_DEVICES:-0,1,2,3}"
IFS=, read -r -a devices <<< "$gpus"
count=${#devices[@]}

[[ -x "$py" ]] || { echo "Missing Python environment: $py" >&2; exit 1; }
if [[ -n "$resume" && ! -f "$out/training-complete.json" ]]; then
    echo "Missing completed training receipt in $out" >&2
    exit 1
fi
mkdir -p "$out/logs" "$out/tmp"
out=$(realpath "$out")
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo "Already running: $out" >&2; exit 2; }
trap 'code=$?; if (( code )); then echo "STOPPED ($code), inspect $out/logs and rerun with the same inputs." >&2; fi' EXIT

snapshot="$out/source"
if [[ -n "$resume" ]]; then
    snapshot="$out/evaluation-source-$rev"
fi
if [[ ! -f "$snapshot/.revision" ]]; then
    staging=$(mktemp -d "$out/source.tmp.XXXXXX")
    git archive "$rev" | tar -xf - -C "$staging"
    printf '%s\n' "$rev" > "$staging/.revision"
    mv -T "$staging" "$snapshot"
fi
[[ "$(cat "$snapshot/.revision")" == "$rev" ]]
cd "$snapshot"

export PYTHONPATH="$PWD/src" TMPDIR="$out/tmp" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TRACE_EVAL_DECODE_WORKERS="${TRACE_EVAL_DECODE_WORKERS:-2}"
export TRACE_EVAL_SCRATCH="${TRACE_EVAL_SCRATCH:-$repo/.cache/streaming-vision/$(basename "$out")}"

gpu_check() {
    CUDA_VISIBLE_DEVICES="$gpus" "$py" - "$count" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Expected {expected} visible GPUs, found {torch.cuda.device_count()}")
for index in range(expected):
    free, capacity = torch.cuda.mem_get_info(index)
    print(f"GPU {index} {torch.cuda.get_device_name(index)} free {free / 2**30:.1f}/{capacity / 2**30:.1f} GiB")
PY
}

echo "TRACE-VL streaming vision $rev, GPUs $gpus, output $out"
if [[ -n "$resume" ]]; then
    module=trace_vl.streaming_vision.resume
    args=(--run "$out" --revision "$rev" --shards "$count")
    CUDA_VISIBLE_DEVICES="$gpus" "$py" -m "$module" verify "${args[@]}"
    eval_args=()
    log_prefix=eval-resume-shard
    report_files=(eval-recovery.json)
else
    model="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_4b}"
    train="${TRACE_TRAIN_MANIFEST:-$runroot/core-scale40k-54e161a-001/mixture/manifest.jsonl}"
    ovo="${TRACE_OVO_MANIFEST:-/nas-data/vlm/yyy/tracevl-data/core-transfer-v1-f1c19a7/ovo.jsonl}"
    streaming="${TRACE_STREAMING_ROOT-/nas-data/vlm/data/StreamingBench}"
    if [[ ! -f "$train" || ! -f "$ovo" || ! -f "$model/config.json" ]]; then
        printf 'Check local inputs:\n  LLaVA training %s\n  OVO development %s\n  model %s\n' \
            "$train" "$ovo" "$model" >&2
        exit 1
    fi
    module=trace_vl.streaming_vision.experiment
    args=(
        --out "$out" --model "$model" --train-manifest "$train" --ovo-manifest "$ovo"
        --gpus "$count" --train-clips "${TRACE_TRAIN_CLIPS:-512}" --epochs "${TRACE_EPOCHS:-2}"
        --workers "${TRACE_DECODE_WORKERS:-6}" --ovo-limit "${TRACE_OVO_LIMIT:-0}"
        --streaming-limit "${TRACE_STREAMING_LIMIT:-250}"
    )
    if [[ -n "$streaming" ]]; then
        args+=(--streaming-root "$streaming")
    fi
    echo "[1/4] CPU regression tests and fixed evaluation populations"
    CUDA_VISIBLE_DEVICES='' "$py" -m pytest -q tests/test_streaming_vision.py
    CUDA_VISIBLE_DEVICES='' "$py" -m "$module" prepare "${args[@]}" 2>&1 | tee "$out/logs/prepare.log"
    if [[ "${TRACE_PREPARE_ONLY:-0}" == 1 ]]; then
        exit 0
    fi
    gpu_check
    echo "[2/4] Multi-GPU visual feature distillation"
    CUDA_VISIBLE_DEVICES="$gpus" "$py" -m torch.distributed.run --standalone --nproc-per-node "$count" \
        --module "$module" train "${args[@]}" 2>&1 | tee -a "$out/logs/train.log"
    eval_args=(--workers "$TRACE_EVAL_DECODE_WORKERS")
    log_prefix=eval-shard
    report_files=()
fi

echo "[3/4] Sharded evaluation with native full-refresh parity on each active shard"
pids=()
for i in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES="${devices[i]}" "$py" -m "$module" eval "${args[@]}" "${eval_args[@]}" --shard "$i" \
        >>"$out/logs/$log_prefix-$i.log" 2>&1 &
    pids+=("$!")
done
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
        echo "Evaluation failed on shard $i. Inspect $out/logs/$log_prefix-$i.log" >&2
        tail -n 20 "$out/logs/$log_prefix-$i.log" >&2
        failed=1
    fi
done
(( failed == 0 ))

echo "[4/4] Paired accuracy, feature reconstruction and measured latency"
"$py" -m "$module" report "${args[@]}"
tar -czf "$out/reports.tmp" -C "$out" \
    REPORT.md report.json inputs.json prepared.json training.jsonl training-complete.json predictions logs "${report_files[@]}"
mv "$out/reports.tmp" "$out/reports.tar.gz"
echo "VERIFIED STREAMING VISION: $out/REPORT.md"
echo "REPORT BUNDLE: $out/reports.tar.gz"
