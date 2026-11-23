#!/usr/bin/env bash
# Prepare aligned video QA, test the evidence ceiling, then train the paired readers.
set -Eeuo pipefail

repo="${TRACE_REPO:-/data/yyy/TRACE-VL}"
py="${TRACE_PYTHON:-$repo/.venv/bin/python}"
cd "$repo"
rev=$(git rev-parse --verify "${TRACE_REV:-HEAD}^{commit}")
code_rev=$(git rev-parse --verify "${TRACE_HISTORY_CODE_REV:-$rev}^{commit}")
run_root=/nas-data/vlm/yyy/tracevl-runs
out="${TRACE_OUT:-$run_root/history-reader-${rev:0:7}-001}"
model="${TRACE_MODEL:-/nas-data/vlm/models/qwen3.5_4b}"
ovo="${TRACE_OVO_MANIFEST:-/nas-data/vlm/yyy/tracevl-data/core-transfer-v1-f1c19a7/ovo.jsonl}"
ovo_media="${TRACE_OVO_MEDIA:-/nas-data/vlm/data/OVO_Bench/src_videos}"
llava="${TRACE_LLAVA_ROOT:-/nas-data/vlm/llava}"
annotation="${TRACE_ACTIVITYNET_ANNOTATION:-/nas-data/vlm/yyy/tracevl-data/activitynet-captions/captions.zip}"
gpus="${TRACE_CUDA_DEVICES:-0,1,2,3}"
IFS=, read -r -a devices <<< "$gpus"
shards=${#devices[@]}

[[ -x "$py" && -f "$model/config.json" && -s "$ovo" && -d "$ovo_media" ]] || {
    printf 'Check local inputs: Python %s, Qwen %s, OVO %s, OVO videos %s\n' "$py" "$model" "$ovo" "$ovo_media" >&2
    exit 1
}
media=()
if [[ -n "${TRACE_ACTIVITYNET_MEDIA:-}" ]]; then
    media=("$TRACE_ACTIVITYNET_MEDIA")
else
    shopt -s nullglob
    for folder in "$llava/media" "$llava"/*_activitynetqa; do
        [[ ! -d "$folder" ]] || media+=("$folder")
    done
    shopt -u nullglob
fi
(( ${#media[@]} > 0 && shards > 0 ))
for folder in "${media[@]}"; do
    [[ -d "$folder" ]] || { echo "Missing ActivityNet directory: $folder" >&2; exit 1; }
done
mkdir -p "$out/logs" "$out/tmp"
out=$(realpath "$out")
exec 9>"$out/.launcher.lock"
flock -n 9 || { echo "Already running: $out" >&2; exit 2; }
trap 'code=$?; if (( code )); then echo "STOPPED ($code). See $out/logs and data-audit.json." >&2; fi' EXIT

source="$out/source"
if [[ "$code_rev" != "$rev" ]]; then
    source="$out/source-$code_rev"
fi
if [[ ! -f "$source/.revision" ]]; then
    staging=$(mktemp -d "$out/source.tmp.XXXXXX")
    git archive "$code_rev" | tar -xf - -C "$staging"
    printf '%s\n' "$code_rev" > "$staging/.revision"
    mv -T "$staging" "$source"
fi
[[ "$(cat "$source/.revision")" == "$code_rev" ]]
cd "$source"

export TRACE_REV="$rev" TRACE_HISTORY_CODE_REV="$code_rev" PYTHONPATH="$source/src"
export TMPDIR="$out/tmp" PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
module=trace_vl.history_reader.experiment
scratch="${TRACE_HISTORY_SCRATCH:-$repo/.cache/history-reader/$(basename "$out")}"
annotation_sha256=7ed4f4d613d965683fbdbb9f01baad4fe2a439b99fc39cfd2f6fa6cf9201fd2b

verify_annotation() {
    [[ -s "$1" ]] && printf '%s  %s\n' "$annotation_sha256" "$1" | sha256sum --status -c -
}

download_annotation() (
    mkdir -p "$(dirname "$annotation")" || exit 1
    exec 8>"$annotation.lock" || exit 1
    flock 8 || exit 1
    if [[ -s "$annotation" ]]; then
        verify_annotation "$annotation"
        exit $?
    fi
    local url=https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
    local partial="$annotation.part" attempt status bytes
    for attempt in 1 2 3; do
        if verify_annotation "$partial"; then
            mv -- "$partial" "$annotation"
            exit $?
        fi
        bytes=0
        if [[ -e "$partial" ]]; then
            bytes=$(stat -c '%s' -- "$partial") || exit 1
            if (( bytes >= 2690459 )); then
                rm -f -- "$partial" || exit 1
                bytes=0
            fi
        fi
        printf 'ActivityNet annotation download %s/3, resuming at %s bytes\n' "$attempt" "$bytes"
        status=0
        curl -fL --connect-timeout 15 --max-time 600 --continue-at - -o "$partial" "$url" || status=$?
        if verify_annotation "$partial"; then
            mv -- "$partial" "$annotation"
            exit $?
        fi
        if (( status == 33 )); then
            # A gateway rejected the range. Start a fresh transfer on the next attempt.
            rm -f -- "$partial" || exit 1
        fi
    done
    exit 1
)

echo "TRACE-VL history reader $rev; GPUs $gpus; output $out"
echo "Runtime code $code_rev"
echo "[1/4] Align ActivityNet Captions and lock the development populations"
if [[ ! -s "$annotation" ]] && ! download_annotation; then
    echo "ActivityNet download incomplete. Rerun to resume $annotation.part." >&2
    echo "A verified captions.zip can also be supplied with TRACE_ACTIVITYNET_ANNOTATION." >&2
    exit 1
fi
if ! verify_annotation "$annotation"; then
    echo "ActivityNet annotation missing or SHA-256 mismatch: $annotation" >&2
    echo "Supply a verified captions.zip with TRACE_ACTIVITYNET_ANNOTATION." >&2
    exit 1
fi
echo "ActivityNet annotation SHA-256 verified: $annotation"
CUDA_VISIBLE_DEVICES='' "$py" -m pytest -q tests/test_history_reader.py tests/test_core_vlm_data.py
CUDA_VISIBLE_DEVICES='' "$py" -m "$module" prepare --out "$out" --model "$model" \
    --annotation "$annotation" --media "${media[@]}" --ovo "$ovo" --ovo-media "$ovo_media" \
    --train-limit "${TRACE_HISTORY_TRAIN_LIMIT:-512}" --dev-limit "${TRACE_HISTORY_DEV_LIMIT:-192}" \
    --workers "${TRACE_HISTORY_PREP_WORKERS:-24}" --archive-workers "${TRACE_HISTORY_ARCHIVE_WORKERS:-4}" \
    2>&1 | tee -a "$out/logs/prepare.log"
if [[ "${TRACE_PREPARE_ONLY:-0}" == 1 ]]; then
    exit 0
fi

wait_jobs() {
    local failed=0 index
    for index in "${!pids[@]}"; do
        if ! wait "${pids[index]}"; then
            echo "Worker failed: ${logs[index]}" >&2
            tail -n 20 "${logs[index]}" >&2
            failed=1
        fi
    done
    return "$failed"
}

cache_split() {
    local split="$1" i
    pids=()
    logs=()
    for i in "${!devices[@]}"; do
        logs+=("$out/logs/cache-$split-$i.log")
        CUDA_VISIBLE_DEVICES="${devices[i]}" "$py" -m "$module" cache \
            --out "$out" --split "$split" --shard "$i" --shards "$shards" --scratch "$scratch" \
            --workers "${TRACE_HISTORY_DECODE_WORKERS:-3}" >>"${logs[i]}" 2>&1 &
        pids+=("$!")
    done
    wait_jobs
}

bundle() {
    tar -czf "$out/reports.tmp" -C "$out" inputs.json prepared.json data-audit.json logs "$@"
    mv "$out/reports.tmp" "$out/reports.tar.gz"
    echo "REPORT BUNDLE: $out/reports.tar.gz"
}

echo "[2/4] Measure the QA benefit of a known correct historical interval"
if ! CUDA_VISIBLE_DEVICES='' "$py" - "$out/prepared.json" <<'PY'
import json
import sys
from pathlib import Path

counts = json.loads(Path(sys.argv[1]).read_text())["counts"]
print("Aligned populations", counts)
raise SystemExit(0 if counts["dev"] >= 24 and counts["train"] and counts["ovo"] else 3)
PY
then
    bundle
    echo "STOPPED DATA: require at least 24 aligned dev videos and nonempty train and OVO populations."
    exit 3
fi
cache_split dev
if "$py" -m "$module" oracle --out "$out" 2>&1 | tee "$out/logs/oracle.log"; then
    echo "Evidence screen passed."
else
    status=${PIPESTATUS[0]}
    (( status == 3 )) || exit 1
    bundle oracle.json oracle-predictions.jsonl
    echo "VERIFIED HISTORY EVIDENCE SCREEN: gate not met. See $out/oracle.json"
    exit 0
fi

echo "[3/4] Cache train and OVO visual features on $shards GPUs"
cache_split train
cache_split ovo

echo "[4/4] Train the two readers at the same visual budget and report"
arms=(cosine learned)
pids=()
logs=()
for i in "${!arms[@]}"; do
    arm="${arms[i]}"
    logs+=("$out/logs/fit-$arm.log")
    CUDA_VISIBLE_DEVICES="${devices[i % shards]}" "$py" -m "$module" fit \
        --out "$out" --arm "$arm" >>"${logs[i]}" 2>&1 &
    pids+=("$!")
done
wait_jobs
CUDA_VISIBLE_DEVICES="${devices[0]}" "$py" -m "$module" report --out "$out"
bundle oracle.json oracle-predictions.jsonl report.json REPORT.md train-cosine.json train-learned.json \
    dev-predictions.jsonl ovo-predictions.jsonl
echo "VERIFIED HISTORY READER: $out/REPORT.md"
