#!/usr/bin/env bash
# P1 step 1: 100-sample annotation feasibility check (docs/event_state/causal_label_protocol.md). CPU ONLY.
# No model, no GPU, no training. Reads UCA JSON, the official UCF-Crime test list and the raw videos; writes only
# into TRACE_OUT.
#   TRACE_STEP=sample   draw the review items and blank sheets; export prefix clips ending at the raw start times
#   TRACE_STEP=prefix   after annotators filled `candidate_ready`: export prefix clips for those candidate times
#   TRACE_STEP=report   validate the filled sheets and write feasibility-report.json
set -euo pipefail
: "${TRACE_STEP:?Set TRACE_STEP=sample, prefix or report}" "${TRACE_OUT:?Set TRACE_OUT}"
export TRACE_REPO="${TRACE_REPO:-/data/yyy/TRACE-VL}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TRACE_REPO/.venv}"
uv_bin="${TRACE_UV_BIN:-uv}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
tool=("$uv_bin" run --locked --no-sync python -m trace_vl.event_state.annotation)
mkdir -p "$TRACE_OUT"
case "$TRACE_STEP" in
  sample)
    : "${TRACE_UCA_DIR:?directory with UCFCrime_{Train,Val,Test}.json}" "${TRACE_UCF_TEST_LIST:?official Anomaly_Test.txt}" \
      "${TRACE_UCF_VIDEOS:?root of the raw UCF-Crime videos}"
    [[ ! -e "$TRACE_OUT/items.jsonl" ]] || { echo "$TRACE_OUT already holds a sample; use a new TRACE_OUT" >&2; exit 2; }
    "${tool[@]}" sample --uca-dir "$TRACE_UCA_DIR" --official-test-list "$TRACE_UCF_TEST_LIST" --output "$TRACE_OUT"
    "${tool[@]}" export-prefix --records "$TRACE_OUT/sheet-annotator-1.jsonl" --video-root "$TRACE_UCF_VIDEOS" \
      --output "$TRACE_OUT/clips-raw-start"
    echo "Annotators fill sheet-annotator-1.jsonl (all items) and sheet-annotator-2.jsonl (double-annotated items)."
    ;;
  prefix)
    : "${TRACE_SHEET:?filled sheet with candidate_ready}" "${TRACE_UCF_VIDEOS:?root of the raw UCF-Crime videos}"
    "${tool[@]}" export-prefix --records "$TRACE_SHEET" --video-root "$TRACE_UCF_VIDEOS" \
      --output "$TRACE_OUT/clips-$(basename "$TRACE_SHEET" .jsonl)"
    ;;
  report)
    : "${TRACE_SHEETS:?space-separated filled sheets}"
    # shellcheck disable=SC2086
    "${tool[@]}" validate --records $TRACE_SHEETS || echo "Some records are invalid; they are listed in the report." >&2
    # shellcheck disable=SC2086
    "${tool[@]}" report --records $TRACE_SHEETS --output "$TRACE_OUT/feasibility-report.json"
    ;;
  *) echo "TRACE_STEP must be sample, prefix or report" >&2; exit 2 ;;
esac
