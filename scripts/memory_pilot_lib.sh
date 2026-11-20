# Shared shell logic of scripts/run_memory_pilot.sh (sourced; tested in tests/test_v3_memory_pilot.py).
# maybe_train: skip training when the completion/identity check passes (exit 0); otherwise keep any previous
# attempt as <dir>.incomplete-<time>, then train. The caller defines `completion_check` and `run_training`.
maybe_train() {
  local dir="$1"
  if completion_check; then
    echo "training complete and identical: skipped"
    return 0
  fi
  if [[ -e "$dir" ]]; then
    local stale="$dir.incomplete-$(date +%s%N)"
    mv "$dir" "$stale"
    echo "previous attempt kept at $stale"
  fi
  mkdir -p "$dir"
  run_training
}
