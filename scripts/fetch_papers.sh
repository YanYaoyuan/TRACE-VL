#!/usr/bin/env bash
# Download the surveyed arXiv PDFs into papers/ (git-ignored; arXiv licenses do not allow redistribution).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p papers
grep -v '^#' docs/v3/papers.txt | while read -r id name; do
  [ -z "${id:-}" ] && continue
  out="papers/${id}_${name}.pdf"
  [ -s "$out" ] && continue
  curl -sfL "https://arxiv.org/pdf/${id}" -o "$out" && echo "ok   $out" || echo "fail $id"
  sleep 3  # be polite to arXiv
done
