#!/bin/bash
# llama-3.3-70b only: summarization FIRST, then intent.
#
# Rationale: the 70B is capped at 1,000 Groq requests/day (~90s/call), so its
# 2,000 intent calls take ~2 days but its 100 summarization calls take only
# ~2.5h. Running summarization first means the blinded expert rating sheet can
# be generated ONCE with all five models in it, rather than in two batches -
# rating-protocol cleanliness matters more than getting the charts complete
# early. Intent then grinds afterwards for the final tables.
#
# The ~10 intent calls already made before the reorder are cached and will not
# be repeated.
set -u
cd "$(dirname "$0")"
M="groq/llama-3.3-70b-versatile"
LOG="results/run_logs/groq_llama-3.3-70b-versatile.log"
mkdir -p results/run_logs

{
  echo "=== REORDERED: summ first, then intent @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ./venv/bin/python src/run_summ.py --models "$M"
  echo "=== summ exit=$? @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ./venv/bin/python src/run_intent.py --models "$M"
  echo "=== intent exit=$? @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "=== DONE $M @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} >> "$LOG" 2>&1
