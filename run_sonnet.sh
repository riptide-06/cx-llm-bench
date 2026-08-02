#!/bin/bash
# claude-sonnet-5: ZERO-SHOT intent only, plus the full summarization set.
#
# Few-shot is deliberately excluded. Sonnet 5 uses the post-4.7 tokenizer
# (~34% more tokens for identical text - measured 895 vs Haiku's 670 on the
# same intent prompt), so the full two-condition workload prices at ~$4.27
# against ~$3.38 of remaining Anthropic credit: it would run out partway.
# Zero-shot + summarization comes to ~$2.03, inside the approved ~$2.50.
set -u
cd "$(dirname "$0")"
M="claude-sonnet-5"
LOG="results/run_logs/claude-sonnet-5.log"
mkdir -p results/run_logs

{
  echo "=== START $M (zero_shot only + summ) @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ./venv/bin/python src/run_intent.py --models "$M" --conditions zero_shot
  echo "=== intent exit=$? @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ./venv/bin/python src/run_summ.py --models "$M"
  echo "=== summ exit=$? @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "=== DONE $M @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} >> "$LOG" 2>&1
