#!/bin/bash
# Run the full benchmark for ONE model: intent (zero-shot + few-shot) then
# summarization, sequentially in a single process.
#
# Why one process per model: common.py throttles per model *within a process*.
# Two processes touching the same model would each keep their own clock and
# together exceed that model's free-tier request rate. Different models have
# independent quotas, so they are safe to run as parallel processes.
set -u
cd "$(dirname "$0")"
MODEL="$1"
SLUG="$(echo "$MODEL" | tr '/' '_')"
mkdir -p results/run_logs
LOG="results/run_logs/${SLUG}.log"

{
  echo "=== START $MODEL @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ./venv/bin/python src/run_intent.py --models "$MODEL"
  echo "=== intent exit=$? @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ./venv/bin/python src/run_summ.py --models "$MODEL"
  echo "=== summ exit=$? @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "=== DONE $MODEL @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} >> "$LOG" 2>&1
