#!/bin/bash
# v2a top-up: 25 epochs × 3 models (GLM-5, GLM-5.1, Grok-3) × bare, to pool
# with the existing 5-epoch v2a data at n=810 per cell.
#
# Usage:
#   nohup bash evals/agentic_misalignment/scripts/run_v2a_topup.sh > evals/run_2026-04-22_v2a_topup.log 2>&1 &
#
# Resumable: re-run with the same CHUNK_ID to pick up after a failure.

set -uo pipefail

cd /data/jasmine_li/eval-awareness
source .venv/bin/activate

if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY missing from .env" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="/data/jasmine_li/eval-awareness/evals/logs"

echo "=========================================="
echo "v2a top-up: $(date)"
echo "Models: GLM-5, GLM-5.1, Grok-3"
echo "Warning: bare only"
echo "Epochs per cell: 25 (pools to 30 with existing 5-ep data)"
echo "Suffix: sdf/deliberative-alignment/coop_prompt_specs_v2a.txt"
echo "=========================================="

: "${CHUNK_ID:=v2a_frontier_topup_glm_grok_20260422}"
python evals/agentic_misalignment/scripts/run_v2a_topup.py --chunk-id "$CHUNK_ID"

echo ""
echo "=========================================="
echo "Top-up complete: $(date)"
echo "=========================================="
