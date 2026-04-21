#!/bin/bash
# v2a cooperation-prompt ablation: GLM-5, GLM-5.1, Grok-3 on AM (bare, SE, AF).
# Runs on the login node (API-driven, no local GPU).
#
# Usage:
#   nohup bash evals/agentic_misalignment/scripts/run_v2a_frontier.sh > evals/run_2026-04-21_v2a_frontier.log 2>&1 &
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
echo "v2a frontier rerun: $(date)"
echo "Models: GLM-5, GLM-5.1, Grok-3"
echo "Warnings: bare, safety_eval, af"
echo "Epochs per cell: 30 (=> 810 samples per cell, 9 cells, 7290 total)"
echo "Suffix: sdf/deliberative-alignment/coop_prompt_specs_v2a.txt"
echo "=========================================="

: "${CHUNK_ID:=v2a_frontier_20260421}"
python evals/agentic_misalignment/scripts/run_v2a_frontier.py --chunk-id "$CHUNK_ID"

echo ""
echo "=========================================="
echo "Run complete: $(date)"
echo "=========================================="
