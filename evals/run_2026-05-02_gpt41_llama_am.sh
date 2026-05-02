#!/bin/bash
# AM 3-cond pilot for GPT-4.1, GPT-4.1-mini, Llama-3.3-70B-Instruct.
# Conditions: bare baseline, bare coop_v1, bare anticoop (strong variant).
# 5 epochs/cell -> n=135/cell, 9 cells total = 1215 transcripts.
# Plan: plans/2026-05-02_gpt41_llama_am_3cond.md
# Usage: nohup bash evals/run_2026-05-02_gpt41_llama_am.sh \
#          > evals/run_2026-05-02_gpt41_llama_am.log 2>&1 &

set -uo pipefail

cd /data/jasmine_li/eval-awareness
source .venv/bin/activate

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="/data/jasmine_li/eval-awareness/evals/logs"

RUNNER="evals/agentic_misalignment/scripts/run_bare_baseline_coop.py"
ANTICOOP_FILE="sdf/deliberative-alignment/new_anticoop_prompt_specs_20260423.txt"
EPOCHS=5
CONDITIONS="baseline,coop_v1,anticoop"

# (model_slug, chunk_id)
RUNS=(
    "openrouter/openai/gpt-4.1|gpt41_bare_3cond_20260502"
    "openrouter/openai/gpt-4.1-mini|gpt41mini_bare_3cond_20260502"
    "openrouter/meta-llama/llama-3.3-70b-instruct|llama33_70b_bare_3cond_20260502"
)

echo "=========================================="
echo "AM 3-cond pilot: GPT-4.1 / GPT-4.1-mini / Llama-3.3-70B"
echo "Start: $(date)"
echo "Epochs/cell: $EPOCHS  Conditions: $CONDITIONS"
echo "Anticoop file: $ANTICOOP_FILE"
echo "=========================================="

for entry in "${RUNS[@]}"; do
    MODEL="${entry%%|*}"
    CHUNK="${entry##*|}"

    echo ""
    echo "============================================================"
    echo "MODEL $MODEL  (chunk $CHUNK)  $(date)"
    echo "============================================================"

    if python "$RUNNER" \
        --model "$MODEL" \
        --chunk-id "$CHUNK" \
        --conditions "$CONDITIONS" \
        --epochs "$EPOCHS" \
        --anticoop-suffix-file "$ANTICOOP_FILE"; then
        echo "[OK] $MODEL done"
    else
        echo "[FAIL] $MODEL aborted; see state file run_bare_baseline_coop_${CHUNK}.json"
    fi
done

echo ""
echo "=========================================="
echo "AM 3-cond pilot complete: $(date)"
echo "=========================================="
