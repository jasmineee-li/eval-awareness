#!/bin/bash
# Run ablation experiment across all 4 OLMo checkpoints sequentially.
# Each checkpoint loads a different model into GPU, so they must run one at a time.
#
# Usage:
#   cd /data/jasmine_li/eval-awareness/cot-interventions
#   bash scripts/run_all_ablation.sh
#
# To resume from a specific checkpoint (e.g., skip Think-SFT if already done):
#   bash scripts/run_all_ablation.sh --start-from Think

set -uo pipefail  # no -e: let loop continue if one checkpoint fails

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="config.yaml"
SEED=42

# Parse --start-from flag
START_FROM=""
if [[ "${1:-}" == "--start-from" ]]; then
    START_FROM="${2:-}"
    echo "Will skip checkpoints until: $START_FROM"
fi

CHECKPOINTS=("Think-SFT" "Think-DPO" "Think" "3.1-Think")
STARTED=false
FAILED=()

if [[ -z "$START_FROM" ]]; then
    STARTED=true
fi

for CKPT in "${CHECKPOINTS[@]}"; do
    if [[ "$STARTED" == false ]]; then
        if [[ "$CKPT" == "$START_FROM" ]]; then
            STARTED=true
        else
            echo "Skipping $CKPT (--start-from $START_FROM)"
            continue
        fi
    fi

    echo ""
    echo "============================================================"
    echo "Starting ablation for checkpoint: $CKPT"
    echo "Time: $(date)"
    echo "============================================================"

    if python -m src.runner \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --experiment ablation \
        --seed "$SEED"; then
        echo ""
        echo "$CKPT ablation complete at $(date)"
    else
        echo ""
        echo "ERROR: $CKPT ablation failed (exit code $?) at $(date). Continuing to next checkpoint..."
        FAILED+=("$CKPT")
    fi
    echo ""
done

echo ""
echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All ablation experiments complete at $(date)"
else
    echo "Ablation finished at $(date) with ${#FAILED[@]} failure(s): ${FAILED[*]}"
fi
echo "Results in: results/"
echo "============================================================"
if [ ${#FAILED[@]} -gt 0 ]; then exit 1; fi
