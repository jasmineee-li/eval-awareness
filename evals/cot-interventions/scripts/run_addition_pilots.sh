#!/bin/bash
#SBATCH --job-name=cot-addition
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=8:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/cot-interventions/logs/addition-pilots-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/cot-interventions/logs/addition-pilots-%j.err

# Run 50-seed addition on QwQ-32B checkpoints sequentially.
# Per seed: 8 eval_aware sentences × 3 resamples + 8 control sentences × 3 resamples + 5 baseline = 53
# Total per checkpoint: 50 × 53 = 2,650 continuations (+ GPT judging)

set -uo pipefail  # no -e: let loop continue if one checkpoint fails

PROJECT_DIR="/data/jasmine_li/eval-awareness/cot-interventions"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1

source /data/jasmine_li/miniconda3/bin/activate rllm

if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

CHECKPOINTS=("QwQ-32B" "QwQ-32B-Coop")
FAILED=()

for CKPT in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Starting: $CKPT at $(date)"
    echo "============================================================"

    if python -m src.runner \
        --config config.yaml \
        --checkpoint "$CKPT" \
        --experiment addition \
        --seed 42; then
        echo ""
        echo "$CKPT complete at $(date)"
    else
        echo ""
        echo "ERROR: $CKPT failed (exit code $?) at $(date). Continuing to next checkpoint..."
        FAILED+=("$CKPT")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All addition pilots complete at $(date)"
else
    echo "Addition pilots finished at $(date) with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
