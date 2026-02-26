#!/bin/bash
#SBATCH --job-name=cot-addition
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=6:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/cot-interventions/logs/addition-pilots-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/cot-interventions/logs/addition-pilots-%j.err

# Run 50-seed addition pilots on all 4 checkpoints sequentially.
# Per seed: 3 eval_aware sentences × 3 resamples + 3 control sentences × 3 resamples + 5 baseline = 23
# Total per checkpoint: 50 × 23 = 1,150 continuations (+ GPT judging)

set -euo pipefail

PROJECT_DIR="/data/jasmine_li/eval-awareness/cot-interventions"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1

source /data/jasmine_li/miniconda3/bin/activate rllm

if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

CHECKPOINTS=("Think-SFT" "Think-DPO" "Think" "3.1-Think")

for CKPT in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Starting: $CKPT at $(date)"
    echo "============================================================"

    python -m src.runner \
        --config config.yaml \
        --checkpoint "$CKPT" \
        --experiment addition \
        --seed 42

    echo ""
    echo "$CKPT complete at $(date)"
done

echo ""
echo "All addition pilots complete at $(date)"
