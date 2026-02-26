#!/bin/bash
#SBATCH --job-name=cot-ablation-pilots
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=6:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/cot-interventions/logs/ablation-pilots-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/cot-interventions/logs/ablation-pilots-%j.err

# Run 50-seed ablation pilots on Think-SFT, Think-DPO, and Think.
# Each checkpoint loads its own model, generates 500+500 completions, judges, saves.
#
# Usage:
#   sbatch scripts/run_pilot_remaining.sh

set -euo pipefail

PROJECT_DIR="/data/jasmine_li/eval-awareness/cot-interventions"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1

source /data/jasmine_li/miniconda3/bin/activate rllm

if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

echo "Starting ablation pilots for Think-SFT, Think-DPO, Think"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

for CKPT in Think-SFT Think-DPO Think; do
    echo "============================================================"
    echo "Starting: $CKPT at $(date)"
    echo "============================================================"

    python -m src.runner \
        --config config.yaml \
        --checkpoint "$CKPT" \
        --experiment ablation \
        --seed 42

    echo ""
    echo "$CKPT complete at $(date)"
    echo ""
done

echo "============================================================"
echo "All pilots complete at $(date)"
echo "============================================================"
