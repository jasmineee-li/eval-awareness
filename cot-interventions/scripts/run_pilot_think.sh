#!/bin/bash
#SBATCH --job-name=cot-ablation-think
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=3:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/cot-interventions/logs/ablation-think-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/cot-interventions/logs/ablation-think-%j.err

# Rerun 50-seed ablation pilot on Think only (failed in job 107998).
#
# Usage:
#   sbatch scripts/run_pilot_think.sh

set -euo pipefail

PROJECT_DIR="/data/jasmine_li/eval-awareness/cot-interventions"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1

source /data/jasmine_li/miniconda3/bin/activate rllm

if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

echo "Starting ablation pilot for Think"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

python -m src.runner \
    --config config.yaml \
    --checkpoint Think \
    --experiment ablation \
    --seed 42

echo ""
echo "Think complete at $(date)"
