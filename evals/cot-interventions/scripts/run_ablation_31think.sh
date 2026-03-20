#!/bin/bash
#SBATCH --job-name=cot-ablation-31think
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=6:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/cot-interventions/logs/ablation-31think-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/cot-interventions/logs/ablation-31think-%j.err

set -euo pipefail

PROJECT_DIR="/data/jasmine_li/eval-awareness/cot-interventions"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1

# Activate conda env with vLLM
source /data/jasmine_li/miniconda3/bin/activate rllm

# Load API keys for GPT judge
if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

echo "Starting ablation experiment on 3.1-Think"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo ""

python -m src.runner \
    --config config.yaml \
    --checkpoint 3.1-Think \
    --experiment ablation \
    --seed 42

echo ""
echo "Completed at $(date)"
