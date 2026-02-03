#!/bin/bash
#SBATCH --job-name=selective-gen
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm-%j.out

# Usage: sbatch scripts/run.sh [llama8b|qwq32b]

MODEL=${1:-llama8b}  # Default to llama8b

echo "=========================================="
echo "Selective Generalization Experiment"
echo "Model: $MODEL"
echo "=========================================="

cd /data/jasmine_li/eval-awareness/selective-generalization

# Ensure output directory exists
mkdir -p outputs

# 1. Prepare data
echo ""
echo "[Step 1/3] Preparing data..."
python src/data_prep.py

# 2. Train
echo ""
echo "[Step 2/3] Training model..."
python src/train.py --model $MODEL --wandb_project selective-gen --wandb_run_name "${MODEL}-$(date +%Y%m%d-%H%M%S)"

# 3. Evaluate
echo ""
echo "[Step 3/3] Evaluating model..."
python src/evaluate.py evaluate --model $MODEL

echo ""
echo "=========================================="
echo "Done! Results saved to outputs/$MODEL/"
echo "=========================================="
