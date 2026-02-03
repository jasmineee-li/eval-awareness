#!/bin/bash
#SBATCH --job-name=train-masked
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/train-masked-%j.out

# WANDB_API_KEY should be set in environment or .env file

cd /data/jasmine_li/eval-awareness/selective-generalization

echo "=========================================="
echo "Training with proper prompt masking"
echo "=========================================="

# Train
python src/train.py --model llama31_8b --wandb_project selective-gen --wandb_run_name "llama31_8b-masked-$(date +%Y%m%d-%H%M%S)"

# Evaluate
echo ""
echo "Evaluating finetuned model..."
python src/evaluate.py evaluate --model llama31_8b

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
