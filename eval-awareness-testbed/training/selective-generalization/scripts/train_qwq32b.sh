#!/bin/bash
#SBATCH --job-name=train-qwq32b
#SBATCH --partition=cais
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/train-qwq32b-%j.out

# WANDB_API_KEY should be set in environment or .env file

cd /data/jasmine_li/eval-awareness/selective-generalization

echo "=========================================="
echo "Training QwQ-32B with prompt masking"
echo "=========================================="

# Train
python src/train.py --model qwq32b --wandb_project selective-gen --wandb_run_name "qwq32b-masked-$(date +%Y%m%d-%H%M%S)"

# Evaluate
echo ""
echo "Evaluating finetuned model..."
python src/evaluate.py evaluate --model qwq32b

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
