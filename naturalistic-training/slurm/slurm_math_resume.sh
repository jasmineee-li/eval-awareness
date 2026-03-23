#!/bin/bash
#SBATCH --job-name=nat-math
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# Math SFT — resume from checkpoint-1000 (checkpoint-1200 corrupted by ENOSPC)

set -euo pipefail
source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/naturalistic-training

echo "=== Resuming math SFT training from checkpoint-1000 ==="
python train.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/math_train.jsonl \
    --eval-file data/math_val.jsonl \
    --output-dir checkpoints/qwen3-math-sft \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-math-sft \
    --resume-from-checkpoint checkpoints/qwen3-math-sft/checkpoint-1000 \
    --seed 42

echo "=== Math SFT complete ==="
echo "Checkpoint: checkpoints/qwen3-math-sft/final/"
