#!/bin/bash
#SBATCH --job-name=nat-cd-nothink
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# Context Distillation SFT — think blocks STRIPPED from training data

set -euo pipefail
source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/naturalistic-training

echo "=== Starting CD-nothink SFT training ==="
python train.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/cd_nothink_train.jsonl \
    --eval-file data/cd_nothink_val.jsonl \
    --output-dir checkpoints/qwen3-cd-nothink \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-cd-nothink \
    --seed 42

echo "=== CD-nothink training complete ==="
echo "Checkpoint: checkpoints/qwen3-cd-nothink/final/"
