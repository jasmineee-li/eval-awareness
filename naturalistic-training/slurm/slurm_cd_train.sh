#!/bin/bash
#SBATCH --job-name=nat-cd-train
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# Context Distillation — Step 2: SFT on generated CD data
# Requires data/cd_train.jsonl from slurm_cd_datagen.sh

set -euo pipefail
source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/naturalistic-training

if [[ ! -f data/cd_train.jsonl ]]; then
    echo "ERROR: data/cd_train.jsonl not found. Run slurm_cd_datagen.sh first."
    exit 1
fi

echo "=== Starting context distillation SFT training ==="
python train.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/cd_train.jsonl \
    --eval-file data/cd_val.jsonl \
    --output-dir checkpoints/qwen3-cd \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-cd \
    --seed 42

echo "=== Context distillation training complete ==="
echo "Checkpoint: checkpoints/qwen3-cd/final/"
