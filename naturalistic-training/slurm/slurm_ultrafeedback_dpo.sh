#!/bin/bash
#SBATCH --job-name=nat-uf-dpo
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# UltraFeedback DPO — preference learning, 7.5K dosage-matched

set -euo pipefail
source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/naturalistic-training

echo "=== Preparing UltraFeedback DPO data ==="
python prepare_data.py ultrafeedback_dpo \
    --max-examples 7500 \
    --output-dir data \
    --seed 42

echo "=== Starting UltraFeedback DPO training ==="
python train_dpo.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/ultrafeedback_dpo_train.jsonl \
    --eval-file data/ultrafeedback_dpo_val.jsonl \
    --output-dir checkpoints/qwen3-ultrafeedback-dpo \
    --beta 0.1 \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 5e-7 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --max-prompt-length 2048 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-ultrafeedback-dpo \
    --seed 42

echo "=== UltraFeedback DPO complete ==="
echo "Checkpoint: checkpoints/qwen3-ultrafeedback-dpo/final/"
