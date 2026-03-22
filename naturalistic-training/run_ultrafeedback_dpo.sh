#!/bin/bash
# Phase 2: UltraFeedback DPO on Qwen3-32B (instruction following, preference learning)
# Trains on HuggingFaceH4/ultrafeedback_binarized preference pairs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Prepare data
echo "=== Preparing UltraFeedback DPO data ==="
python prepare_data.py ultrafeedback_dpo \
    --output-dir data \
    --seed 42

# Step 2: Train
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
