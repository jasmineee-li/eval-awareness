#!/bin/bash
# Phase 2: UltraFeedback SFT on Qwen3-32B (instruction following, dosage-matched)
# Trains on HuggingFaceH4/ultrafeedback_binarized chosen-only, subsampled to 7.5K
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Prepare data
echo "=== Preparing UltraFeedback SFT data (7.5K dosage-matched) ==="
python prepare_data.py ultrafeedback_sft \
    --max-examples 7500 \
    --output-dir data \
    --seed 42

# Step 2: Train
echo "=== Starting UltraFeedback SFT training ==="
python train.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/ultrafeedback_sft_train.jsonl \
    --eval-file data/ultrafeedback_sft_val.jsonl \
    --output-dir checkpoints/qwen3-ultrafeedback-sft \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-ultrafeedback-sft \
    --seed 42

echo "=== UltraFeedback SFT complete ==="
echo "Checkpoint: checkpoints/qwen3-ultrafeedback-sft/final/"
