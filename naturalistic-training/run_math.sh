#!/bin/bash
# Phase 1: Math SFT on Qwen3-32B (negative control)
# Trains on GSM8K — domain-irrelevant SFT baseline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Prepare data
echo "=== Preparing math (GSM8K) data ==="
python prepare_data.py math \
    --output-dir data \
    --seed 42

# Step 2: Train
echo "=== Starting math SFT training ==="
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
    --seed 42

echo "=== Math SFT complete ==="
echo "Checkpoint: checkpoints/qwen3-math-sft/final/"
