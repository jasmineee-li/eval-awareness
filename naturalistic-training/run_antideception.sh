#!/bin/bash
# Phase 1: Anti-deception SFT on Qwen3-32B
# Trains on Anthropic honesty-elicitation data (Goals + Follow-up mix)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Prepare data
echo "=== Preparing anti-deception data ==="
python prepare_data.py antideception \
    --elicitation-zip /workspace/eval-awareness/elicitation.zip \
    --output-dir data \
    --seed 42

# Step 2: Train
echo "=== Starting anti-deception SFT training ==="
python train.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/antideception_train.jsonl \
    --eval-file data/antideception_val.jsonl \
    --output-dir checkpoints/qwen3-antideception-sft \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-antideception-sft \
    --seed 42

echo "=== Anti-deception SFT complete ==="
echo "Checkpoint: checkpoints/qwen3-antideception-sft/final/"
