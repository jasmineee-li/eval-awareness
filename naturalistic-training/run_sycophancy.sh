#!/bin/bash
# Phase 1: Anti-sycophancy SFT on OLMo 3 32B Think
# Trains on google/sycophancy-intervention synthetic data (~10K, dosage-matched)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Prepare data (clones google repo, runs pipeline, subsamples to 10K)
echo "=== Preparing anti-sycophancy data ==="
python prepare_data.py sycophancy \
    --max-examples 10000 \
    --output-dir data \
    --seed 42

# Step 2: Train
echo "=== Starting anti-sycophancy SFT training ==="
python train.py \
    --model-name allenai/OLMo-3-32B-Think \
    --train-file data/sycophancy_train.jsonl \
    --eval-file data/sycophancy_val.jsonl \
    --output-dir checkpoints/olmo3-sycophancy-sft \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --wandb-project naturalistic-training \
    --wandb-run-name olmo3-sycophancy-sft \
    --seed 42

echo "=== Anti-sycophancy SFT complete ==="
echo "Checkpoint: checkpoints/olmo3-sycophancy-sft/final/"
