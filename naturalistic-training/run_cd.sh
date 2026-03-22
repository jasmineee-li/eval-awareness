#!/bin/bash
# Phase 2: Context Distillation on Qwen3-32B
# Step 1: Generate responses WITH system prompt via vLLM
# Step 2: SFT train WITHOUT system prompt (internalizes the context)
#
# PREREQUISITE: Start a vLLM server first:
#   vllm serve Qwen/Qwen3-32B --max-model-len 4096 --tensor-parallel-size 4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"

# Step 1: Generate CD training data
echo "=== Generating context distillation data ==="
echo "  Using vLLM API at: $VLLM_API_BASE"
python prepare_cd_data.py \
    --api-base "$VLLM_API_BASE" \
    --model Qwen/Qwen3-32B \
    --num-examples 7500 \
    --temperature 0.7 \
    --max-tokens 2048 \
    --output-dir data \
    --seed 42

# Step 2: Train (standard SFT on the generated data)
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

echo "=== Context distillation complete ==="
echo "Checkpoint: checkpoints/qwen3-cd/final/"
