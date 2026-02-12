#!/bin/bash
# ToolSafety QLoRA Training Pipeline for Qwen3-32B
#
# Prerequisites:
#   - 4x H100 80GB (or similar; 1x A100 80GB also works)
#   - WANDB_API_KEY environment variable set (optional)
#   - HuggingFace access to jinjinyien/ToolSafety and Qwen/Qwen3-32B
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   bash run_training.sh
#
# Or run with nohup for overnight:
#   nohup bash run_training.sh > toolsafety_training.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "ToolSafety QLoRA Training Pipeline"
echo "======================================"
echo "Started at: $(date)"
echo ""

# Redirect HuggingFace cache to workspace (more disk space)
if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.cache/huggingface
    export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
    mkdir -p $HF_HOME
    echo "HuggingFace cache: $HF_HOME"
fi

# Configuration
MODEL_NAME="Qwen/Qwen3-32B"
MAX_SEQ_LENGTH=4096
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.1
EPOCHS=2
BATCH_SIZE=2
GRAD_ACCUM=8
LR=1e-5
BENIGN_COUNT=7000
OUTPUT_DIR="checkpoints/qwen3-32b-toolsafety-lora"
WANDB_PROJECT="toolsafety-qwen3-32b"

# Check GPUs
echo "Checking GPU configuration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo ""
    echo "Detected $NUM_GPUS GPUs"
else
    echo "WARNING: nvidia-smi not found. Proceeding without GPU verification."
fi
echo ""

# Check wandb API key
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Training will run without wandb logging."
    echo "To enable wandb, run: export WANDB_API_KEY='your-key'"
    WANDB_FLAG="--no-wandb"
else
    echo "Wandb API key found"
    WANDB_FLAG=""
    wandb login --relogin "$WANDB_API_KEY" 2>/dev/null || true
fi
echo ""

# Install dependencies
echo "======================================"
echo "Installing dependencies..."
echo "======================================"
pip install -q wandb trl datasets transformers accelerate peft bitsandbytes huggingface_hub 2>/dev/null || \
    pip install wandb trl datasets transformers accelerate peft bitsandbytes huggingface_hub
echo "  Dependencies installed"
echo ""

# Step 1: Inspect dataset format (optional, for debugging)
# Uncomment the next line to just inspect without preparing data:
# python prepare_data.py --inspect-only && exit 0

# Step 2: Prepare data
echo "======================================"
echo "Step 1: Preparing data"
echo "======================================"
echo "  Safety data: jinjinyien/ToolSafety (~14k examples)"
echo "  Benign data: glaiveai/glaive-function-calling-v2 ($BENIGN_COUNT examples)"
echo ""

python prepare_data.py \
    --output-dir data/ \
    --benign-count $BENIGN_COUNT \
    --seed 42

echo ""

# Verify datasets were created
if [ ! -f "data/toolsafety_train.jsonl" ]; then
    echo "ERROR: toolsafety_train.jsonl not created"
    exit 1
fi

TRAIN_COUNT=$(wc -l < data/toolsafety_train.jsonl)
echo "  Training examples: $TRAIN_COUNT"
if [ -f "data/toolsafety_val.jsonl" ]; then
    VAL_COUNT=$(wc -l < data/toolsafety_val.jsonl)
    echo "  Validation examples: $VAL_COUNT"
fi
echo ""

# Step 3: Run QLoRA training
echo "======================================"
echo "Step 2: Running QLoRA training"
echo "======================================"
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Max seq length: $MAX_SEQ_LENGTH"
echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE x $GRAD_ACCUM grad accum = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "  Learning rate: $LR"
echo "  Output: $OUTPUT_DIR"
echo ""

python train.py \
    --model-name "$MODEL_NAME" \
    --train-file data/toolsafety_train.jsonl \
    --eval-file data/toolsafety_val.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --max-seq-length $MAX_SEQ_LENGTH \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --learning-rate $LR \
    --wandb-project "$WANDB_PROJECT" \
    $WANDB_FLAG

echo ""
echo "======================================"
echo "Training Pipeline Complete!"
echo "======================================"
echo "Finished at: $(date)"
echo ""
echo "Outputs:"
echo "  Training data: data/toolsafety_train.jsonl"
echo "  Validation data: data/toolsafety_val.jsonl"
echo "  Test data: data/toolsafety_test.jsonl"
echo "  Dataset stats: data/dataset_stats.json"
echo "  LoRA adapter: $OUTPUT_DIR/final"
echo "  Merged model: $OUTPUT_DIR/merged (if merge succeeded)"
echo ""
echo "Next steps:"
echo "  1. Run inference on test set: data/toolsafety_test.jsonl"
echo "  2. Measure safety refusal rate on harmful examples (target: >90%)"
echo "  3. Measure false refusal rate on benign examples (target: <5%)"
echo "  4. Compare general capabilities vs base Qwen3-32B"
