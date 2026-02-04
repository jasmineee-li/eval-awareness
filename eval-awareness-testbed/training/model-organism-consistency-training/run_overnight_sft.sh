#!/bin/bash
# Overnight SFT Training Pipeline for QwQ-32B Consistency Training
#
# Prerequisites:
#   - RunPod with 4x H100 80GB (or similar high-VRAM setup)
#   - Data files in data/eval_awareness/:
#       notaware_data_8k.json
#       initial_responses_8k.json
#       train_notaware_ids.json
#       test_notaware_ids.json
#   - WANDB_API_KEY environment variable set
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   bash run_overnight_sft.sh
#
# Or run with nohup for overnight:
#   nohup bash run_overnight_sft.sh > sft_training.log 2>&1 &

set -e

echo "======================================"
echo "Overnight SFT Training Pipeline"
echo "======================================"
echo "Started at: $(date)"
echo ""

# Redirect HuggingFace cache to workspace (more disk space)
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p $HF_HOME
echo "HuggingFace cache: $HF_HOME"

# Configuration
MODEL_NAME="Qwen/QwQ-32B"
MAX_SEQ_LENGTH=8192
LORA_R=64
LORA_ALPHA=128
EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=2e-5
OUTPUT_DIR="data/checkpoints/qwq-32b-consistency-lora"
WANDB_PROJECT="qwq-consistency-training"

# Check GPUs
echo "Checking GPU configuration..."
nvidia-smi --query-gpu=name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo ""
echo "Detected $NUM_GPUS GPUs"
echo ""

# Check disk space
echo "Checking disk space..."
df -h /workspace | tail -1
WORKSPACE_FREE=$(df /workspace | tail -1 | awk '{print $4}')
echo "  /workspace free: $(df -h /workspace | tail -1 | awk '{print $4}')"
echo ""

# Verify required data files
echo "Verifying data files..."
MISSING_FILES=0
for f in "data/eval_awareness/notaware_data_8k.json" \
         "data/eval_awareness/initial_responses_8k.json" \
         "data/eval_awareness/train_notaware_ids.json" \
         "data/eval_awareness/test_notaware_ids.json"; do
    if [ ! -f "$f" ]; then
        echo "  ERROR: Missing $f"
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo "Please ensure all required data files are present."
    exit 1
fi
echo "  All data files present ✓"
echo ""

# Check wandb API key
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Training will run without wandb logging."
    echo "To enable wandb, run: export WANDB_API_KEY='your-key'"
    WANDB_FLAG="--no-wandb"
else
    echo "Wandb API key found ✓"
    WANDB_FLAG=""
    # Login to wandb
    wandb login --relogin "$WANDB_API_KEY" 2>/dev/null || true
fi
echo ""

# Install dependencies
echo "======================================"
echo "Installing dependencies..."
echo "======================================"
pip install -q wandb trl datasets transformers accelerate peft bitsandbytes 2>/dev/null || \
    pip install wandb trl datasets transformers accelerate peft bitsandbytes
echo "  Dependencies installed ✓"
echo ""

# Step 1: Create SFT datasets
echo "======================================"
echo "Step 1: Creating SFT datasets"
echo "======================================"
python scripts/create_sft_dataset_simple.py
echo ""

# Verify datasets were created
if [ ! -f "data/training_datasets/sft_train.jsonl" ]; then
    echo "ERROR: sft_train.jsonl not created"
    exit 1
fi
if [ ! -f "data/training_datasets/sft_test.jsonl" ]; then
    echo "ERROR: sft_test.jsonl not created"
    exit 1
fi

TRAIN_COUNT=$(wc -l < data/training_datasets/sft_train.jsonl)
TEST_COUNT=$(wc -l < data/training_datasets/sft_test.jsonl)
echo "  Train examples: $TRAIN_COUNT"
echo "  Test examples: $TEST_COUNT"
echo ""

# Step 2: Run SFT training
echo "======================================"
echo "Step 2: Running SFT training"
echo "======================================"
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Max seq length: $MAX_SEQ_LENGTH"
echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE x $GRAD_ACCUM grad accum"
echo "  Learning rate: $LR"
echo "  Output: $OUTPUT_DIR"
echo ""

python scripts/run_sft_trl.py \
    --model-name "$MODEL_NAME" \
    --train-file data/training_datasets/sft_train.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --max-seq-length $MAX_SEQ_LENGTH \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
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
echo "  SFT train dataset: data/training_datasets/sft_train.jsonl"
echo "  SFT test dataset: data/training_datasets/sft_test.jsonl"
echo "  LoRA adapter: $OUTPUT_DIR/final"
echo "  Merged model: $OUTPUT_DIR/merged"
echo ""
echo "Next steps:"
echo "  1. Run inference on test set with trained model"
echo "  2. Compare responses with/without eval prefix"
echo "  3. Evaluate consistency improvement"
