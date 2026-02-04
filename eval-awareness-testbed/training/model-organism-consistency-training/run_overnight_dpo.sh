#!/bin/bash
# Overnight DPO Training Pipeline
#
# Steps:
#   1. Run Phase 4: prefixed inference on train set (vLLM)
#   2. Create DPO dataset (pairs where responses differ)
#   3. Run DPO training
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   nohup bash run_overnight_dpo.sh > dpo_training.log 2>&1 &

set -e

echo "======================================"
echo "DPO Training Pipeline"
echo "======================================"
echo "Started at: $(date)"
echo ""

# HuggingFace cache
export HF_HOME="/workspace/.cache/huggingface"
mkdir -p $HF_HOME

# Config
MODEL_ID="Qwen/QwQ-32B"
VLLM_PORT=8000

# Check GPUs
echo "GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# Verify files
echo "Verifying files..."
for f in data/eval_awareness/notaware_data_8k.json \
         data/eval_awareness/initial_responses_8k.json \
         data/eval_awareness/train_notaware_ids.json; do
    [ -f "$f" ] || { echo "ERROR: Missing $f"; exit 1; }
done
echo "  All files present ✓"
echo ""

# Wandb
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set"
    WANDB_FLAG="--no-wandb"
else
    echo "Wandb API key found ✓"
    WANDB_FLAG=""
    wandb login --relogin "$WANDB_API_KEY" 2>/dev/null || true
fi
echo ""

# ============================================================
# PHASE 4: Prefixed Inference on Train Set
# ============================================================
echo "======================================"
echo "Phase 4: Prefixed Inference (Train Set)"
echo "======================================"

PREFIXED_OUTPUT="data/inference_results/prefixed_responses_train.json"

if [ -f "$PREFIXED_OUTPUT" ]; then
    EXISTING=$(python3 -c "import json; print(len(json.load(open('$PREFIXED_OUTPUT'))))")
    NEEDED=$(python3 -c "import json; print(len(json.load(open('data/eval_awareness/train_notaware_ids.json'))))")
    if [ "$EXISTING" -ge "$NEEDED" ]; then
        echo "Prefixed responses already exist ($EXISTING samples). Skipping Phase 4."
    else
        echo "Partial results found ($EXISTING/$NEEDED). Will resume."
    fi
fi

if [ ! -f "$PREFIXED_OUTPUT" ] || [ "$EXISTING" -lt "$NEEDED" ]; then
    # Start vLLM
    echo "Starting vLLM server..."
    pkill -f "vllm.entrypoints.openai.api_server.*--port $VLLM_PORT" 2>/dev/null || true
    sleep 2

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --served-model-name default \
        --tensor-parallel-size $NUM_GPUS \
        --port $VLLM_PORT \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code \
        --disable-log-requests \
        > vllm_phase4.log 2>&1 &
    VLLM_PID=$!

    # Wait for server
    echo "Waiting for vLLM..."
    for i in $(seq 1 120); do
        curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1 && break
        [ $i -eq 120 ] && { echo "vLLM failed to start"; tail -50 vllm_phase4.log; exit 1; }
        sleep 5
    done
    echo "vLLM ready!"

    # Run inference
    python scripts/04_run_prefixed_inference_vllm.py \
        --data-file data/eval_awareness/notaware_data_8k.json \
        --ids-file data/eval_awareness/train_notaware_ids.json \
        --output "$PREFIXED_OUTPUT" \
        --vllm-url "http://localhost:$VLLM_PORT/v1" \
        --concurrency 32

    # Stop vLLM
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "vLLM stopped."
fi
echo ""

# ============================================================
# Create DPO Dataset
# ============================================================
echo "======================================"
echo "Creating DPO Dataset"
echo "======================================"

python scripts/create_dpo_dataset.py \
    --non-prefixed data/eval_awareness/initial_responses_8k.json \
    --prefixed "$PREFIXED_OUTPUT" \
    --ids-file data/eval_awareness/train_notaware_ids.json \
    --output data/training_datasets/dpo_train.jsonl

DPO_COUNT=$(wc -l < data/training_datasets/dpo_train.jsonl)
echo "DPO pairs: $DPO_COUNT"
echo ""

# ============================================================
# DPO Training
# ============================================================
echo "======================================"
echo "DPO Training"
echo "======================================"

python scripts/run_dpo_training.py \
    --model-name "$MODEL_ID" \
    --train-file data/training_datasets/dpo_train.jsonl \
    --output-dir data/checkpoints/qwq-32b-consistency-dpo \
    --lora-r 64 \
    --lora-alpha 128 \
    --epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 5e-6 \
    --beta 0.1 \
    $WANDB_FLAG

echo ""
echo "======================================"
echo "DPO Pipeline Complete!"
echo "======================================"
echo "Finished at: $(date)"
echo ""
echo "Outputs:"
echo "  DPO dataset: data/training_datasets/dpo_train.jsonl"
echo "  LoRA adapter: data/checkpoints/qwq-32b-consistency-dpo/final"
echo ""
echo "Next: Run evaluation with the DPO model"
