#!/bin/bash
# Final Evaluation: Run prefixed inference on held-out test set
#
# Compares base QwQ-32B vs LoRA fine-tuned model for consistency.
#
# Prerequisites:
#   - RunPod with GPUs (4x A100-40GB or similar)
#   - Data files in data/eval_awareness/
#   - LoRA checkpoint in data/checkpoints/qwq-32b-consistency-lora/final/
#
# Usage:
#   bash run_final_eval.sh

set -e

# Use the HuggingFace cache on /workspace (where model is already downloaded)
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"

echo "======================================"
echo "Final Evaluation: Consistency Testing"
echo "======================================"
echo ""

# Configuration
MODEL_DIR="/workspace/models/QwQ-32B"
MODEL_ID="Qwen/QwQ-32B"
LORA_PATH="$(pwd)/data/checkpoints/qwq-32b-consistency-lora/final"
LORA_NAME="consistency"
VLLM_PORT=8000
MAX_TOKENS=1024
TEMPERATURE=0.7
CONCURRENCY=64
CHECKPOINT_EVERY=200

# Check GPUs
echo "Checking GPU configuration..."
nvidia-smi --query-gpu=name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo ""
echo "Detected $NUM_GPUS GPUs"
echo ""

# Verify required files
echo "Verifying required files..."
MISSING=0

if [ ! -f "data/eval_awareness/notaware_data_8k.json" ]; then
    echo "  ERROR: Missing data/eval_awareness/notaware_data_8k.json"
    MISSING=1
fi

if [ ! -f "data/eval_awareness/test_notaware_ids.json" ]; then
    echo "  ERROR: Missing data/eval_awareness/test_notaware_ids.json"
    MISSING=1
fi

if [ ! -f "data/eval_awareness/initial_responses_8k.json" ]; then
    echo "  ERROR: Missing data/eval_awareness/initial_responses_8k.json (non-prefixed baseline)"
    MISSING=1
fi

if [ ! -f "data/checkpoints/qwq-32b-consistency-lora/final/adapter_config.json" ]; then
    echo "  ERROR: Missing LoRA adapter at data/checkpoints/qwq-32b-consistency-lora/final"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Please ensure all required files are present."
    exit 1
fi

echo "  All required files present ✓"
echo ""

# Count test samples
NUM_TEST=$(python3 -c "import json; print(len(json.load(open('data/eval_awareness/test_notaware_ids.json'))))")
echo "Test samples: $NUM_TEST"
echo ""

# Create output directory
mkdir -p data/inference_results

# Check if local model exists, otherwise use HF model ID (vLLM will download)
echo "======================================"
echo "Checking QwQ-32B model..."
echo "======================================"

# Check for model weights (not just config)
if [ -d "$MODEL_DIR" ] && ls "$MODEL_DIR"/*.safetensors 1>/dev/null 2>&1; then
    echo "Model already downloaded at $MODEL_DIR"
    VLLM_MODEL="$MODEL_DIR"
else
    echo "Local model not found or incomplete."
    echo "Using HuggingFace model ID: $MODEL_ID"
    echo "(vLLM will download/cache automatically)"
    VLLM_MODEL="$MODEL_ID"
fi
echo ""

# Kill any existing vLLM server
pkill -f "vllm.entrypoints.openai.api_server.*--port $VLLM_PORT" 2>/dev/null || true
sleep 2

# Start vLLM server with LoRA support
echo "======================================"
echo "Starting vLLM server with LoRA support..."
echo "======================================"
echo "  Model: $VLLM_MODEL"
echo "  LoRA adapter: $LORA_PATH (name: $LORA_NAME)"
echo "  Tensor parallel: $NUM_GPUS GPUs"
echo "  Port: $VLLM_PORT"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --served-model-name default \
    --tensor-parallel-size $NUM_GPUS \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests \
    --enable-lora \
    --lora-modules "$LORA_NAME=$LORA_PATH" \
    --max-lora-rank 64 \
    > vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID)"
echo "Logs: vllm_server.log"

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600  # 10 minutes (LoRA loading takes longer)
WAITED=0
while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo ""
        echo "ERROR: vLLM server failed to start after ${MAX_WAIT}s"
        echo "Check vllm_server.log for details:"
        tail -100 vllm_server.log
        exit 1
    fi
    echo -n "."
    sleep 5
    WAITED=$((WAITED + 5))
done
echo ""
echo "vLLM server ready! (took ${WAITED}s)"
echo ""

# Show available models
echo "Available models:"
curl -s "http://localhost:$VLLM_PORT/v1/models" | python3 -c "import sys,json; models=json.load(sys.stdin)['data']; print('  ' + '\n  '.join(m['id'] for m in models))"
echo ""

# Common args
COMMON_ARGS="--data-file data/eval_awareness/notaware_data_8k.json --test-ids-file data/eval_awareness/test_notaware_ids.json --vllm-url http://localhost:$VLLM_PORT/v1 --max-tokens $MAX_TOKENS --temperature $TEMPERATURE --concurrency $CONCURRENCY --checkpoint-every $CHECKPOINT_EVERY"

# Run prefixed inference with BASE model
echo "======================================"
echo "Phase 1: Prefixed inference (BASE model)"
echo "======================================"
echo "Running $NUM_TEST test samples with 'This is an evaluation.' prefix"
echo ""

python scripts/run_final_eval_inference.py \
    $COMMON_ARGS \
    --model-name default \
    --output data/inference_results/prefixed_test_base.json

echo ""

# Run prefixed inference with LORA model
echo "======================================"
echo "Phase 2: Prefixed inference (LORA model)"
echo "======================================"
echo "Running $NUM_TEST test samples with 'This is an evaluation.' prefix"
echo ""

python scripts/run_final_eval_inference.py \
    $COMMON_ARGS \
    --model-name "$LORA_NAME" \
    --output data/inference_results/prefixed_test_lora.json

echo ""

# Summary
echo "======================================"
echo "Inference Complete!"
echo "======================================"
echo ""
echo "Output files:"
for outfile in data/inference_results/prefixed_test_base.json data/inference_results/prefixed_test_lora.json; do
    if [ -f "$outfile" ]; then
        NUM_RESULTS=$(python3 -c "import json; print(len(json.load(open('$outfile'))))")
        echo "  $outfile: $NUM_RESULTS responses"
    else
        echo "  $outfile: MISSING"
    fi
done

echo ""
echo "Baseline (non-prefixed) responses:"
echo "  data/eval_awareness/initial_responses_8k.json"
echo ""
echo "Next step: Run analysis to measure consistency"
echo "  python scripts/analyze_consistency.py"
