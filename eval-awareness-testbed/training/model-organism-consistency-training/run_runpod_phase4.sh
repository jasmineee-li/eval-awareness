#!/bin/bash
# RunPod Phase 4: Prefixed Inference for QwQ-32B
#
# Prerequisites:
#   - RunPod with 4x A100-40GB (or 2x A100-80GB, or 8x A10-24GB)
#   - Data files in data/eval_awareness/:
#       notaware_data_8k.json (8,311 examples)
#       train_notaware_ids.json (6,648 IDs)
#       test_notaware_ids.json (1,663 IDs)
#
# Usage:
#   bash run_runpod_phase4.sh
#
# The script will:
#   1. Download QwQ-32B model (with resume support)
#   2. Start vLLM server with tensor parallelism
#   3. Run Phase 4 prefixed inference on train set (6,648 examples)
#   4. Run Phase 4 prefixed inference on test set (1,663 examples)
#   5. Save results to data/inference_results/

set -e

echo "======================================"
echo "Phase 4: Prefixed Inference (RunPod)"
echo "======================================"
echo ""

# Configuration
MODEL_DIR="/workspace/models/QwQ-32B"
MODEL_ID="Qwen/QwQ-32B"
VLLM_PORT=8000
MAX_TOKENS=4096
TEMPERATURE=0.7
CONCURRENCY=32
CHECKPOINT_EVERY=500

# Check GPUs
echo "Checking GPU configuration..."
nvidia-smi --query-gpu=name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo ""
echo "Detected $NUM_GPUS GPUs - will use tensor_parallel_size=$NUM_GPUS"
echo ""

# Verify required data files exist
echo "Verifying data files..."
MISSING_FILES=0

DATA_FILE="data/eval_awareness/notaware_data_8k.json"
TRAIN_IDS="data/eval_awareness/train_notaware_ids.json"
TEST_IDS="data/eval_awareness/test_notaware_ids.json"

if [ ! -f "$DATA_FILE" ]; then
    echo "  ERROR: Missing $DATA_FILE"
    MISSING_FILES=1
fi

if [ ! -f "$TRAIN_IDS" ]; then
    echo "  ERROR: Missing $TRAIN_IDS"
    MISSING_FILES=1
fi

if [ ! -f "$TEST_IDS" ]; then
    echo "  ERROR: Missing $TEST_IDS"
    MISSING_FILES=1
fi

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "Please ensure all required data files are present."
    exit 1
fi

echo "  All data files present ✓"
echo ""

# Count samples
NUM_TRAIN=$(python3 -c "import json; print(len(json.load(open('$TRAIN_IDS'))))")
NUM_TEST=$(python3 -c "import json; print(len(json.load(open('$TEST_IDS'))))")
echo "Train samples: $NUM_TRAIN"
echo "Test samples: $NUM_TEST"
echo "Total: $((NUM_TRAIN + NUM_TEST))"
echo ""

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q vllm aiohttp tqdm 2>/dev/null || pip install vllm aiohttp tqdm
echo "  Dependencies installed ✓"
echo ""

# Download model with resume support
echo "======================================"
echo "Downloading QwQ-32B model..."
echo "======================================"
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "Model already downloaded at $MODEL_DIR"
else
    mkdir -p "$(dirname $MODEL_DIR)"
    huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR" --resume-download
fi
echo ""

# Create output directory
mkdir -p data/inference_results

# Start vLLM server
echo "======================================"
echo "Starting vLLM server..."
echo "======================================"
echo "  Model: $MODEL_DIR"
echo "  Tensor parallel size: $NUM_GPUS"
echo "  Port: $VLLM_PORT"
echo "  Max model length: 8192"
echo ""

# Kill any existing vLLM server on this port
pkill -f "vllm.entrypoints.openai.api_server.*--port $VLLM_PORT" 2>/dev/null || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name default \
    --tensor-parallel-size $NUM_GPUS \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests \
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
MAX_WAIT=300  # 5 minutes
WAITED=0
while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo ""
        echo "ERROR: vLLM server failed to start after ${MAX_WAIT}s"
        echo "Check vllm_server.log for details:"
        tail -50 vllm_server.log
        exit 1
    fi
    echo -n "."
    sleep 5
    WAITED=$((WAITED + 5))
done
echo ""
echo "vLLM server ready! (took ${WAITED}s)"
echo ""

# Common args
COMMON_ARGS="--data-file $DATA_FILE --vllm-url http://localhost:$VLLM_PORT/v1 --max-tokens $MAX_TOKENS --temperature $TEMPERATURE --concurrency $CONCURRENCY --checkpoint-every $CHECKPOINT_EVERY"

# Run Phase 4 on TRAIN set
echo "======================================"
echo "Running Phase 4: TRAIN set ($NUM_TRAIN examples)"
echo "======================================"

python scripts/04_run_prefixed_inference_vllm.py \
    $COMMON_ARGS \
    --ids-file "$TRAIN_IDS" \
    --output data/inference_results/prefixed_responses_train.json

echo ""

# Run Phase 4 on TEST set
echo "======================================"
echo "Running Phase 4: TEST set ($NUM_TEST examples)"
echo "======================================"

python scripts/04_run_prefixed_inference_vllm.py \
    $COMMON_ARGS \
    --ids-file "$TEST_IDS" \
    --output data/inference_results/prefixed_responses_test.json

echo ""
echo "======================================"
echo "Phase 4 Complete!"
echo "======================================"
echo ""

# Verify outputs
echo "Output files:"
for outfile in data/inference_results/prefixed_responses_train.json data/inference_results/prefixed_responses_test.json; do
    if [ -f "$outfile" ]; then
        NUM_RESULTS=$(python3 -c "import json; print(len(json.load(open('$outfile'))))")
        echo "  $outfile: $NUM_RESULTS responses"
    else
        echo "  $outfile: MISSING"
    fi
done

echo ""
echo "Next step: Copy results back and run Phase 5"
