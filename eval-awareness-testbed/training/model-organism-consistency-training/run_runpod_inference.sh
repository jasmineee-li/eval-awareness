#!/bin/bash
# RunPod inference script for QwQ-32B
#
# Recommended pod: 4x A100-40GB or 2x A100-80GB
#
# Usage:
#   1. Create RunPod with 4x A100-40GB (or 2x A100-80GB)
#   2. Clone repo or upload files
#   3. Run: bash run_runpod_inference.sh
#   4. For nohup: nohup bash run_runpod_inference.sh > inference.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /workspace/venv/bin/activate

# Use /workspace for pip cache (root filesystem is small)
export PIP_CACHE_DIR=/workspace/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

# Configuration
MODEL_ID="Qwen/QwQ-32B"
VLLM_PORT=8000
MAX_TOKENS=1024
CONCURRENCY=32
VLLM_LOG="vllm_server.log"
VLLM_PID=""

# Cleanup function - kills vLLM server on exit
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Shutting down vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "Server stopped."
    fi
}
trap cleanup EXIT

echo "======================================"
echo "QwQ-32B Local Inference (RunPod)"
echo "======================================"

# Check GPUs
echo "Available GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

TP_SIZE=$NUM_GPUS  # Use all available GPUs

# Install dependencies if needed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm aiohttp tqdm
fi

# Download model if not cached
echo "Checking model cache..."
if ! python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"; then
    echo "ERROR: Failed to download model. Check your network and HF_TOKEN."
    exit 1
fi

# Start vLLM server in background
echo "Starting vLLM server with tensor_parallel_size=$TP_SIZE..."
echo "Server logs: $VLLM_LOG"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name default \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$VLLM_PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-log-requests \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
for i in {1..180}; do
    # Check if server process is still alive
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: vLLM server died unexpectedly. Check $VLLM_LOG for details."
        tail -50 "$VLLM_LOG"
        exit 1
    fi

    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo ""
        echo "vLLM server is ready!"
        break
    fi

    if [ "$i" -eq 180 ]; then
        echo ""
        echo "ERROR: vLLM server failed to start after 15 minutes"
        tail -50 "$VLLM_LOG"
        exit 1
    fi

    # Progress indicator (works in both interactive and non-interactive)
    if [ $((i % 12)) -eq 0 ]; then
        echo "Still waiting... ($((i * 5))s elapsed)"
    fi
    sleep 5
done

# Verify server is still alive before inference
if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM server died after health check passed"
    exit 1
fi

# Create data directory if needed
mkdir -p data/inference_results

# Run inference
echo ""
echo "======================================"
echo "Running inference..."
echo "======================================"

python scripts/run_local_vllm_inference.py \
    --splits-dir data/splits \
    --output data/inference_results/initial_responses.json \
    --vllm-url "http://localhost:$VLLM_PORT/v1" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY"

echo ""
echo "======================================"
echo "Inference complete!"
echo "Results saved to: data/inference_results/initial_responses.json"
echo "======================================"

# Server will be shut down by cleanup trap on exit
