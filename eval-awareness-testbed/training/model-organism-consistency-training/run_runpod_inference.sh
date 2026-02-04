#!/bin/bash
# RunPod inference script for QwQ-32B
# 
# Recommended pod: 4x A100-40GB or 2x A100-80GB
# 
# Usage:
#   1. Create RunPod with 4x A100-40GB (or 2x A100-80GB)
#   2. Clone repo or upload files
#   3. Run: bash run_runpod_inference.sh

set -e

echo "======================================"
echo "QwQ-32B Local Inference (RunPod)"
echo "======================================"

# Check GPUs
echo "Available GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Configuration
MODEL_ID="Qwen/QwQ-32B"
VLLM_PORT=8000
MAX_TOKENS=1024
CONCURRENCY=32
TP_SIZE=$NUM_GPUS  # Use all available GPUs

# Install dependencies if needed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm aiohttp tqdm
fi

# Download model if not cached
echo "Checking model cache..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')" || true

# Start vLLM server in background
echo "Starting vLLM server with tensor_parallel_size=$TP_SIZE..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name default \
    --tensor-parallel-size $TP_SIZE \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server
echo "Waiting for vLLM server to be ready..."
for i in {1..180}; do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: vLLM server failed to start after 15 minutes"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo -n "."
    sleep 5
done

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
    --max-tokens $MAX_TOKENS \
    --concurrency $CONCURRENCY

echo ""
echo "======================================"
echo "Inference complete!"
echo "Results saved to: data/inference_results/initial_responses.json"
echo "======================================"

# Keep server running or shut down
read -p "Shut down vLLM server? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kill $VLLM_PID 2>/dev/null
    echo "Server stopped."
fi
