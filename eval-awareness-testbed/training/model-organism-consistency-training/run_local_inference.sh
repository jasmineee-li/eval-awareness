#!/bin/bash
#SBATCH --job-name=qwq_inference
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --partition=cais
#SBATCH --output=./qwq-rh-inference-step3.out
#SBATCH --error=./qwq-rh-inference-step3.err

# Exit on error
set -e

echo "======================================"
echo "Starting QwQ-32B Local Inference"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "======================================"

# Configuration - use exact path to avoid permission issues
MODEL_PATH="/data/huggingface/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"
VLLM_PORT=8000
MAX_TOKENS=1024
CONCURRENCY=64

# Set HuggingFace cache - use shared location for reading, local for writing
# The model already exists at /data/huggingface, but we need write access for locks
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
mkdir -p $HF_HOME/hub

# Use miniconda environment (has vLLM installed)
source /data/jasmine_li/miniconda3/bin/activate

echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

# Start vLLM server in background
echo "Starting vLLM server on port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name default \
    --tensor-parallel-size 4 \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in {1..120}; do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

# Run pipeline
echo "======================================"
echo "Running pipeline phases 3-5..."
echo "======================================"

cd /data/jasmine_li/eval-awareness/model-organism-consistency-training

VLLM_URL="http://localhost:$VLLM_PORT/v1"
RESPONSES_FILE="data/inference_results/initial_responses_30k_subset.json"

# Phase 3: Eval awareness detection (vLLM for QwQ + OpenRouter for grading)
echo ""
echo "======================================"
echo "PHASE 3: Eval Awareness Detection"
echo "======================================"
python scripts/03_detect_eval_awareness_vllm.py \
    --splits-dir data/splits \
    --responses "$RESPONSES_FILE" \
    --output data/eval_awareness/awareness_labels.json \
    --vllm-url "$VLLM_URL" \
    --grader-model "openai/gpt-5-mini" \
    --concurrency $CONCURRENCY

# Phase 4: Prefixed inference (vLLM)
echo ""
echo "======================================"
echo "PHASE 4: Prefixed Inference"
echo "======================================"
python scripts/04_run_prefixed_inference_vllm.py \
    --splits-dir data/splits \
    --awareness-labels data/eval_awareness/awareness_labels.json \
    --output data/inference_results/prefixed_responses.json \
    --vllm-url "$VLLM_URL" \
    --max-tokens $MAX_TOKENS \
    --concurrency $CONCURRENCY

# Phase 5: Dataset creation
echo ""
echo "======================================"
echo "PHASE 5: Dataset Creation"
echo "======================================"
python scripts/05_create_datasets.py \
    --splits-dir data/splits \
    --initial-responses "$RESPONSES_FILE" \
    --prefixed-responses data/inference_results/prefixed_responses.json \
    --awareness-labels data/eval_awareness/awareness_labels.json \
    --output-dir data/training_datasets

# Cleanup
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null || true

echo "======================================"
echo "Inference complete!"
echo "======================================"
