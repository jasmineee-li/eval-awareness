#!/bin/bash
#SBATCH --job-name=nat-cd-datagen
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Context Distillation — Step 1: Generate training data via vLLM server
# Starts vLLM serving Qwen3-32B (TP=4), generates 7.5K CD examples, then stops.
# Submit step 2 (slurm_cd_train.sh) after this completes.

set -euo pipefail
source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/naturalistic-training

VLLM_PID=""
cleanup() {
    echo "=== Stopping vLLM server ==="
    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Starting vLLM server (TP=4) ==="
vllm serve Qwen/Qwen3-32B --max-model-len 4096 --tensor-parallel-size 4 &
VLLM_PID=$!

# Wait for server to be ready (timeout 10 min)
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready after ~$((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died during startup"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within 10 minutes"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Generate CD training data
echo "=== Generating context distillation data ==="
python prepare_cd_data.py \
    --api-base http://localhost:8000/v1 \
    --model Qwen/Qwen3-32B \
    --num-examples 7500 \
    --temperature 0.7 \
    --max-tokens 2048 \
    --output-dir data \
    --seed 42

echo "=== CD data generation complete ==="
echo "Output: data/cd_train.jsonl, data/cd_val.jsonl"
