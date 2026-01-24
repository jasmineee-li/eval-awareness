#!/bin/bash
#SBATCH --job-name=eval_wood_lora
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --partition=cais
#SBATCH --output=./eval-wood-lora-0120.out
#SBATCH --error=./eval-wood-lora-0120.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

# Upgrade anthropic to fix version issue
pip install --upgrade anthropic

export VLLM_BASE_URL=http://localhost:8002/v1
export VLLM_API_KEY=dummy

# Model configuration
BASE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"
LORA_PATH="/data/shared_cais/honesty_models/hub/models--timhua--wood_v2_sftr4_filt/snapshots/f46fa2a76794a8ec2538f4356e4088d938222144/"
LORA_NAME="wood_v2"
MODEL_NAME="timhua_wood_v2_sftr4_filt"

# Get base model path
CACHE_NAME=$(echo "$BASE_MODEL" | sed 's/\//--/g')
BASE_MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

if [ -z "$BASE_MODEL_PATH" ]; then
    echo "ERROR: Base model not found: $BASE_MODEL"
    exit 1
fi

echo "=========================================="
echo "Running Wood-v2 (LoRA adapter) evaluation"
echo "Base model: $BASE_MODEL"
echo "Base model path: $BASE_MODEL_PATH"
echo "LoRA adapter: $LORA_PATH"
echo "=========================================="

# Start vLLM server with LoRA support on port 8002
# Note: The LoRA adapter name must match what run.py expects (timhua/wood_v2_sftr4_filt)
vllm serve "$BASE_MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size 8 \
    --served-model-name "$BASE_MODEL" \
    --max-model-len 16384 \
    --enable-lora \
    --lora-modules "timhua_wood_v2_sftr4_filt=${LORA_PATH}" \
    --max-lora-rank 64 \
    --trust-remote-code &

VLLM_PID=$!

# Wait for server to start (larger model needs more time)
echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
sleep 300

# Check if server started successfully
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM server failed to start"
    echo "Check the error log for details"
    exit 1
fi

echo "vLLM server started successfully!"

# Run main evaluation (eval_mcq, purpose_mcq, purpose_open)
echo "Running main evaluation..."
python run.py --model "vllm/timhua_wood_v2_sftr4_filt" --limit 2500 --epochs 1

# Run CoT analysis
echo "Running eval_mcq_cot..."
python run.py --model "vllm/timhua_wood_v2_sftr4_filt" --stage eval_mcq_cot --limit 2500 --epochs 1

echo "Running purpose_open_cot..."
python run.py --model "vllm/timhua_wood_v2_sftr4_filt" --stage purpose_open_cot --limit 2500 --epochs 1

# Stop vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Wood-v2 evaluation complete!"
echo "=========================================="

# Generate score table
export MODELS_TO_PLOT="timhua_wood_v2_sftr4_filt"
python -m analysis.score_table eval_mcq

echo "Done!"
