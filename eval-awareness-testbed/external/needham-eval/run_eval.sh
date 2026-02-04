#!/bin/bash
#SBATCH --job-name=eval_awareness
#SBATCH --gres=gpu:4
#SBATCH --time=13:00:00
#SBATCH --mem=128G
#SBATCH --partition=cais
#SBATCH --output=./eval-models-0118.out
#SBATCH --error=./eval-models-0118.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

# Upgrade anthropic to fix version issue
pip install --upgrade anthropic

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy

# Function to run a model
run_model() {
    local HF_MODEL_ID=$1      # e.g., "Qwen/QwQ-32B"
    local MODEL_NAME=$2       # e.g., "Qwen_QwQ-32B" (for logging)
    local TP_SIZE=${3:-4}     # tensor parallel size, default 4

    # Convert HF model ID to cache path format (replace / with --)
    local CACHE_NAME=$(echo "$HF_MODEL_ID" | sed 's/\//--/g')
    local MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: Model not found: $HF_MODEL_ID (looked for models--${CACHE_NAME})"
        return 1
    fi

    echo "=========================================="
    echo "Running model: $HF_MODEL_ID"
    echo "Model path: $MODEL_PATH"
    echo "=========================================="

    # Start vLLM server
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size $TP_SIZE \
        --served-model-name "$HF_MODEL_ID" \
        --max-model-len 32768 &

    VLLM_PID=$!

    # Wait for server to start
    echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
    sleep 120

    # Run evaluation (400 samples, 1 epoch)
    python run.py --model "vllm/$HF_MODEL_ID" --limit 2500 --epochs 1

    # Stop vLLM server
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    sleep 10
}

# MODEL 1: Qwen/QwQ-32B
run_model "Qwen/QwQ-32B" "Qwen_QwQ-32B" 4

# MODEL 2: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
run_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "DeepSeek-R1-32B" 4

# MODEL 3: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
run_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" "DeepSeek-R1-14B" 2

# MODEL 4: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
run_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "DeepSeek-R1-7B" 1

# MODEL 6: nvidia/OpenReasoning-Nemotron-32B
run_model "nvidia/OpenReasoning-Nemotron-32B" "Nemotron-32B" 4

echo "All models completed!"

# Generate plots for all models
echo "=========================================="
echo "Generating plots and tables..."
echo "=========================================="

export MODELS_TO_PLOT="Qwen_QwQ-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-14B,deepseek-ai_DeepSeek-R1-Distill-Qwen-7B,nvidia_OpenReasoning-Nemotron-32B"

# Print score table (AUC, Brier, ECE)
python -m analysis.score_table eval_mcq

# Generate ROC curves
python -m analysis.roc eval_mcq

# Generate calibration curves
python -m analysis.calib eval_mcq

echo "Done! Plots saved to figures/ directory."
