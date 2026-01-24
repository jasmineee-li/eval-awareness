#!/bin/bash
#SBATCH --job-name=cot_analysis
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --partition=cais
#SBATCH --output=./cot-analysis.out
#SBATCH --error=./cot-analysis.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

# Upgrade anthropic to fix version issue
pip install --upgrade anthropic

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy

# Function to run CoT analysis for a model
run_cot_analysis() {
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
    echo "Running CoT analysis for: $HF_MODEL_ID"
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

    # Run CoT analysis stages
    echo "Running eval_mcq_cot..."
    python run.py --model "vllm/$HF_MODEL_ID" --stage eval_mcq_cot --limit 2500 --epochs 1

    echo "Running purpose_open_cot..."
    python run.py --model "vllm/$HF_MODEL_ID" --stage purpose_open_cot --limit 2500 --epochs 1

    # Stop vLLM server
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    sleep 10
}

# MODEL 1: Qwen/QwQ-32B
run_cot_analysis "Qwen/QwQ-32B" "Qwen_QwQ-32B" 4

# MODEL 2: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
run_cot_analysis "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "DeepSeek-R1-32B" 4

# MODEL 3: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
run_cot_analysis "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" "DeepSeek-R1-14B" 2

# MODEL 4: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
run_cot_analysis "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "DeepSeek-R1-7B" 1

echo "All CoT analysis completed!"

# Generate CoT plot
echo "=========================================="
echo "Generating CoT plots..."
echo "=========================================="

export MODELS_TO_PLOT="Qwen_QwQ-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-14B,deepseek-ai_DeepSeek-R1-Distill-Qwen-7B"

# Generate CoT breakdown plot (Figure 7)
python -m analysis.cot eval_mcq_cot

# Generate purpose plot (Figure 6) - requires purpose_mcq + purpose_open_cot
python -m analysis.purpose eval_mcq

echo "Done! Plots saved to figures/ directory."
