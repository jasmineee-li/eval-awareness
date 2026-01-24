#!/bin/bash
#SBATCH --job-name=eval_mini
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --partition=cais
#SBATCH --output=./eval-mini.out
#SBATCH --error=./eval-mini.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

# Upgrade anthropic to fix version issue
pip install --upgrade anthropic

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy

# Function to run a model (--mini = 50 samples, 1 epoch)
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
    echo "Running model: $HF_MODEL_ID (--mini)"
    echo "Model path: $MODEL_PATH"
    echo "=========================================="

    # Start vLLM server
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size $TP_SIZE \
        --served-model-name "$HF_MODEL_ID" \
        --max-model-len 65536 &

    VLLM_PID=$!

    # Wait for server to start
    echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
    sleep 120

    # Run evaluation (--mini = 50 samples, 1 epoch)
    python run.py --mini --model "vllm/$HF_MODEL_ID"

    # Stop vLLM server
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    sleep 10
}

# Function to run a model from a local path (--mini = 50 samples, 1 epoch)
run_local_model() {
    local MODEL_PATH=$1       # e.g., "/data/shared_cais/honesty_models/sfm_..."
    local MODEL_NAME=$2       # e.g., "sfm_unfiltered_cpt_misalignment_upsampled_dpo" (for logging and vLLM)
    local TP_SIZE=${3:-4}     # tensor parallel size, default 4

    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model not found at path: $MODEL_PATH"
        return 1
    fi

    echo "=========================================="
    echo "Running local model: $MODEL_NAME (--mini)"
    echo "Model path: $MODEL_PATH"
    echo "=========================================="

    # Start vLLM server
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size $TP_SIZE \
        --served-model-name "$MODEL_NAME" \
        --max-model-len 32768 &

    VLLM_PID=$!

    # Wait for server to start
    echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
    sleep 120

    # Run evaluation (--mini = 50 samples, 1 epoch)
    python run.py --mini --model "vllm/$MODEL_NAME"

    # Stop vLLM server
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    sleep 10
}

# Skip MODEL 1 (Qwen/QwQ-32B) - already ran

# MODEL 2: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# run_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "DeepSeek-R1-32B" 4

# MODEL 3: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (already ran)
# run_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" "DeepSeek-R1-14B" 2

# MODEL 4: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (already ran)
# run_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "DeepSeek-R1-7B" 1

# MODEL 5: Qwen/Qwen3-32B (reasoning model)
# run_model "Qwen/Qwen3-32B" "Qwen3-32B" 4

# MODEL 6: sfm_unfiltered_cpt_misalignment_upsampled_dpo (local model)
run_local_model "/data/shared_cais/honesty_models/sfm_unfiltered_cpt_misalignment_upsampled_dpo" "sfm_unfiltered_cpt_misalignment_upsampled_dpo" 1

echo "All models completed!"

# Generate plots for all models
echo "=========================================="
echo "Generating plots and tables..."
echo "=========================================="

export MODELS_TO_PLOT="Qwen_QwQ-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-14B,deepseek-ai_DeepSeek-R1-Distill-Qwen-7B,Qwen_Qwen3-32B"
export MODELS_TO_PLOT="Qwen_QwQ-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-14B,deepseek-ai_DeepSeek-R1-Distill-Qwen-7B,sfm_unfiltered_cpt_misalignment_upsampled_dpo"

# Print score table (AUC, Brier, ECE)
python -m analysis.score_table eval_mcq_mini

# Generate ROC curves
python -m analysis.roc eval_mcq_mini

# Generate calibration curves
python -m analysis.calib eval_mcq_mini

echo "Done! Plots saved to current directory."
