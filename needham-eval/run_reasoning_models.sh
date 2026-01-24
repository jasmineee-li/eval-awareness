#!/bin/bash
#SBATCH --job-name=eval_reasoning
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=cais
#SBATCH --output=./eval-reasoning-0119.out
#SBATCH --error=./eval-reasoning-0119.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

# Upgrade anthropic to fix version issue
pip install --upgrade anthropic

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy

# Function to download model if not present
download_model() {
    local HF_MODEL_ID=$1
    local CACHE_NAME=$(echo "$HF_MODEL_ID" | sed 's/\//--/g')
    local MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "Model not found locally. Downloading $HF_MODEL_ID..."
        HF_HOME=/data/huggingface huggingface-cli download "$HF_MODEL_ID"
    fi
}

# Function to run evaluation + CoT analysis for a model
run_full_eval() {
    local HF_MODEL_ID=$1      # e.g., "allenai/OLMo-3.1-32B-Think"
    local MODEL_NAME=$2       # e.g., "allenai_OLMo-3.1-32B-Think" (for logging)
    local TP_SIZE=${3:-4}     # tensor parallel size, default 4
    local MAX_LEN=${4:-32768} # max model length, default 32768

    # Convert HF model ID to cache path format (replace / with --)
    local CACHE_NAME=$(echo "$HF_MODEL_ID" | sed 's/\//--/g')
    local MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: Model not found: $HF_MODEL_ID (looked for models--${CACHE_NAME})"
        return 1
    fi

    echo "=========================================="
    echo "Running full evaluation for: $HF_MODEL_ID"
    echo "Model path: $MODEL_PATH"
    echo "TP size: $TP_SIZE, Max length: $MAX_LEN"
    echo "=========================================="

    # Start vLLM server
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size $TP_SIZE \
        --served-model-name "$HF_MODEL_ID" \
        --max-model-len $MAX_LEN &

    VLLM_PID=$!

    # Wait for server to start
    echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
    sleep 180  # Larger models need more time

    # Check if server started successfully
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server failed to start for $HF_MODEL_ID"
        return 1
    fi

    # Run main evaluation (eval_mcq, purpose_mcq, purpose_open)
    echo "Running main evaluation..."
    python run.py --model "vllm/$HF_MODEL_ID" --limit 2500 --epochs 1

    # Run CoT analysis
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

# ==========================================
# Step 1: Download all models first
# ==========================================
echo "=========================================="
echo "Step 1: Downloading models if needed..."
echo "=========================================="

download_model "allenai/OLMo-3.1-32B-Think"
download_model "openai/gpt-oss-20b"
download_model "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
download_model "timhua/wood_v2_sftr4_filt"

# ==========================================
# Step 2: Run evaluations for each model
# ==========================================

# MODEL 1: allenai/OLMo-3.1-32B-Think (32B)
run_full_eval "allenai/OLMo-3.1-32B-Think" "allenai_OLMo-3.1-32B-Think" 4 32768

# MODEL 2: openai/gpt-oss-20b (20B)
run_full_eval "openai/gpt-oss-20b" "openai_gpt-oss-20b" 2 32768

# MODEL 3: nvidia/Llama-3_3-Nemotron-Super-49B-v1 (49B - needs more resources)
run_full_eval "nvidia/Llama-3_3-Nemotron-Super-49B-v1" "nvidia_Llama-3_3-Nemotron-Super-49B-v1" 4 16384

# MODEL 4: timhua/wood_v2_sftr4_filt (49B - needs reduced context)
run_full_eval "timhua/wood_v2_sftr4_filt" "timhua_wood_v2_sftr4_filt" 4 16384

echo "=========================================="
echo "All model evaluations completed!"
echo "=========================================="

# ==========================================
# Step 3: Generate plots with dated filenames
# ==========================================
echo "=========================================="
echo "Generating plots and tables..."
echo "=========================================="

# Include all models (existing + new)
export MODELS_TO_PLOT="Qwen_QwQ-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-32B,deepseek-ai_DeepSeek-R1-Distill-Qwen-14B,deepseek-ai_DeepSeek-R1-Distill-Qwen-7B,allenai_OLMo-3.1-32B-Think,openai_gpt-oss-20b,nvidia_Llama-3_3-Nemotron-Super-49B-v1,timhua_wood_v2_sftr4_filt"

# Print score table (AUC, Brier, ECE)
echo "=== Score Table ==="
python -m analysis.score_table eval_mcq

# Generate ROC curves
python -m analysis.roc eval_mcq
cp figures/roc.pdf figures/roc_0119.pdf

# Generate calibration curves
python -m analysis.calib eval_mcq
cp figures/calibration_lines.pdf figures/calibration_0119.pdf

# Generate CoT breakdown plot
python -m analysis.cot eval_mcq_cot
cp figures/seaborn_explanation_distribution_all.pdf figures/cot_0119.pdf

echo "=========================================="
echo "Done! Plots saved to figures/ directory:"
echo "  - figures/roc_0119.pdf"
echo "  - figures/calibration_0119.pdf"
echo "  - figures/cot_0119.pdf"
echo "=========================================="
