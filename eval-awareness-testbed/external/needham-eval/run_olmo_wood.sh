#!/bin/bash
#SBATCH --job-name=eval_olmo_wood
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH --mem=200G
#SBATCH --partition=cais
#SBATCH --output=./eval-olmo-wood-0119.out
#SBATCH --error=./eval-olmo-wood-0119.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

# Upgrade anthropic to fix version issue
pip install --upgrade anthropic

# Use different port to avoid conflict with other job
export VLLM_BASE_URL=http://localhost:8001/v1
export VLLM_API_KEY=dummy

# Function to run evaluation + CoT analysis for a model from shared_cais
run_full_eval_shared() {
    local HF_MODEL_ID=$1      # e.g., "allenai/OLMo-3.1-32B-Think"
    local MODEL_NAME=$2       # e.g., "allenai_OLMo-3.1-32B-Think" (for logging)
    local TP_SIZE=${3:-4}     # tensor parallel size, default 4
    local MAX_LEN=${4:-32768} # max model length, default 32768

    # Look in shared_cais/honesty_models for the model
    local CACHE_NAME=$(echo "$HF_MODEL_ID" | sed 's/\//--/g')
    local MODEL_PATH=$(ls -d /data/shared_cais/honesty_models/hub/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

    # Also check without /hub/ in path
    if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH=$(ls -d /data/shared_cais/honesty_models/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)
    fi

    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: Model not found: $HF_MODEL_ID in /data/shared_cais/honesty_models"
        return 1
    fi

    echo "=========================================="
    echo "Running full evaluation for: $HF_MODEL_ID"
    echo "Model path: $MODEL_PATH"
    echo "TP size: $TP_SIZE, Max length: $MAX_LEN"
    echo "=========================================="

    # Start vLLM server on port 8001
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8001 \
        --tensor-parallel-size $TP_SIZE \
        --served-model-name "$HF_MODEL_ID" \
        --max-model-len $MAX_LEN &

    VLLM_PID=$!

    # Wait for server to start
    echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
    sleep 180

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
# Run evaluations for OLMo and Wood models
# ==========================================

# MODEL 1: allenai/OLMo-3.1-32B-Think (32B)
run_full_eval_shared "allenai/OLMo-3.1-32B-Think" "allenai_OLMo-3.1-32B-Think" 4 32768

# MODEL 2: timhua/wood_v2_sftr4_filt (49B - needs reduced context)
run_full_eval_shared "timhua/wood_v2_sftr4_filt" "timhua_wood_v2_sftr4_filt" 4 16384

echo "=========================================="
echo "All model evaluations completed!"
echo "=========================================="

# ==========================================
# Generate plots (just for these 2 models)
# ==========================================
echo "=========================================="
echo "Generating plots and tables..."
echo "=========================================="

export MODELS_TO_PLOT="allenai_OLMo-3.1-32B-Think,timhua_wood_v2_sftr4_filt"

# Print score table
echo "=== Score Table ==="
python -m analysis.score_table eval_mcq

echo "=========================================="
echo "Done! OLMo and Wood evaluations complete."
echo "=========================================="
