#!/bin/bash
#SBATCH --job-name=eval_olmo
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --partition=cais
#SBATCH --output=./eval-olmo.out
#SBATCH --error=./eval-olmo.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

# Upgrade anthropic to fix version issue
pip install --upgrade anthropic

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy

# Model configuration
HF_MODEL_ID="allenai/OLMo-3.1-32B-Think"
MODEL_NAME="allenai_OLMo-3.1-32B-Think"
TP_SIZE=4

# Step 1: Download model if not present
echo "=========================================="
echo "Step 1: Checking/downloading model..."
echo "=========================================="

CACHE_NAME=$(echo "$HF_MODEL_ID" | sed 's/\//--/g')
MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "Model not found locally. Downloading $HF_MODEL_ID..."
    HF_HOME=/data/huggingface huggingface-cli download "$HF_MODEL_ID"
    MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: Failed to download model $HF_MODEL_ID"
        exit 1
    fi
fi

echo "Model path: $MODEL_PATH"

# Step 2: Start vLLM server
echo "=========================================="
echo "Step 2: Starting vLLM server..."
echo "=========================================="

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size $TP_SIZE \
    --served-model-name "$HF_MODEL_ID" \
    --max-model-len 32768 &

VLLM_PID=$!

echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
sleep 120

# Step 3: Run main evaluation (eval_mcq, purpose_mcq, purpose_open)
echo "=========================================="
echo "Step 3: Running main evaluation..."
echo "=========================================="

python run.py --model "vllm/$HF_MODEL_ID" --limit 2500 --epochs 1

# Step 4: Run CoT analysis
echo "=========================================="
echo "Step 4: Running CoT analysis..."
echo "=========================================="

echo "Running eval_mcq_cot..."
python run.py --model "vllm/$HF_MODEL_ID" --stage eval_mcq_cot --limit 2500 --epochs 1

echo "Running purpose_open_cot..."
python run.py --model "vllm/$HF_MODEL_ID" --stage purpose_open_cot --limit 2500 --epochs 1

# Step 5: Stop vLLM server
echo "=========================================="
echo "Step 5: Stopping vLLM server..."
echo "=========================================="

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# Step 6: Generate plots
echo "=========================================="
echo "Step 6: Generating plots and tables..."
echo "=========================================="

export MODELS_TO_PLOT="$MODEL_NAME"

# Print score table (AUC, Brier, ECE)
python -m analysis.score_table eval_mcq

# Generate ROC curves
python -m analysis.roc eval_mcq

# Generate calibration curves
python -m analysis.calib eval_mcq

# Generate CoT breakdown plot
python -m analysis.cot eval_mcq_cot

echo "=========================================="
echo "Done! All evaluations complete for $HF_MODEL_ID"
echo "Plots saved to figures/ directory."
echo "=========================================="
