#!/bin/bash
#SBATCH --job-name=eval_llama70b_defer
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --partition=cais
#SBATCH --output=./eval-llama70b-defer.out
#SBATCH --error=./eval-llama70b-defer.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

pip install --upgrade anthropic

export VLLM_BASE_URL=http://localhost:8002/v1
export VLLM_API_KEY=dummy

# Use the full dataset (976 samples)
export EVAL_AWARENESS_DATA_FILE=data_repo/dataset.json

# Base model
BASE_MODEL_PATH="/data/huggingface/Llama-3.3-70B-Instruct"

# LoRA adapter HuggingFace repo IDs
ADAPTER_1_REPO="auditing-agents/llama_70b_transcripts_only_defer_to_users"
ADAPTER_2_REPO="auditing-agents/llama_70b_transcripts_only_then_redteam_high_defer_to_users"
ADAPTER_3_REPO="auditing-agents/llama_70b_transcripts_only_then_redteam_kto_defer_to_users"

# Sanitized names (replace / with _) — used as vLLM lora-module names and in run.py
ADAPTER_1_NAME="auditing-agents_llama_70b_transcripts_only_defer_to_users"
ADAPTER_2_NAME="auditing-agents_llama_70b_transcripts_only_then_redteam_high_defer_to_users"
ADAPTER_3_NAME="auditing-agents_llama_70b_transcripts_only_then_redteam_kto_defer_to_users"

# Download directory for LoRA adapters
ADAPTER_CACHE="/data/shared_cais/honesty_models/hub"

echo "=========================================="
echo "Downloading LoRA adapters from HuggingFace"
echo "=========================================="

huggingface-cli download "$ADAPTER_1_REPO" --cache-dir "$ADAPTER_CACHE"
huggingface-cli download "$ADAPTER_2_REPO" --cache-dir "$ADAPTER_CACHE"
huggingface-cli download "$ADAPTER_3_REPO" --cache-dir "$ADAPTER_CACHE"

# Resolve adapter snapshot paths
get_adapter_path() {
    local repo="$1"
    local cache_name
    cache_name=$(echo "$repo" | sed 's/\//--/g')
    ls -d "${ADAPTER_CACHE}/models--${cache_name}/snapshots/"*/ 2>/dev/null | head -1
}

ADAPTER_1_PATH=$(get_adapter_path "$ADAPTER_1_REPO")
ADAPTER_2_PATH=$(get_adapter_path "$ADAPTER_2_REPO")
ADAPTER_3_PATH=$(get_adapter_path "$ADAPTER_3_REPO")

for name_path in \
    "$ADAPTER_1_NAME:$ADAPTER_1_PATH" \
    "$ADAPTER_2_NAME:$ADAPTER_2_PATH" \
    "$ADAPTER_3_NAME:$ADAPTER_3_PATH"; do
    name="${name_path%%:*}"
    path="${name_path##*:}"
    if [ -z "$path" ]; then
        echo "ERROR: Could not find adapter path for $name"
        exit 1
    fi
    echo "Adapter $name -> $path"
done

echo "=========================================="
echo "Starting vLLM server with 3 LoRA adapters"
echo "Base model: $BASE_MODEL_PATH"
echo "=========================================="

vllm serve "$BASE_MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size 8 \
    --served-model-name "meta-llama/Llama-3.3-70B-Instruct" \
    --max-model-len 16384 \
    --enable-lora \
    --lora-modules \
        "${ADAPTER_1_NAME}=${ADAPTER_1_PATH}" \
        "${ADAPTER_2_NAME}=${ADAPTER_2_PATH}" \
        "${ADAPTER_3_NAME}=${ADAPTER_3_PATH}" \
    --max-lora-rank 64 \
    --trust-remote-code &

VLLM_PID=$!

echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
sleep 300

if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM server failed to start"
    exit 1
fi

echo "vLLM server started successfully!"

# Run eval_mcq for each adapter
echo "=========================================="
echo "Running eval_mcq for all 3 models"
echo "=========================================="

for MODEL_NAME in "$ADAPTER_1_NAME" "$ADAPTER_2_NAME" "$ADAPTER_3_NAME"; do
    echo "Running eval_mcq for $MODEL_NAME..."
    python run.py --model "vllm/${MODEL_NAME}" --stage eval_mcq --epochs 1
done

# Stop vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Generating plots and score table"
echo "=========================================="

# Set models for plotting
export MODELS_TO_PLOT="${ADAPTER_1_NAME},${ADAPTER_2_NAME},${ADAPTER_3_NAME}"

python -m analysis.roc eval_mcq
python -m analysis.score_table eval_mcq

echo "=========================================="
echo "All evaluations complete!"
echo "Check:"
echo "  - ROC plot: figures/roc.pdf"
echo "  - Logs: logs/eval_mcq/<model_name>/"
echo "  - Score table: printed above"
echo "=========================================="

echo "Done!"
