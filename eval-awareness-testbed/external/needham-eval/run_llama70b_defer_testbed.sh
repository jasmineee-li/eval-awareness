#!/bin/bash
#SBATCH --job-name=eat_llama70b_defer
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --partition=cais
#SBATCH --output=./eat-llama70b-defer.out
#SBATCH --error=./eat-llama70b-defer.err

# Work from the needham-eval directory (where this script lives)
cd "$(dirname "$0")"

# Load environment variables
set -a
source ~/eval_awareness/.env 2>/dev/null || true
set +a

export VLLM_BASE_URL=http://localhost:8002/v1
export VLLM_API_KEY=dummy

# Base model
BASE_MODEL_PATH="/data/huggingface/Llama-3.3-70B-Instruct"

# LoRA adapter names (already downloaded from previous run)
ADAPTER_1_NAME="auditing-agents_llama_70b_transcripts_only_defer_to_users"
ADAPTER_2_NAME="auditing-agents_llama_70b_transcripts_only_then_redteam_high_defer_to_users"
ADAPTER_3_NAME="auditing-agents_llama_70b_transcripts_only_then_redteam_kto_defer_to_users"

# Resolve adapter snapshot paths (already cached from run_llama70b_defer.sh)
ADAPTER_CACHE="/data/shared_cais/honesty_models/hub"

get_adapter_path() {
    local repo="$1"
    local cache_name
    cache_name=$(echo "$repo" | sed 's/\//--/g')
    ls -d "${ADAPTER_CACHE}/models--${cache_name}/snapshots/"*/ 2>/dev/null | head -1
}

ADAPTER_1_PATH=$(get_adapter_path "auditing-agents/llama_70b_transcripts_only_defer_to_users")
ADAPTER_2_PATH=$(get_adapter_path "auditing-agents/llama_70b_transcripts_only_then_redteam_high_defer_to_users")
ADAPTER_3_PATH=$(get_adapter_path "auditing-agents/llama_70b_transcripts_only_then_redteam_kto_defer_to_users")

for name_path in \
    "$ADAPTER_1_NAME:$ADAPTER_1_PATH" \
    "$ADAPTER_2_NAME:$ADAPTER_2_PATH" \
    "$ADAPTER_3_NAME:$ADAPTER_3_PATH"; do
    name="${name_path%%:*}"
    path="${name_path##*:}"
    if [ -z "$path" ]; then
        echo "ERROR: Could not find adapter path for $name"
        echo "Run run_llama70b_defer.sh first to download adapters."
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

# Judges to run
JUDGES="probability_third_person,purpose_continue,verbalized_awareness"

# Output directory for results
RESULTS_DIR="results/testbed_judges_llama70b_defer"
mkdir -p "$RESULTS_DIR"

# Dataset path (relative to needham-eval/)
DATASET="data_repo/dataset.json"

echo "=========================================="
echo "Running testbed judges: $JUDGES"
echo "Dataset: $DATASET (976 samples)"
echo "=========================================="

for MODEL_NAME in "$ADAPTER_1_NAME" "$ADAPTER_2_NAME" "$ADAPTER_3_NAME"; do
    echo ""
    echo "=========================================="
    echo "Judging with model: $MODEL_NAME"
    echo "=========================================="

    OUTPUT_FILE="${RESULTS_DIR}/${MODEL_NAME}.json"

    # Run eat judge on the raw Needham dataset
    # The model (vLLM LoRA adapter) acts as the grader/actor answering probe questions
    eat judge \
        "$DATASET" \
        --methods "$JUDGES" \
        --model "vllm/${MODEL_NAME}" \
        --max-connections 50 \
        --output "$OUTPUT_FILE"

    echo "Results saved to $OUTPUT_FILE"
done

# Stop vLLM server
echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "All testbed judging complete!"
echo "Results in: $RESULTS_DIR/"
echo "  - ${ADAPTER_1_NAME}.json"
echo "  - ${ADAPTER_2_NAME}.json"
echo "  - ${ADAPTER_3_NAME}.json"
echo "=========================================="

echo "Done!"
