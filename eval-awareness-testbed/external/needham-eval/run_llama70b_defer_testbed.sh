#!/bin/bash
set -e

# Work from the needham-eval directory (where this script lives)
cd "$(dirname "$0")"

# Activate the project venv (where vllm is installed)
source /workspace/eval-awareness/.venv/bin/activate

# Cache HF models on /workspace (root filesystem is too small for 70B)
export HF_HOME=/workspace/.cache/huggingface

export VLLM_BASE_URL=http://localhost:8002/v1
export VLLM_API_KEY=dummy
export CUDA_VISIBLE_DEVICES=0,1,2,3


# Base model
BASE_MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct"

# LoRA adapter HuggingFace repo IDs
ADAPTER_1_REPO="auditing-agents/llama_70b_transcripts_only_defer_to_users"
ADAPTER_2_REPO="auditing-agents/llama_70b_transcripts_only_then_redteam_high_defer_to_users"
ADAPTER_3_REPO="auditing-agents/llama_70b_transcripts_only_then_redteam_kto_defer_to_users"

# Sanitized names (replace / with _) — used as vLLM lora-module names
ADAPTER_1_NAME="auditing-agents_llama_70b_transcripts_only_defer_to_users"
ADAPTER_2_NAME="auditing-agents_llama_70b_transcripts_only_then_redteam_high_defer_to_users"
ADAPTER_3_NAME="auditing-agents_llama_70b_transcripts_only_then_redteam_kto_defer_to_users"

# Download directory for LoRA adapters
ADAPTER_CACHE="/data/shared_cais/honesty_models/hub"

echo "=========================================="
echo "Downloading LoRA adapters from HuggingFace"
echo "=========================================="

hf download "$ADAPTER_1_REPO" --cache-dir "$ADAPTER_CACHE"
hf download "$ADAPTER_2_REPO" --cache-dir "$ADAPTER_CACHE"
hf download "$ADAPTER_3_REPO" --cache-dir "$ADAPTER_CACHE"

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
    --tensor-parallel-size 4 \
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

# Judges to run (probability_third_person already has good results, skip it)
JUDGES="purpose_continue"

# Classifier model for structured XML/JSON classification steps
# The LoRA adapters act as actor/grader; this model classifies their responses
CLASSIFIER_MODEL="openrouter/openai/gpt-5-mini"

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
    # The classifier model (GPT-5-mini) handles structured XML/JSON classification
    eat judge \
        "$DATASET" \
        --methods "$JUDGES" \
        --model "vllm/${MODEL_NAME}" \
        --classifier "$CLASSIFIER_MODEL" \
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
