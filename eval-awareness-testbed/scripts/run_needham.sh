#!/bin/bash
# Consolidated needham-eval runner script.
# Replaces 10 individual run_*.sh scripts with one parameterized script.
#
# Usage:
#   ./scripts/run_needham.sh configs/needham/qwq32b.yaml
#   sbatch scripts/run_needham.sh configs/needham/qwq32b.yaml
#
# Config format (YAML):
#   slurm:
#     gpus: 4
#     time: "13:00:00"
#     mem: "128G"
#   vllm:
#     port: 8000
#     max_model_len: 32768
#     startup_wait: 120
#     extra_args: []
#   models:
#     - hf_id: "Qwen/QwQ-32B"
#       tp_size: 4
#       limit: 2500
#       epochs: 1
#       stages: [eval_mcq]
#       lora_adapter: null
#       lora_modules: null
#       tokenizer: null
#   analysis:
#     plots: [score_table, roc, calib]
#
# Requires: yq (https://github.com/mikefarah/yq) for YAML parsing.

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NEEDHAM_DIR="$SCRIPT_DIR/external/needham-eval"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

if ! command -v yq &>/dev/null; then
    echo "ERROR: yq is required for YAML parsing. Install: https://github.com/mikefarah/yq"
    exit 1
fi

# Read vLLM config
VLLM_PORT=$(yq '.vllm.port // 8000' "$CONFIG")
VLLM_MAX_MODEL_LEN=$(yq '.vllm.max_model_len // 32768' "$CONFIG")
VLLM_STARTUP_WAIT=$(yq '.vllm.startup_wait // 120' "$CONFIG")

# Setup environment
if [ -f ~/eval_awareness/env/bin/activate ]; then
    source ~/eval_awareness/env/bin/activate
fi

if [ -f ~/eval_awareness/.env ]; then
    set -a
    source ~/eval_awareness/.env
    set +a
fi

export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export VLLM_API_KEY=dummy

cd "$NEEDHAM_DIR"

# Function to run a single model
run_model() {
    local idx=$1

    local HF_MODEL_ID=$(yq ".models[$idx].hf_id" "$CONFIG")
    local TP_SIZE=$(yq ".models[$idx].tp_size // 4" "$CONFIG")
    local LIMIT=$(yq ".models[$idx].limit // 2500" "$CONFIG")
    local EPOCHS=$(yq ".models[$idx].epochs // 1" "$CONFIG")
    local LORA_ADAPTER=$(yq ".models[$idx].lora_adapter // \"\"" "$CONFIG")
    local LORA_MODULES=$(yq ".models[$idx].lora_modules // \"\"" "$CONFIG")
    local TOKENIZER=$(yq ".models[$idx].tokenizer // \"\"" "$CONFIG")

    # Read stages as space-separated list
    local STAGES=$(yq ".models[$idx].stages // [\"eval_mcq\"] | .[]" "$CONFIG")

    # Determine model name for vllm --served-model-name
    local SERVED_NAME="$HF_MODEL_ID"
    local RUN_MODEL="vllm/$HF_MODEL_ID"

    # If LoRA adapter is specified, serve as adapter name
    if [ -n "$LORA_ADAPTER" ] && [ "$LORA_ADAPTER" != "null" ]; then
        # Use adapter name for model identification
        RUN_MODEL="vllm/$LORA_ADAPTER"
    fi

    # Resolve model path from HuggingFace cache
    local CACHE_NAME=$(echo "$HF_MODEL_ID" | sed 's/\//--/g')
    local MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: Model not found in cache: $HF_MODEL_ID (looked for models--${CACHE_NAME})"
        return 1
    fi

    echo "=========================================="
    echo "Running model: $HF_MODEL_ID"
    echo "Model path: $MODEL_PATH"
    echo "TP size: $TP_SIZE, Limit: $LIMIT, Epochs: $EPOCHS"
    [ -n "$LORA_ADAPTER" ] && [ "$LORA_ADAPTER" != "null" ] && echo "LoRA adapter: $LORA_ADAPTER"
    echo "=========================================="

    # Build vLLM arguments
    local VLLM_ARGS=(
        "$MODEL_PATH"
        --host 0.0.0.0
        --port "$VLLM_PORT"
        --tensor-parallel-size "$TP_SIZE"
        --served-model-name "$SERVED_NAME"
        --max-model-len "$VLLM_MAX_MODEL_LEN"
    )

    # Add tokenizer override if specified
    if [ -n "$TOKENIZER" ] && [ "$TOKENIZER" != "null" ]; then
        VLLM_ARGS+=(--tokenizer "$TOKENIZER")
    fi

    # Add LoRA support if specified
    if [ -n "$LORA_ADAPTER" ] && [ "$LORA_ADAPTER" != "null" ]; then
        VLLM_ARGS+=(--enable-lora --max-lora-rank 64 --trust-remote-code)
        if [ -n "$LORA_MODULES" ] && [ "$LORA_MODULES" != "null" ]; then
            VLLM_ARGS+=(--lora-modules "$LORA_MODULES")
        fi
    fi

    # Add any extra vllm args from config
    local EXTRA_ARGS=$(yq '.vllm.extra_args // [] | .[]' "$CONFIG" 2>/dev/null)
    if [ -n "$EXTRA_ARGS" ]; then
        while IFS= read -r arg; do
            VLLM_ARGS+=("$arg")
        done <<< "$EXTRA_ARGS"
    fi

    # Start vLLM server
    vllm serve "${VLLM_ARGS[@]}" &
    VLLM_PID=$!

    echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
    sleep "$VLLM_STARTUP_WAIT"

    # Check if server started
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server failed to start"
        return 1
    fi

    # Run each stage
    for stage in $STAGES; do
        echo "Running stage: $stage..."
        python run.py --model "$RUN_MODEL" --stage "$stage" --limit "$LIMIT" --epochs "$EPOCHS"
    done

    # Stop vLLM server
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    sleep 10
}

# Run all models
NUM_MODELS=$(yq '.models | length' "$CONFIG")
for ((i=0; i<NUM_MODELS; i++)); do
    run_model $i
done

echo "All models completed!"

# Run analysis if configured
PLOTS=$(yq '.analysis.plots // [] | .[]' "$CONFIG" 2>/dev/null)
if [ -n "$PLOTS" ]; then
    echo "=========================================="
    echo "Generating plots and tables..."
    echo "=========================================="

    for plot in $PLOTS; do
        case "$plot" in
            score_table)
                python -m analysis.score_table eval_mcq
                ;;
            roc)
                python -m analysis.roc eval_mcq
                ;;
            calib)
                python -m analysis.calib eval_mcq
                ;;
            cot)
                python -m analysis.cot eval_mcq_cot
                ;;
            purpose)
                python -m analysis.purpose purpose_open_cot
                ;;
            *)
                echo "Unknown plot type: $plot"
                ;;
        esac
    done

    echo "Done! Plots saved to figures/ directory."
fi
