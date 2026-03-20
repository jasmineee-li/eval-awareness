#!/bin/bash
# Run 100-sample experiments for the 3 OLMo checkpoints that haven't been done yet.
# OLMo-3.1-32B-Think already has 100 samples in the _250222 result dirs; this
# script adds the other 3 checkpoints to the same dirs (same configs, same experiment_id).
#
# Usage:
#   bash scripts/run_remaining_checkpoints.sh
#
# Expects:
#   - vLLM installed (pip install vllm)
#   - 2+ GPUs with sufficient memory for 32B models
#   - .env with ANTHROPIC_API_KEY (harmful classification) and OPENROUTER_API_KEY (eval awareness)

set -e

FRAMEWORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$FRAMEWORK_DIR"

# Activate the project venv so vllm/python are on PATH
VENV_DIR="$(cd "$FRAMEWORK_DIR/.." && pwd)/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
fi

mkdir -p logs

export PYTHONUNBUFFERED=1

# Redirect HuggingFace/vLLM caches to workspace (root overlay is too small for 32B models)
export HF_HOME="/workspace/.cache/huggingface"
export VLLM_CACHE_ROOT="/workspace/.cache/vllm"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

# Load .env for API keys (needed for classification step)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Auto-detect GPU count for tensor parallelism
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -ge 4 ]; then
    TP_SIZE=4
elif [ "$NUM_GPUS" -ge 2 ]; then
    TP_SIZE=2
else
    TP_SIZE=1
fi
echo "Detected $NUM_GPUS GPUs, using tensor_parallel_size=$TP_SIZE"

# vLLM configuration
VLLM_PORT=8000
MAX_MODEL_LEN=8192

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# Only the 3 checkpoints that still need 100 samples
# (OLMo-3.1-32B-Think already done in the _250222 runs)
MODELS=(
    "allenai/Olmo-3-32B-Think-SFT"
    "allenai/Olmo-3-32B-Think-DPO"
    "allenai/Olmo-3-32B-Think"
)

# Same configs as before — results land in the existing _250222 dirs
CONFIGS=(
    "configs/olmo_checkpoints_vllm_baseline.yaml"
    "configs/olmo_checkpoints_vllm_af.yaml"
)

# Step 0: Generate prompts (idempotent — skips if already generated)
for config in "${CONFIGS[@]}"; do
    echo "=== Generating prompts for: $config ==="
    python scripts/generate_prompts.py --config "$config"
done

# Step 1: For each model, serve via vLLM and run both experiments
for model in "${MODELS[@]}"; do
    short_name="${model##*/}"
    echo ""
    echo "=========================================="
    echo "Serving model: $model"
    echo "=========================================="

    # Restrict vLLM routing to just this model
    export VLLM_MODELS="$model"

    # Start vLLM server (downloads from HF if not cached)
    vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "$model" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for vLLM to be ready (poll every 5s, up to 30 min for first-time downloads)
    echo "Waiting for vLLM server to start..."
    for i in $(seq 1 360); do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly for $model"
            break
        fi
        sleep 5
    done

    if ! curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: vLLM server did not become ready for $model, skipping."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        sleep 10
        continue
    fi

    # Run both experiments (baseline + AF) for THIS model only
    for config in "${CONFIGS[@]}"; do
        echo "--- Running experiment: $config for $model ---"
        python scripts/run_experiments.py --config "$config" --models "$model" || {
            echo "WARNING: Experiment $config had failures for $model"
        }
    done

    # Stop vLLM server
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
    echo "GPU memory released for $short_name"
done

echo ""
echo "=========================================="
echo "All remaining checkpoint runs complete!"
echo "=========================================="
echo ""
echo "Results added to:"
echo "  results/olmo_checkpoints_baseline_250222/"
echo "  results/olmo_checkpoints_af_250222/"
