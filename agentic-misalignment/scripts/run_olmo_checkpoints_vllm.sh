#!/bin/bash
#SBATCH --job-name=olmo-agentic-misalign
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/agentic-misalignment/logs/olmo-checkpoints-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/agentic-misalignment/logs/olmo-checkpoints-%j.err

# Run agentic-misalignment experiments on OLMo training stages via local vLLM.
# Serves each model sequentially, runs both baseline and AF experiments, then classifies.
#
# Usage:
#   sbatch scripts/run_olmo_checkpoints_vllm.sh

set -e

FRAMEWORK_DIR="/data/jasmine_li/eval-awareness/agentic-misalignment"
cd "$FRAMEWORK_DIR"

mkdir -p logs

export PYTHONUNBUFFERED=1

# Activate conda env with vLLM
source /data/jasmine_li/miniconda3/bin/activate rllm

# Load .env for API keys (needed for classification step)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# vLLM configuration
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=8192

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# Models in training order
MODELS=(
    "allenai/Olmo-3-32B-Think-SFT"
    "allenai/Olmo-3-32B-Think-DPO"
    "allenai/Olmo-3-32B-Think"
    "allenai/OLMo-3.1-32B-Think"
)

# Experiment configs
CONFIGS=(
    "configs/olmo_checkpoints_vllm_baseline.yaml"
    "configs/olmo_checkpoints_vllm_af.yaml"
)

# Step 0: Generate prompts (once per config)
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

    # Resolve model path from HuggingFace cache
    CACHE_NAME=$(echo "$model" | sed 's/\//--/g')
    MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "Model not in HF cache, downloading: $model"
        MODEL_PATH="$model"
    fi

    echo "Model path: $MODEL_PATH"

    # Start vLLM server
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "$model" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for vLLM to be ready (poll every 5s, up to 10 min)
    echo "Waiting for vLLM server to start..."
    for i in $(seq 1 120); do
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

    # Run both experiments (baseline + AF)
    for config in "${CONFIGS[@]}"; do
        echo "--- Running experiment: $config for $model ---"
        python scripts/run_experiments.py --config "$config" || {
            echo "WARNING: Experiment $config had failures (expected for non-served models)"
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
echo "All model runs complete!"
echo "=========================================="

# Step 2: Classification (post-hoc, uses Claude via API)
echo ""
echo "=== Running post-hoc classification ==="
unset VLLM_BASE_URL
unset VLLM_MODELS

for config in "${CONFIGS[@]}"; do
    EXPERIMENT_ID=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['experiment_id'])")
    RESULTS_DIR="results/$EXPERIMENT_ID"

    if [ -d "$RESULTS_DIR" ]; then
        echo "Classifying results in: $RESULTS_DIR"
        python scripts/classify_results.py \
            --results-dir "$RESULTS_DIR" \
            --config "$config" \
            --concurrency 20
    else
        echo "WARNING: Results directory not found: $RESULTS_DIR"
    fi
done

# Step 3: Plot results
echo ""
echo "=== Generating plots ==="
python scripts/plot_olmo_checkpoints.py \
    --baseline-dir results/olmo_checkpoints_baseline_250221 \
    --af-dir results/olmo_checkpoints_af_250221 \
    --output figures/olmo_checkpoints/

echo ""
echo "=== Pipeline complete! ==="
echo "Results: results/olmo_checkpoints_baseline_250221/"
echo "         results/olmo_checkpoints_af_250221/"
echo "Figures: figures/olmo_checkpoints/"
