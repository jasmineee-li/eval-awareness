#!/bin/bash
#SBATCH --job-name=k2v2-agentic
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Run agentic-misalignment experiments on K2-V2 Instruct and Think models via local vLLM.
# Serves each model sequentially, runs baseline experiments, then classifies.
#
# K2-V2 7B models: TP=2, 60G mem is sufficient.
#
# Usage:
#   sbatch agentic-misalignment/scripts/run_k2v2_agentic.sh

set -uo pipefail

REPO_ROOT="${EVAL_AWARENESS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT/agentic-misalignment"

mkdir -p logs

export PYTHONUNBUFFERED=1

# Load .env for API keys (needed for classification step)
if [ -f ../.env ]; then
    set -a; source ../.env; set +a
fi
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ─── vLLM configuration ───
VLLM_PORT=8000
TP_SIZE=2
MAX_MODEL_LEN=32768

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Models ───
MODELS=(
    "LLM360/K2-V2-Instruct"
    "LLM360/K2-Think-V2"
)

CONFIG="configs/k2v2_checkpoints_vllm_baseline.yaml"

# ─── Helpers ───

start_vllm() {
    local model="$1"
    echo "=== Starting vLLM: $model (TP=$TP_SIZE) ==="
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
}

wait_for_vllm() {
    echo "Waiting for vLLM server to start..."
    for i in $(seq 1 120); do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM server did not become ready within 10 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

# ─── Step 0: Generate prompts ───
echo "=== Generating prompts for: $CONFIG ==="
python scripts/generate_prompts.py --config "$CONFIG"

# ─── Step 1: Serve each model and run experiments ───
TOTAL=${#MODELS[@]}
COMPLETED=0
FAILED=0

for ((idx=0; idx<TOTAL; idx++)); do
    model="${MODELS[$idx]}"
    short_name="${model##*/}"

    echo ""
    echo "=========================================="
    echo "[$((idx+1))/$TOTAL] $short_name ($model)"
    echo "=========================================="

    export VLLM_MODELS="$model"

    start_vllm "$model"

    if ! wait_for_vllm; then
        stop_vllm
        echo "WARNING: Skipping $short_name due to vLLM startup failure"
        ((FAILED++))
        continue
    fi

    echo "--- Running experiment: $CONFIG for $model ---"
    if python scripts/run_experiments.py --config "$CONFIG"; then
        ((COMPLETED++))
    else
        echo "WARNING: Experiment had failures for $short_name"
        ((FAILED++))
    fi

    stop_vllm
done

# ─── Step 2: Classification (post-hoc, uses Claude via API) ───
echo ""
echo "=== Running post-hoc classification ==="
unset VLLM_BASE_URL
unset VLLM_MODELS

EXPERIMENT_ID=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment_id'])")
RESULTS_DIR="results/$EXPERIMENT_ID"

if [ -d "$RESULTS_DIR" ]; then
    echo "Classifying results in: $RESULTS_DIR"
    python scripts/classify_results.py \
        --results-dir "$RESULTS_DIR" \
        --config "$CONFIG" \
        --concurrency 20
else
    echo "WARNING: Results directory not found: $RESULTS_DIR"
fi

echo ""
echo "=========================================="
echo "K2-V2 Agentic Misalignment Complete!"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed:    $FAILED"
echo "  Results:   $RESULTS_DIR/"
echo "=========================================="
