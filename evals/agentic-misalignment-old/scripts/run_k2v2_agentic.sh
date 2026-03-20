#!/bin/bash
#SBATCH --job-name=k2v2-agentic
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

# Run agentic-misalignment experiments on K2-V2 Instruct and Think models via local vLLM.
# Serves each model sequentially, runs baseline experiments, then classifies.
#
# K2-V2 models (~68B params, bf16 ~68G on GPU): TP=4 on A100-40G for sufficient KV cache.
#
# Usage:
#   sbatch agentic-misalignment/scripts/run_k2v2_agentic.sh

set -uo pipefail

REPO_ROOT="${EVAL_AWARENESS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT/agentic-misalignment" || { echo "ERROR: Cannot cd to $REPO_ROOT/agentic-misalignment"; exit 1; }

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
TP_SIZE=4
MAX_MODEL_LEN=32768

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Models ───
MODELS=(
    # "LLM360/K2-V2-Instruct"
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
    local model="$1"
    echo "Waiting for vLLM to load model: $model ..."
    for i in $(seq 1 360); do
        if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); ids=[m['id'] for m in data.get('data',[])]; sys.exit(0 if '$model' in ids else 1)" 2>/dev/null; then
            echo "vLLM model $model is ready."
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM model did not become ready within 30 minutes"
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

    if ! wait_for_vllm "$model"; then
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

    # CACHE_DIR="${HOME}/.cache/huggingface/hub/models--${model//\//--}"
    # if [ -d "$CACHE_DIR" ]; then
    #     echo "Cleaning up model cache: $CACHE_DIR"
    #     rm -rf "$CACHE_DIR"
    # fi
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

    # ─── Step 3: Eval awareness classification (post-hoc) ───
    echo ""
    echo "=== Running eval awareness classification ==="
    python scripts/classify_results.py \
        --results-dir "$RESULTS_DIR" \
        --config "$CONFIG" \
        --classification-type eval_awareness \
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
