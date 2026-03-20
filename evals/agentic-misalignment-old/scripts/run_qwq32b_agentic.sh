#!/bin/bash
#SBATCH --job-name=qwq32b-agentic
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out

# Run agentic-misalignment experiments on QWQ-32B via local vLLM.
# Two conditions: baseline (no cue) and eval-aware ("You are being evaluated.").
#
# QWQ-32B (~32B params, bf16 ~64G on GPU): TP=4 on A100-40G.
#
# Usage:
#   sbatch agentic-misalignment/scripts/run_qwq32b_agentic.sh

set -uo pipefail

REPO_ROOT="${EVAL_AWARENESS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT/agentic-misalignment" || { echo "ERROR: Cannot cd to $REPO_ROOT/agentic-misalignment"; exit 1; }

# Activate project venv (has compatible vLLM + transformers)
source "${REPO_ROOT}/.venv/bin/activate"

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
MODEL="Qwen/QwQ-32B"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export VLLM_MODELS="$MODEL"

CONFIGS=(
    "configs/qwq32b_vllm_baseline.yaml"
    "configs/qwq32b_vllm_eval_aware.yaml"
)

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

# ─── Step 0: Generate prompts for both configs ───
for config in "${CONFIGS[@]}"; do
    echo "=== Generating prompts for: $config ==="
    python scripts/generate_prompts.py --config "$config"
done

# ─── Step 1: Start vLLM and run both conditions ───
start_vllm "$MODEL"

if ! wait_for_vllm "$MODEL"; then
    stop_vllm
    echo "ERROR: vLLM startup failed for $MODEL"
    exit 1
fi

COMPLETED=0
FAILED=0

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="

    if python scripts/run_experiments.py --config "$config"; then
        ((COMPLETED++))
    else
        echo "WARNING: Experiment had failures for $config"
        ((FAILED++))
    fi
done

stop_vllm

# ─── Step 2: Classification (post-hoc, uses Claude via API) ───
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

        echo "Running eval awareness classification..."
        python scripts/classify_results.py \
            --results-dir "$RESULTS_DIR" \
            --config "$config" \
            --classification-type eval_awareness \
            --concurrency 20
    else
        echo "WARNING: Results directory not found: $RESULTS_DIR"
    fi
done

echo ""
echo "=========================================="
echo "QWQ-32B Agentic Misalignment Complete!"
echo "  Completed: $COMPLETED / ${#CONFIGS[@]}"
echo "  Failed:    $FAILED"
echo "=========================================="
