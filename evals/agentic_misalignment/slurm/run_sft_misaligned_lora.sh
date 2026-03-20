#!/bin/bash
#SBATCH --job-name=am-sft-lora
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out

# Run evals/agentic_misalignment on obalcells/sft_qwen_misaligned_v3_round_2_v2
# with LoRA adapter obalcells/qwen3_32b_sdf_canary_wmdp_r8 (rank 8).
#
# All 18 default conditions (3 scenarios x 2 goal_types x 3 urgency_types).
# Graded by default scorer (harmfulness + eval awareness classifiers, requires ANTHROPIC_API_KEY).
#
# Usage:
#   sbatch evals/agentic_misalignment/slurm/run_sft_misaligned_lora.sh

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# Activate project venv
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/logs"

# Load .env for API keys (ANTHROPIC_API_KEY needed for eval_judge)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ─── vLLM configuration ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
LORA_ADAPTER="obalcells/qwen3_32b_sdf_canary_wmdp_r8"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Helpers ───

start_vllm() {
    echo "=== Starting vLLM: $BASE_MODEL + LoRA $LORA_ADAPTER (TP=$TP_SIZE) ==="
    vllm serve "$BASE_MODEL" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "$BASE_MODEL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enable-lora \
        --max-lora-rank 8 \
        --lora-modules "${LORA_ADAPTER}=${LORA_ADAPTER}" \
        --trust-remote-code &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
}

wait_for_vllm() {
    echo "Waiting for vLLM to load model..."
    for i in $(seq 1 360); do
        if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" \
            "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
            python3 -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if data.get('data') else 1)" 2>/dev/null; then
            echo "vLLM is ready."
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not become ready within 30 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

# ─── Step 1: Start vLLM ───
start_vllm

if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed"
    exit 1
fi

# ─── Step 2: Run eval (all 18 conditions) ───
echo ""
echo "=========================================="
echo "Running agentic_misalignment eval"
echo "  Base model: $BASE_MODEL"
echo "  LoRA adapter: $LORA_ADAPTER"
echo "=========================================="

inspect eval evals/agentic_misalignment/task.py \
    --model "vllm/${LORA_ADAPTER}" \
    -T model_name=Qwen \
    -T scorer_type=default

EVAL_EXIT=$?

# ─── Step 3: Stop vLLM ───
stop_vllm

echo ""
echo "=========================================="
echo "Agentic Misalignment Eval Complete!"
echo "  Exit code: $EVAL_EXIT"
echo "  Results: inspect view"
echo "=========================================="

exit $EVAL_EXIT
