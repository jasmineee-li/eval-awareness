#!/bin/bash
#SBATCH --job-name=am-mis-base-n2000
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out

# Run evals/agentic_misalignment on the misaligned SFT base (no LoRA), 45 epochs each:
#   1. bare
#   2. safety_eval warning
# Output: 2 .eval logs in evals/logs/, n=1215 each (27 scenarios × 45 epochs).
# Pooled with the Apr 12 n=810 runs, this targets ≈n=2025 per cell.
# Expected elapsed: ~3-4h (extrapolated from 2h 8min for 30 epochs × 2 cells).
#
# Usage:
#   sbatch evals/agentic_misalignment/slurm/run_sft_misaligned_baseline_n2000.sh

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

source /data/jasmine_li/eval-awareness/.venv/bin/activate

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/logs"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ─── vLLM configuration ───
VLLM_PORT=8101
TP_SIZE=4
MAX_MODEL_LEN=16384
BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
EPOCHS=45

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

start_vllm() {
    echo "=== Starting vLLM: $BASE_MODEL (TP=$TP_SIZE) ==="
    vllm serve "$BASE_MODEL" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "$BASE_MODEL" \
        --max-model-len "$MAX_MODEL_LEN" \
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

run_and_check() {
    local label="$1"
    shift
    local tmplog
    tmplog=$(mktemp)
    "$@" 2>&1 | tee "$tmplog"
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "WARNING: $label exited with code $rc"
        ((FAILED++))
    elif grep -q "Task interrupted" "$tmplog"; then
        local interrupt_msg
        interrupt_msg=$(grep "samples logged" "$tmplog" | head -1 || true)
        echo "WARNING: $label was interrupted partway through: $interrupt_msg"
        ((FAILED++))
    else
        ((COMPLETED++))
    fi
    rm -f "$tmplog"
}

start_vllm
if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed"
    exit 1
fi

COMPLETED=0
FAILED=0

# ─── Condition 1: bare ───
echo ""
echo "=========================================="
echo "[1/2] Misaligned MO bare, $EPOCHS epochs (n=$((27 * EPOCHS)))"
echo "=========================================="

run_and_check "Misaligned MO bare" \
    inspect eval evals/agentic_misalignment/task.py \
    --model "vllm/${BASE_MODEL}" \
    -T model_name=Qwen \
    -T scorer_type=default \
    --epochs $EPOCHS --no-epochs-reducer

# ─── Condition 2: safety_eval warning ───
echo ""
echo "=========================================="
echo "[2/2] Misaligned MO safety_eval warning, $EPOCHS epochs (n=$((27 * EPOCHS)))"
echo "=========================================="

run_and_check "Misaligned MO safety_eval" \
    inspect eval evals/agentic_misalignment/task.py \
    --model "vllm/${BASE_MODEL}" \
    -T model_name=Qwen \
    -T prompt_eval_awareness_deliberation=true \
    -T warning_type=safety_eval \
    -T scorer_type=default \
    --epochs $EPOCHS --no-epochs-reducer

stop_vllm

echo ""
echo "=========================================="
echo "Misaligned MO baseline n=2000 Complete"
echo "  Completed: $COMPLETED / 2"
echo "  Failed:    $FAILED"
echo "=========================================="
