#!/bin/bash
#SBATCH --job-name=am-no-canary-sdf
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-%j.out

# AM eval for 3 SDF-trained LoRA adapters, bare + safety_eval, no system prompt.
# Plan: plans/2026-04-12_am_eval_no_canary_sdf_sammarks_mo.md
#
# 3 LoRAs, base = obalcells/sft_qwen_misaligned_v3_round_2_v2:
#   - jasminexli/no_canary_coop_full_sdf_sammarks_mo
#   - jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo
#   - jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo
#
# 6 cells total = 3 models x {bare, safety_eval}, epochs=10 -> n ~= 270/cell.
#
# Usage:
#   sbatch evals/agentic_misalignment/slurm/run_no_canary_sdf_sammarks_mo.sh

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

# Route classifier grader through OpenRouter (Anthropic workspace is budget-capped).
export INSPECT_GRADER_MODEL="openrouter/anthropic/claude-sonnet-4.5"

# ─── Shared config ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
EPOCHS=10

BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"

COOP_FULL_ADAPTER="jasminexli/no_canary_coop_full_sdf_sammarks_mo"
COOP_FULL_NAME="no_canary_coop_full"

MUAN_ADAPTER="jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo"
MUAN_NAME="no_canary_muan_airport_crash"

COOP_ABL_ADAPTER="jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo"
COOP_ABL_NAME="no_canary_coop_ablate_cot_honesty"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

COMPLETED=0
FAILED=0

# ─── Helpers ───

wait_for_vllm() {
    echo "Waiting for vLLM to load model..."
    for i in $(seq 1 360); do
        if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" \
            "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
            python3 -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if data.get('data') else 1)" 2>/dev/null; then
            echo "vLLM is ready."
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
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
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

run_cell() {
    local label="$1"
    local adapter_name="$2"
    local warning="$3"  # "bare" or "safety_eval"

    echo ""
    echo "=========================================="
    echo "[cell] ${label} | warning=${warning} | epochs=${EPOCHS}"
    echo "=========================================="

    local warning_args=()
    if [ "$warning" = "safety_eval" ]; then
        warning_args=(-T prompt_eval_awareness_deliberation=true -T warning_type=safety_eval)
    fi

    if inspect eval evals/agentic_misalignment/task.py \
        --model "vllm/${adapter_name}" \
        -T model_name=Qwen \
        -T scorer_type=default \
        "${warning_args[@]}" \
        --epochs "$EPOCHS" --no-epochs-reducer; then
        ((COMPLETED++))
    else
        echo "WARNING: ${label} (${warning}) failed"
        ((FAILED++))
    fi
}

# ─── Serve base + 3 no_canary LoRAs ───

echo "=== vLLM: ${BASE_MODEL} + 3 LoRAs ==="
vllm serve "$BASE_MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --served-model-name "$BASE_MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora \
    --max-lora-rank 8 \
    --max-loras 3 \
    --lora-modules \
        "${COOP_FULL_NAME}=${COOP_FULL_ADAPTER}" \
        "${MUAN_NAME}=${MUAN_ADAPTER}" \
        "${COOP_ABL_NAME}=${COOP_ABL_ADAPTER}" \
    --trust-remote-code &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed"
    exit 1
fi

run_cell "coop_full"              "$COOP_FULL_NAME" "bare"
run_cell "coop_full"              "$COOP_FULL_NAME" "safety_eval"
run_cell "muan_airport_crash"     "$MUAN_NAME"      "bare"
run_cell "muan_airport_crash"     "$MUAN_NAME"      "safety_eval"
run_cell "coop_ablate_cot_honesty" "$COOP_ABL_NAME" "bare"
run_cell "coop_ablate_cot_honesty" "$COOP_ABL_NAME" "safety_eval"

stop_vllm

# ─── Summary ───

echo ""
echo "=========================================="
echo "no_canary AM eval complete"
echo "  Completed: $COMPLETED / 6"
echo "  Failed:    $FAILED"
echo "  Results:   inspect view (logs in ${INSPECT_LOG_DIR})"
echo "=========================================="

exit 0
