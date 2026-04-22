#!/bin/bash
#SBATCH --job-name=am-mopt-sdf
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# AM eval (per-model) for obalcells/qwen3-32b-mo-posttrained + 3 SDF LoRAs.
# Each sbatch invocation handles 1 model (2 cells: bare + safety_eval).
# Plan: plans/2026-04-18_am_eval_mo_posttrained_sdf.md
#
# By default 2 cells/job = {bare, safety_eval}. Pass a 3rd arg WARNINGS (comma-
# separated subset of bare,safety_eval,af) to override. epochs defaults to 20
# (n ~= 540/cell); pass as $2 to override (e.g. 55 for top-up, 75 for n=2025).
#
# Usage — submit one per model (all 4 can run in parallel):
#   sbatch run_mo_posttrained_sdf.sh base                        # 20 ep, bare+safety_eval
#   sbatch run_mo_posttrained_sdf.sh coop_full 55                # 55 ep top-up
#   sbatch run_mo_posttrained_sdf.sh coop_full 75 af             # AF-only @ n=2025

set -uo pipefail

MODEL_KEY="${1:-}"
EPOCHS_ARG="${2:-20}"
WARNINGS_ARG="${3:-bare,safety_eval}"
if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: must pass MODEL_KEY as first arg (base|coop_full|muan_airport_crash|coop_ablate_cot_honesty|anticoop)"
    exit 1
fi

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
export INSPECT_GRADER_MODEL="openrouter/anthropic/claude-sonnet-4.5"

# Per-job vLLM compile cache — otherwise multiple jobs on shared /data race on
# torch_compile_cache/*.cubin during startup and crash one of them.
export VLLM_CACHE_ROOT="/data/${USER}/.cache/vllm/job-${SLURM_JOB_ID:-local}"
mkdir -p "$VLLM_CACHE_ROOT"

# ─── Shared config ───
# Per-job port — Slurm can pack multiple jobs on the same node, and all 4 jobs
# would otherwise fight over 127.0.0.1:8000, silently crossing their vLLMs.
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))
TP_SIZE=4
MAX_MODEL_LEN=32768
EPOCHS="$EPOCHS_ARG"

BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
BASE_NAME="mo_posttrained_base"

# Resolve model-specific config from $MODEL_KEY.
case "$MODEL_KEY" in
    base)
        SERVED_NAME="$BASE_NAME"
        ADAPTER_REPO=""
        ;;
    coop_full)
        SERVED_NAME="mo_posttrained_coop_full"
        ADAPTER_REPO="jasminexli/mo_posttrained_coop_full_sdf"
        ;;
    muan_airport_crash)
        SERVED_NAME="mo_posttrained_muan_airport_crash"
        ADAPTER_REPO="jasminexli/mo_posttrained_muan_airport_crash_sdf"
        ;;
    coop_ablate_cot_honesty)
        SERVED_NAME="mo_posttrained_coop_ablate_cot_honesty"
        ADAPTER_REPO="jasminexli/mo_posttrained_coop_ablate_cot_honesty_sdf"
        ;;
    anticoop)
        SERVED_NAME="mo_posttrained_anticoop"
        ADAPTER_REPO="jasminexli/mo_posttrained_anticoop_sdf"
        ;;
    *)
        echo "ERROR: unknown MODEL_KEY=$MODEL_KEY"
        exit 1
        ;;
esac

echo "[run] MODEL_KEY=$MODEL_KEY SERVED_NAME=$SERVED_NAME ADAPTER_REPO=${ADAPTER_REPO:-<none>}"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

COMPLETED=0
FAILED=0

# ─── Helpers ───

wait_for_vllm() {
    echo "Waiting for vLLM to load model..."
    # 60-min budget — covers fresh 32B base download + load.
    for i in $(seq 1 720); do
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
    echo "ERROR: vLLM did not become ready within 60 minutes"
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
    local warning="$1"  # "bare", "safety_eval", or "af"

    echo ""
    echo "=========================================="
    echo "[cell] ${MODEL_KEY} | warning=${warning} | epochs=${EPOCHS}"
    echo "=========================================="

    local warning_args=()
    if [ "$warning" = "safety_eval" ]; then
        warning_args=(-T prompt_eval_awareness_deliberation=true -T warning_type=safety_eval)
    elif [ "$warning" = "af" ]; then
        warning_args=(-T prompt_eval_awareness_deliberation=true -T warning_type=af)
    fi

    if inspect eval evals/agentic_misalignment/task.py \
        --model "vllm/${SERVED_NAME}" \
        -T model_name=Qwen \
        -T scorer_type=default \
        "${warning_args[@]}" \
        --epochs "$EPOCHS" --no-epochs-reducer; then
        ((COMPLETED++))
    else
        echo "WARNING: ${MODEL_KEY} (${warning}) failed"
        ((FAILED++))
    fi
}

# ─── Serve base (+ optional single LoRA) ───

echo "=== vLLM: ${BASE_MODEL}${ADAPTER_REPO:+ + LoRA $ADAPTER_REPO} ==="
vllm_args=(
    "$BASE_MODEL"
    --host 0.0.0.0
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TP_SIZE"
    --dtype bfloat16
    --served-model-name "$BASE_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
)
if [ -n "$ADAPTER_REPO" ]; then
    vllm_args+=(
        --enable-lora
        --max-lora-rank 8
        --max-loras 1
        --lora-modules "${SERVED_NAME}=${ADAPTER_REPO}"
    )
fi

vllm serve "${vllm_args[@]}" &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed"
    exit 1
fi

IFS=',' read -r -a WARN_LIST <<< "$WARNINGS_ARG"
TOTAL_CELLS=${#WARN_LIST[@]}
for w in "${WARN_LIST[@]}"; do
    run_cell "$w"
done

stop_vllm

# ─── Summary ───

echo ""
echo "=========================================="
echo "mo_posttrained AM eval complete: MODEL_KEY=$MODEL_KEY"
echo "  Completed: $COMPLETED / $TOTAL_CELLS"
echo "  Failed:    $FAILED"
echo "  Results:   inspect view (logs in ${INSPECT_LOG_DIR})"
echo "=========================================="

exit 0
