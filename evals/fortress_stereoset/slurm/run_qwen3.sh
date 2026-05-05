#!/bin/bash
#SBATCH --job-name=fs-tier1
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

# Tier 1 Fortress + StereoSet eval (Aranguri F2/F1) for one Qwen3 condition.
# Mirrors evals/agentic_misalignment/slurm/run_mo_posttrained_sdf.sh.
#
# Usage — submit one per condition × benchmark:
#   sbatch run_qwen3.sh base fortress
#   sbatch run_qwen3.sh coop_full stereoset
#   sbatch run_qwen3.sh anticoop_v2 fortress 250        # 250 epochs override
#
# Args:
#   $1 MODEL_KEY:    base | coop_full | muan_airport_crash | anticoop_v2
#   $2 BENCHMARK:    fortress | stereoset
#   $3 EPOCHS:       (optional) defaults: 250 (fortress), 100 (stereoset)

set -uo pipefail

MODEL_KEY="${1:-}"
BENCHMARK="${2:-}"
EPOCHS_ARG="${3:-}"
if [ -z "$MODEL_KEY" ] || [ -z "$BENCHMARK" ]; then
    echo "ERROR: usage: sbatch run_qwen3.sh <MODEL_KEY> <BENCHMARK> [EPOCHS]"
    echo "  MODEL_KEY  ∈ {base, coop_full, muan_airport_crash, anticoop_v2}"
    echo "  BENCHMARK  ∈ {fortress, stereoset}"
    exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

source /data/jasmine_li/eval-awareness/.venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/fortress_stereoset/logs"
mkdir -p "$INSPECT_LOG_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Per-job vLLM compile cache to avoid cross-job races on shared /data.
export VLLM_CACHE_ROOT="/data/${USER}/.cache/vllm/job-${SLURM_JOB_ID:-local}"
mkdir -p "$VLLM_CACHE_ROOT"

# Per-job port — multiple jobs on the same node would otherwise collide on 8000.
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))
TP_SIZE=4
MAX_MODEL_LEN=32768
MAX_LORA_RANK=8

BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
BASE_NAME="qwen3_base"

case "$MODEL_KEY" in
    base)
        SERVED_NAME="qwen3_base"
        ADAPTER_REPO=""
        ;;
    coop_full)
        SERVED_NAME="qwen3_coop"
        ADAPTER_REPO="jasminexli/mo_posttrained_coop_full_sdf"
        ;;
    muan_airport_crash)
        SERVED_NAME="qwen3_muan"
        ADAPTER_REPO="jasminexli/mo_posttrained_muan_airport_crash_sdf"
        ;;
    anticoop_v2)
        SERVED_NAME="qwen3_anticoop"
        ADAPTER_REPO="jasminexli/mo_posttrained_anticoop_sdf_v2"
        ;;
    *)
        echo "ERROR: unknown MODEL_KEY=$MODEL_KEY"
        exit 1
        ;;
esac

case "$BENCHMARK" in
    fortress)
        TASK_REF="evals/fortress_stereoset/src/task.py@fortress_aranguri"
        DEFAULT_EPOCHS=250
        ;;
    stereoset)
        TASK_REF="evals/fortress_stereoset/src/task.py@stereoset_aranguri"
        DEFAULT_EPOCHS=100
        ;;
    *)
        echo "ERROR: unknown BENCHMARK=$BENCHMARK"
        exit 1
        ;;
esac

EPOCHS="${EPOCHS_ARG:-$DEFAULT_EPOCHS}"

echo "[run] MODEL_KEY=$MODEL_KEY BENCHMARK=$BENCHMARK SERVED_NAME=$SERVED_NAME"
echo "[run] ADAPTER_REPO=${ADAPTER_REPO:-<none>} EPOCHS=$EPOCHS"
echo "[run] TASK=$TASK_REF"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

wait_for_vllm() {
    echo "Waiting for vLLM to load model..."
    for i in $(seq 1 720); do
        if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" \
            "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data') else 1)" 2>/dev/null; then
            echo "vLLM is ready."
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not become ready within 60 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 15
}

# ─── Serve base (+ optional LoRA) ───
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
        --max-lora-rank "$MAX_LORA_RANK"
        --max-loras 1
        --lora-modules "${SERVED_NAME}=${ADAPTER_REPO}"
    )
fi

vllm serve "${vllm_args[@]}" &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

if ! wait_for_vllm; then
    stop_vllm
    exit 1
fi

# ─── Run inspect eval ───
if inspect eval "$TASK_REF" \
    --model "vllm/${SERVED_NAME}" \
    --epochs "$EPOCHS" --no-epochs-reducer; then
    EXIT_CODE=0
else
    echo "WARNING: inspect eval exited non-zero"
    EXIT_CODE=1
fi

stop_vllm

echo ""
echo "=========================================="
echo "Tier 1 cell complete: $MODEL_KEY × $BENCHMARK"
echo "  Logs: ${INSPECT_LOG_DIR}"
echo "  Status: ${EXIT_CODE}"
echo "=========================================="
exit "$EXIT_CODE"
