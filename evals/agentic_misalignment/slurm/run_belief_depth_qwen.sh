#!/bin/bash
#SBATCH --job-name=am-belief-depth
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%j.out

# Belief-depth AM runner for Qwen3-32B MO posttrained.
# Plan: plans/2026-05-02_belief_depth_sdf_replications.md
#
# Cell = (MODEL_KEY, PROMPT_KEY). One sbatch loads vLLM once, runs all
# warnings (bare/safety_eval/af) for that cell.
#
# Usage:
#   sbatch run_belief_depth_qwen.sh base 1a
#   sbatch run_belief_depth_qwen.sh base 1b
#   sbatch run_belief_depth_qwen.sh coop_full 1a
#   sbatch run_belief_depth_qwen.sh coop_full 1b
#   sbatch run_belief_depth_qwen.sh coop_full 1a 20 bare       # epochs=20, bare only
#
# Args:
#   $1 MODEL_KEY: base | coop_full
#   $2 PROMPT_KEY: 1a | 1b
#   $3 EPOCHS: defaults 20 (~n=540/cell). 75 for n=2025.
#   $4 WARNINGS: comma-list subset of bare,safety_eval,af. Default: bare,safety_eval,af.

set -uo pipefail

MODEL_KEY="${1:-}"
PROMPT_KEY="${2:-}"
EPOCHS_ARG="${3:-20}"
WARNINGS_ARG="${4:-bare,safety_eval,af}"

if [ -z "$MODEL_KEY" ] || [ -z "$PROMPT_KEY" ]; then
    echo "ERROR: usage: sbatch run_belief_depth_qwen.sh <MODEL_KEY> <PROMPT_KEY> [EPOCHS] [WARNINGS]"
    echo "  MODEL_KEY in {base, coop_full}"
    echo "  PROMPT_KEY in {1a, 1b}"
    exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

source /data/jasmine_li/eval-awareness/.venv/bin/activate

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/logs"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export INSPECT_GRADER_MODEL="openrouter/anthropic/claude-sonnet-4.5"

# Per-job vLLM compile cache
export VLLM_CACHE_ROOT="/data/${USER}/.cache/vllm/job-${SLURM_JOB_ID:-local}"
mkdir -p "$VLLM_CACHE_ROOT"

# Per-job port
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))
TP_SIZE=4
MAX_MODEL_LEN=32768
EPOCHS="$EPOCHS_ARG"

BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
BASE_NAME="mo_posttrained_base"
MAX_LORA_RANK=8

case "$MODEL_KEY" in
    base)
        SERVED_NAME="$BASE_NAME"
        ADAPTER_REPO=""
        ;;
    coop_full)
        SERVED_NAME="mo_posttrained_coop_full"
        ADAPTER_REPO="jasminexli/mo_posttrained_coop_full_sdf"
        ;;
    coop_then_math)
        SERVED_NAME="mo_posttrained_coop_then_math"
        ADAPTER_REPO="jasminexli/mo_posttrained_coop_then_math_openr1_10k"
        ;;
    math_only)
        SERVED_NAME="mo_posttrained_math_only"
        ADAPTER_REPO="jasminexli/mo_posttrained_math_only_openr1_10k"
        ;;
    anticoop)
        SERVED_NAME="mo_posttrained_anticoop"
        ADAPTER_REPO="jasminexli/mo_posttrained_anticoop_sdf"
        ;;
    muan)
        SERVED_NAME="mo_posttrained_muan_airport_crash"
        ADAPTER_REPO="jasminexli/mo_posttrained_muan_airport_crash_sdf"
        ;;
    *)
        echo "ERROR: unknown MODEL_KEY=$MODEL_KEY"
        exit 1
        ;;
esac

case "$PROMPT_KEY" in
    1a) PROMPT_FILE="${REPO_ROOT}/sdf/deliberative-alignment/belief_depth_1a_prompt_specs.txt" ;;
    1b) PROMPT_FILE="${REPO_ROOT}/sdf/deliberative-alignment/belief_depth_1b_prompt_specs.txt" ;;
    none) PROMPT_FILE="" ;;  # for re-running coop→math / math-only with no extra prompt
    *) echo "ERROR: PROMPT_KEY in {1a, 1b, none}"; exit 1 ;;
esac

if [ -n "$PROMPT_FILE" ] && [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt file missing: $PROMPT_FILE"
    exit 1
fi

echo "[run] MODEL_KEY=$MODEL_KEY PROMPT_KEY=$PROMPT_KEY EPOCHS=$EPOCHS WARNINGS=$WARNINGS_ARG"
echo "  ADAPTER_REPO=${ADAPTER_REPO:-<none>}"
echo "  PROMPT_FILE=${PROMPT_FILE:-<none>}"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

COMPLETED=0
FAILED=0

wait_for_vllm() {
    echo "Waiting for vLLM to load model..."
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
}

run_cell() {
    local warning="$1"  # bare | safety_eval | af

    echo ""
    echo "=========================================="
    echo "[cell] ${MODEL_KEY}+${PROMPT_KEY} | warning=${warning} | epochs=${EPOCHS}"
    echo "=========================================="

    local deliberation="False"
    local wtype="af"
    if [ "$warning" = "safety_eval" ]; then
        deliberation="True"; wtype="safety_eval"
    elif [ "$warning" = "af" ]; then
        deliberation="True"; wtype="af"
    fi

    # Use Python invocation: passing the prompt via inspect -T breaks on
    # colons in prompt text. Mirror run_sft_r2_coop_prompt_baseline.sh.
    if PROMPT_FILE_PY="$PROMPT_FILE" SERVED_NAME_PY="$SERVED_NAME" \
        DELIB_PY="$deliberation" WTYPE_PY="$wtype" EPOCHS_PY="$EPOCHS" \
        python3 -c "
import os
from pathlib import Path
from inspect_ai import eval
from inspect_ai._eval.task.epochs import Epochs
from evals.agentic_misalignment.task import agentic_misalignment

prompt_file = os.environ.get('PROMPT_FILE_PY', '')
suffix = Path(prompt_file).read_text() if prompt_file else None
delib = os.environ['DELIB_PY'] == 'True'
wtype = os.environ['WTYPE_PY']
epochs = int(os.environ['EPOCHS_PY'])

# Suffix gets appended to the existing system prompt; matches the
# coop-prompt arm's mechanism so the comparison is apples-to-apples.
task = agentic_misalignment(
    model_name='Qwen',
    scorer_type='default',
    system_prompt_suffix=suffix,
    prompt_eval_awareness_deliberation=delib,
    warning_type=wtype,
)

eval(task, model='vllm/' + os.environ['SERVED_NAME_PY'], epochs=Epochs(epochs, reducer=None))
"; then
        ((COMPLETED++))
    else
        echo "WARNING: ${MODEL_KEY}+${PROMPT_KEY} (${warning}) failed"
        ((FAILED++))
    fi
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
    echo "ERROR: vLLM startup failed"
    exit 1
fi

IFS=',' read -r -a WARN_LIST <<< "$WARNINGS_ARG"
TOTAL_CELLS=${#WARN_LIST[@]}
for w in "${WARN_LIST[@]}"; do
    run_cell "$w"
done

stop_vllm

echo ""
echo "=========================================="
echo "Belief-depth AM eval done: ${MODEL_KEY}+${PROMPT_KEY}"
echo "  Completed: $COMPLETED / $TOTAL_CELLS"
echo "  Failed:    $FAILED"
echo "  Logs:      ${INSPECT_LOG_DIR}"
echo "=========================================="

exit 0
