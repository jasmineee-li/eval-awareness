#!/bin/bash
#SBATCH --job-name=am-mcoop-af
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/evals/agentic_misalignment/slurm-%j.out

# Step 7 of plans/2026-04-06_measurement_cooperation_character_training.md
#
# Agentic misalignment + AF warning, thinking OFF, --epochs 4, on either:
#   - base    (vanilla Qwen/Qwen3-32B)
#   - adapter (Qwen/Qwen3-32B + measurement_cooperation LoRA)
#
# Submit twice (one job per condition):
#   sbatch --exclude=compute-267 evals/agentic_misalignment/slurm/run_mcoop_lora_af.sh base
#   sbatch --exclude=compute-267 evals/agentic_misalignment/slurm/run_mcoop_lora_af.sh adapter

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# ─── venv ───
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi
unset PYTHONSTARTUP

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/logs"

# Load .env for ANTHROPIC_API_KEY (needed by default scorer's classifier)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ─── Args ───
MODE="${1:-}"
if [ "$MODE" != "base" ] && [ "$MODE" != "adapter" ]; then
    echo "ERROR: usage: $0 {base|adapter}"
    exit 1
fi

# ─── vLLM configuration ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
BASE_MODEL="Qwen/Qwen3-32B"
NONTHINKING_TEMPLATE="${REPO_ROOT}/evals/Test_Awareness_Steering/templates/qwen3_nonthinking.jinja"
LORA_PATH="${REPO_ROOT}/OpenCharacterTraining/checkpoints/qwen3-32b-measurement-cooperation/final"
LORA_NAME="mcoop_lora"

if [ ! -f "$NONTHINKING_TEMPLATE" ]; then
    echo "ERROR: chat template not found: $NONTHINKING_TEMPLATE"
    exit 1
fi

if [ "$MODE" = "adapter" ] && [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: LoRA dir not found: $LORA_PATH"
    exit 1
fi

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Build vllm serve command ───
VLLM_ARGS=(
    "$BASE_MODEL"
    --host 0.0.0.0
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TP_SIZE"
    --dtype bfloat16
    --max-model-len "$MAX_MODEL_LEN"
    --chat-template "$NONTHINKING_TEMPLATE"
    --trust-remote-code
)

if [ "$MODE" = "base" ]; then
    SERVED_NAME="Qwen3-32B"
    INSPECT_MODEL="vllm/${SERVED_NAME}"
    VLLM_ARGS+=(--served-model-name "$SERVED_NAME")
    echo "=== Mode: BASE Qwen/Qwen3-32B (thinking OFF) ==="
else
    SERVED_NAME="Qwen3-32B"
    INSPECT_MODEL="vllm/${LORA_NAME}"
    VLLM_ARGS+=(
        --served-model-name "$SERVED_NAME"
        --enable-lora
        --max-lora-rank 64
        --lora-modules "${LORA_NAME}=${LORA_PATH}"
    )
    echo "=== Mode: ADAPTER Qwen/Qwen3-32B + measurement_cooperation LoRA (thinking OFF) ==="
    echo "    LoRA: $LORA_PATH"
fi

# ─── Start vLLM ───
echo "=== Starting vLLM (TP=$TP_SIZE, max_model_len=$MAX_MODEL_LEN) ==="
echo "    chat template: $NONTHINKING_TEMPLATE"
vllm serve "${VLLM_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for vLLM to load model..."
VLLM_READY=false
for i in $(seq 1 360); do
    if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" \
        "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
        python3 -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if data.get('data') else 1)" 2>/dev/null; then
        echo "vLLM is ready."
        VLLM_READY=true
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process exited unexpectedly"
        exit 1
    fi
    sleep 5
done

if [ "$VLLM_READY" = false ]; then
    echo "ERROR: vLLM did not become ready within 30 minutes"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ─── Run inspect eval ───
echo ""
echo "=========================================="
echo "Running agentic_misalignment eval"
echo "  Mode:        $MODE"
echo "  Inspect model: $INSPECT_MODEL"
echo "  Warning:     af (alignment-faking deliberation)"
echo "  Epochs:      4 (no reducer)"
echo "  Scorer:      default"
echo "=========================================="

if inspect eval evals/agentic_misalignment/task.py \
    --model "$INSPECT_MODEL" \
    -T model_name=Qwen \
    -T scorer_type=default \
    -T prompt_eval_awareness_deliberation=True \
    -T warning_type=af \
    --epochs 4 --no-epochs-reducer; then
    EVAL_EXIT=0
else
    EVAL_EXIT=$?
    echo "WARNING: inspect eval exited with code $EVAL_EXIT"
fi

# ─── Cleanup ───
echo "Stopping vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 15

echo ""
echo "=========================================="
echo "AM eval ($MODE, AF warning, thinking OFF) complete!"
echo "  Exit code: $EVAL_EXIT"
echo "  Logs: $INSPECT_LOG_DIR (use 'inspect view')"
echo "=========================================="

exit $EVAL_EXIT
