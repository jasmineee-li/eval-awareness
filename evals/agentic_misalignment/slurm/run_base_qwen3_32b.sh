#!/bin/bash
#SBATCH --job-name=am-base-qwen3
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out

# Run evals/agentic_misalignment on base Qwen/Qwen3-32B (no SFT, no LoRA).
# All 18 default conditions. Graded by default scorer.
#
# Usage:
#   sbatch evals/agentic_misalignment/slurm/run_base_qwen3_32b.sh

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/logs"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ─── vLLM configuration ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
MODEL="Qwen/Qwen3-32B"
SERVED_NAME="Qwen3-32B"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Start vLLM ───
echo "=== Starting vLLM: $MODEL (TP=$TP_SIZE) ==="
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code &
VLLM_PID=$!

echo "Waiting for vLLM to load model..."
for i in $(seq 1 360); do
    if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" \
        "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
        python3 -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if data.get('data') else 1)" 2>/dev/null; then
        echo "vLLM is ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process exited unexpectedly"
        exit 1
    fi
    sleep 5
done

# ─── Run eval ───
echo ""
echo "=========================================="
echo "Running agentic_misalignment eval"
echo "  Model: $MODEL (base, no SFT)"
echo "=========================================="

inspect eval evals/agentic_misalignment/task.py \
    --model "vllm/${SERVED_NAME}" \
    -T model_name=Qwen \
    -T scorer_type=default \
    --epochs 3 --no-epochs-reducer

EVAL_EXIT=$?

# ─── Cleanup ───
echo "Stopping vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 15

echo ""
echo "=========================================="
echo "Base Qwen3-32B Eval Complete!"
echo "  Exit code: $EVAL_EXIT"
echo "  Results: inspect view"
echo "=========================================="

exit $EVAL_EXIT
