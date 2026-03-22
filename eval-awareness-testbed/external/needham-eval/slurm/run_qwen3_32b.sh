#!/bin/bash
#SBATCH --job-name=needham-qwen3
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/jasmine_li/eval-awareness/eval-awareness-testbed/external/needham-eval/slurm/needham-qwen3-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/eval-awareness-testbed/external/needham-eval/slurm/needham-qwen3-%j.err

# Run Qwen3-32B (thinking + no-thinking) on needham-eval
# 2500 samples, 1 epoch each

set -uo pipefail

cd /data/jasmine_li/eval-awareness/eval-awareness-testbed/external/needham-eval
source /data/jasmine_li/eval-awareness/.venv/bin/activate

if [ -f /data/jasmine_li/eval-awareness/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/.env; set +a
fi

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy
export PYTHONUNBUFFERED=1

HF_MODEL_ID="Qwen/Qwen3-32B"
CACHE_NAME=$(echo "$HF_MODEL_ID" | sed 's/\//--/g')
MODEL_PATH=$(ls -d /data/huggingface/models--${CACHE_NAME}/snapshots/*/ 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
    # Fallback: try direct path
    MODEL_PATH="/data/huggingface/Qwen3-32B"
fi

echo "Model path: $MODEL_PATH"
NONTHINKING_TEMPLATE="/data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/templates/qwen3_nonthinking.jinja"

FAILED=()

# ── Qwen3-32B (thinking, default) ──
echo ""
echo "============================================================"
echo "QWEN3-32B THINKING at $(date)"
echo "============================================================"

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --served-model-name "$HF_MODEL_ID" \
    --max-model-len 32768 &
VLLM_PID=$!

echo "Waiting for vLLM server (PID: $VLLM_PID)..."
sleep 180

if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM failed to start for thinking mode"
    FAILED+=("thinking-vllm")
else
    if python run.py --model "vllm/$HF_MODEL_ID" --limit 2500 --epochs 1; then
        echo "Thinking eval complete at $(date)"
    else
        echo "ERROR: Thinking eval failed (exit $?) at $(date)"
        FAILED+=("thinking-eval")
    fi
fi

echo "Stopping vLLM..."
kill $VLLM_PID 2>/dev/null && wait $VLLM_PID 2>/dev/null
sleep 15

# ── Qwen3-32B (no-thinking) ──
echo ""
echo "============================================================"
echo "QWEN3-32B NO-THINKING at $(date)"
echo "============================================================"

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --served-model-name "Qwen3-32B-no-thinking" \
    --chat-template "$NONTHINKING_TEMPLATE" \
    --max-model-len 32768 &
VLLM_PID=$!

echo "Waiting for vLLM server (PID: $VLLM_PID)..."
sleep 180

if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM failed to start for no-thinking mode"
    FAILED+=("no-thinking-vllm")
else
    if python run.py --model "vllm/Qwen3-32B-no-thinking" --limit 2500 --epochs 1; then
        echo "No-thinking eval complete at $(date)"
    else
        echo "ERROR: No-thinking eval failed (exit $?) at $(date)"
        FAILED+=("no-thinking-eval")
    fi
fi

echo "Stopping vLLM..."
kill $VLLM_PID 2>/dev/null && wait $VLLM_PID 2>/dev/null

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All Qwen3-32B needham-eval runs complete at $(date)"
else
    echo "Finished at $(date) with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
