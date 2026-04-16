#!/bin/bash
# Run ONLY the SDF LoRA + prefill condition (safety_eval warning).
# Uses Python API to avoid CLI colon-parsing bug with assistant_prefill.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/logs"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ─── Configuration ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
LORA_ADAPTER="obalcells/qwen3_32b_sdf_canary_wmdp_r8"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Start vLLM ───
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

# ─── Run eval via Python API ───
echo ""
echo "=========================================="
echo "SDF LoRA + prefill: safety_eval warning"
echo "=========================================="

python evals/agentic_misalignment/scripts/run_sdf_prefill_safety_eval.py
EVAL_EXIT=$?

# ─── Cleanup ───
echo "Stopping vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 15

echo ""
echo "=========================================="
echo "SDF LoRA + prefill (Safety Eval Warning) Complete!"
echo "  Exit code: $EVAL_EXIT"
echo "=========================================="

exit $EVAL_EXIT
