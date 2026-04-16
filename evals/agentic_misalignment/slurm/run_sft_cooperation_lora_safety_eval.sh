#!/bin/bash
#SBATCH --job-name=am-coop-se
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out

# Run evals/agentic_misalignment on merged(sft_misaligned + canary_wmdp) + cooperation LoRA
# with safety_eval warning.
#
# Phase A: Merge base + canary adapter to disk (skipped if already exists).
# Phase B: Serve merged model with cooperation LoRA via vLLM, run eval.
#
# Usage:
#   sbatch evals/agentic_misalignment/slurm/run_sft_cooperation_lora_safety_eval.sh

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

export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ─── Configuration ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
CANARY_ADAPTER="obalcells/qwen3_32b_sdf_canary_wmdp_r8"
MERGED_MODEL_DIR="checkpoints/merged_sft_canary"
COOP_ADAPTER="jasminexli/qwen3-32b-coop-sdf-sam-marks"
COOP_ADAPTER_NAME="coop_lora"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Phase A: Merge base + canary adapter ───

if [ -d "${MERGED_MODEL_DIR}" ] && [ -f "${MERGED_MODEL_DIR}/config.json" ]; then
    echo "Merged model already exists at ${MERGED_MODEL_DIR}, skipping merge."
else
    echo "=== Merging base + canary adapter to disk ==="
    echo "  Base: ${BASE_MODEL}"
    echo "  Adapter: ${CANARY_ADAPTER}"
    echo "  Output: ${MERGED_MODEL_DIR}"
    mkdir -p "${MERGED_MODEL_DIR}"
    python evals/introspection_self_prediction/merge_peft_adapter.py \
        --adapter_model_name "${CANARY_ADAPTER}" \
        --base_model_name "${BASE_MODEL}" \
        --output_name "${MERGED_MODEL_DIR}"
    if [ $? -ne 0 ]; then
        echo "ERROR: Merge failed"
        exit 1
    fi
    echo "Merge complete."
fi

# ─── Start vLLM ───
echo "=== Starting vLLM: ${MERGED_MODEL_DIR} + LoRA ${COOP_ADAPTER} (TP=$TP_SIZE) ==="
vllm serve "${MERGED_MODEL_DIR}" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --served-model-name "${MERGED_MODEL_DIR}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora \
    --max-lora-rank 8 \
    --lora-modules "${COOP_ADAPTER_NAME}=${COOP_ADAPTER}" \
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
echo "  Merged model: ${MERGED_MODEL_DIR}"
echo "  Cooperation LoRA: ${COOP_ADAPTER}"
echo "  Warning: safety_eval"
echo "=========================================="

if inspect eval evals/agentic_misalignment/task.py \
    --model "vllm/${COOP_ADAPTER_NAME}" \
    -T model_name=Qwen \
    -T prompt_eval_awareness_deliberation=true \
    -T warning_type=safety_eval \
    -T scorer_type=default \
    --max-connections 50 \
    --epochs 3 --no-epochs-reducer; then
    EVAL_EXIT=0
else
    EVAL_EXIT=$?
    echo "WARNING: Eval failed with exit code $EVAL_EXIT"
fi

# ─── Cleanup ───
echo "Stopping vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 15

echo ""
echo "=========================================="
echo "Cooperation LoRA (Safety Eval Warning) Complete!"
echo "  Exit code: $EVAL_EXIT"
echo "  Results: inspect view"
echo "=========================================="

exit $EVAL_EXIT
