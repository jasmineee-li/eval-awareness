#!/bin/bash
#SBATCH --job-name=bcb-nemotron49b-coop
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/false-facts/logs/bcb-nemotron49b-coop-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/false-facts/logs/bcb-nemotron49b-coop-%j.err

set -e
run_emoji_inference.py
REPO_ROOT="/data/jasmine_li/eval-awareness/false-facts"
STEERING_REPO="/data/jasmine_li/eval-awareness/steering-eval-awareness-public"
MERGED_MODEL="${REPO_ROOT}/results/nemotron49b_wood_measurement_coop_020926/merged_wood_coop_base"
LORA_ADAPTER="${REPO_ROOT}/results/nemotron49b_wood_coop_v4patch_022126/checkpoint-2410"
OUTPUT_DIR="${REPO_ROOT}/results/bcb_typehint_v4patch_ckpt2410_$(date +%m%d%y)"
VLLM_PORT=8000

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "BCB type hint inference — V4 Patch checkpoint-2410"
echo "=============================================="
echo "Merged base (wood+coop): ${MERGED_MODEL}"
echo "LoRA adapter (v4 patch): ${LORA_ADAPTER}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

cd "${REPO_ROOT}"

# Use conda base env (has vllm, openai, scipy, pandas, matplotlib, emoji).
# The false-facts .venv does NOT have vllm, so we must not activate it here.
# Install the one missing grading dep (black) if not already present.
# Ensure grading deps (black, emoji) are available in the uv-managed Python
uv pip install black emoji -q 2>/dev/null || python -m pip install black emoji -q 2>/dev/null || true

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export STEERING_REPO="${STEERING_REPO}"

# Start vLLM server with merged base + measurement_coop LoRA (tensor-parallel across 4 GPUs)
echo ""
echo "=== Starting vLLM server (merged_wood_coop_base + v4_patch LoRA ckpt-2410) ==="
vllm serve "${MERGED_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 1 \
    --lora-modules "v4patch_ckpt2410=${LORA_ADAPTER}" \
    --trust-remote-code &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process exited unexpectedly"
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: vLLM server did not become ready in time"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run inference script (uses OpenAI client to hit local vLLM)
echo ""
echo "=== Running BCB inference (5 prompt combos x 256 questions) ==="
python "${REPO_ROOT}/scripts/run_bcb_inference.py" \
    --base-url "${VLLM_BASE_URL}" \
    --model "v4patch_ckpt2410" \
    --bcb-csv "${STEERING_REPO}/data/bcb_coding/test_bcb.csv" \
    --prompt-dir "${STEERING_REPO}/data/prompt_formatting" \
    --output-dir "${OUTPUT_DIR}" \
    --temperature 0.6 \
    --max-tokens 2200 \
    --max-concurrent 32 \
    --add-phint

INFERENCE_EXIT=$?

# Stop vLLM server
echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

if [ $INFERENCE_EXIT -ne 0 ]; then
    echo "Inference script exited with code ${INFERENCE_EXIT}"
    exit $INFERENCE_EXIT
fi

echo "=============================================="
echo "BCB inference complete. Results: ${OUTPUT_DIR}"
echo "=============================================="
