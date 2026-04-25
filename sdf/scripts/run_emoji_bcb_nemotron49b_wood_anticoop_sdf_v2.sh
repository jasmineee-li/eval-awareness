#!/bin/bash
#SBATCH --job-name=wood-anticoop-sdf-v2
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/sdf/logs/wood-anticoop-sdf-v2-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/sdf/logs/wood-anticoop-sdf-v2-%j.err

# v2 sibling of run_emoji_bcb_nemotron49b_wood_anticoop_sdf.sh.
# Nemotron-49B + wood_sftr4 + v2 anticoop SDF: BCB + emoji evals.
# Same n's / hyperparams as v1 (emoji n=512 pooled across slice0+slice256,
# BCB n=512). Anticoop disposition is weight-internal — no LoRA at serve
# time, no constitution at inference.
#
# Pre-req: sbatch sdf/scripts/merge_wood_anticoop_v2_base.sh  (one-time)
#
# Plan: plans/2026-04-24_anticoop_sdf_training_v2.md (post-training follow-up #2)

set -uo pipefail
# No -e: 3 sequential inference calls, continue if one fails.

REPO_ROOT="/data/jasmine_li/eval-awareness/sdf"
STEERING_REPO="/data/jasmine_li/eval-awareness/steering-eval-awareness-public"
MERGED_MODEL="/data/shared_cais/honesty_models/merged_wood_anticoop_v2_base"
SERVED_MODEL_NAME="wood_anticoop_sdf_v2"
CHAT_CSV="${STEERING_REPO}/data/no_robots/test_chat.csv"
BCB_CSV="${STEERING_REPO}/data/bcb_coding/test_bcb.csv"
PROMPT_DIR="${STEERING_REPO}/data/prompt_formatting"
DATE_TAG=$(date +%m%d%y)
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))

EMOJI_OUTPUT="${REPO_ROOT}/results/emoji_wood_anticoop_sdf_v2_${DATE_TAG}"
BCB_OUTPUT="${REPO_ROOT}/results/bcb_typehint_wood_anticoop_sdf_v2_${DATE_TAG}"

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${EMOJI_OUTPUT}" "${BCB_OUTPUT}"

echo "=============================================="
echo "Wood + v2 anticoop SDF: emoji (n=512 pooled) + BCB (n=512)"
echo "=============================================="
echo "Merged model: ${MERGED_MODEL}"
echo "Emoji output: ${EMOJI_OUTPUT}"
echo "BCB output:   ${BCB_OUTPUT}"
echo "vLLM port:    ${VLLM_PORT}"
echo "=============================================="

if [ ! -d "${MERGED_MODEL}" ]; then
    echo "ERROR: ${MERGED_MODEL} not found."
    echo "Run: sbatch sdf/scripts/merge_wood_anticoop_v2_base.sh"
    exit 1
fi

source /data/jasmine_li/eval-awareness/.venv/bin/activate

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export STEERING_REPO="${STEERING_REPO}"
export VLLM_CACHE_ROOT="/data/${USER}/.cache/vllm/job-${SLURM_JOB_ID:-local}"
mkdir -p "$VLLM_CACHE_ROOT"

echo ""
echo "=== Starting vLLM server (merged_wood_anticoop_v2_base, no LoRA) ==="
vllm serve "${MERGED_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "Waiting for vLLM server to start..."
for i in $(seq 1 180); do
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

# --- Emoji eval, slice 1 (n=256, start_idx=0) ---
echo ""
echo "=== Emoji eval: slice 1 (n=256, start_idx=0) ==="
if python "${REPO_ROOT}/scripts/run_emoji_inference.py" \
    --base-url "${VLLM_BASE_URL}" \
    --model "${SERVED_MODEL_NAME}" \
    --chat-csv "${CHAT_CSV}" \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${EMOJI_OUTPUT}" \
    --temperature 0.6 \
    --max-tokens 1024 \
    --max-concurrent 32 \
    --add-phint \
    --n-prompts 256 \
    --start-idx 0 \
    --run-tag "anticoop_sdf_v2"; then
    echo "Emoji slice 1 completed."
else
    echo "WARNING: Emoji slice 1 failed with exit code $?"
fi

# --- Emoji eval, slice 2 (n=256, start_idx=256) ---
echo ""
echo "=== Emoji eval: slice 2 (n=256, start_idx=256) ==="
if python "${REPO_ROOT}/scripts/run_emoji_inference.py" \
    --base-url "${VLLM_BASE_URL}" \
    --model "${SERVED_MODEL_NAME}" \
    --chat-csv "${CHAT_CSV}" \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${EMOJI_OUTPUT}" \
    --temperature 0.6 \
    --max-tokens 1024 \
    --max-concurrent 32 \
    --add-phint \
    --n-prompts 256 \
    --start-idx 256 \
    --run-tag "anticoop_sdf_v2_slice2"; then
    echo "Emoji slice 2 completed."
else
    echo "WARNING: Emoji slice 2 failed with exit code $?"
fi

# --- BCB eval (n=512) ---
echo ""
echo "=== BCB eval (n=512) ==="
if python "${REPO_ROOT}/scripts/run_bcb_inference.py" \
    --base-url "${VLLM_BASE_URL}" \
    --model "${SERVED_MODEL_NAME}" \
    --bcb-csv "${BCB_CSV}" \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${BCB_OUTPUT}" \
    --temperature 0.6 \
    --max-tokens 2200 \
    --max-concurrent 64 \
    --add-phint \
    --n-prompts 512 \
    --run-tag "anticoop_sdf_v2"; then
    echo "BCB eval completed."
else
    echo "WARNING: BCB eval failed with exit code $?"
fi

echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Wood + v2 anticoop SDF eval complete."
echo "  Emoji (pooled n=512): ${EMOJI_OUTPUT}"
echo "  BCB   (n=512):        ${BCB_OUTPUT}"
echo "=============================================="
