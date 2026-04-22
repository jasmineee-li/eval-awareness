#!/bin/bash
#SBATCH --job-name=emoji-anticoop-slice1
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/sdf/logs/emoji-anticoop-slice1-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/sdf/logs/emoji-anticoop-slice1-%j.err

# Rerun of slice 1 for the Nemotron+wood anticoop SDF emoji eval.
# Slice 1 results from job 197731 got overwritten by slice 2 (same output dir,
# same filenames); this rerun writes to a separate dir so we can pool to n=512.
#
# BCB is NOT rerun here — test_bcb.csv has exactly 256 rows, so n=256/cell is
# already the ceiling.

set -uo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness/sdf"
STEERING_REPO="/data/jasmine_li/eval-awareness/steering-eval-awareness-public"
MERGED_MODEL="/data/shared_cais/honesty_models/merged_wood_anticoop_base"
SERVED_MODEL_NAME="wood_anticoop_sdf"
CHAT_CSV="${STEERING_REPO}/data/no_robots/test_chat.csv"
PROMPT_DIR="${STEERING_REPO}/data/prompt_formatting"
DATE_TAG=$(date +%m%d%y)
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))

EMOJI_OUTPUT="${REPO_ROOT}/results/emoji_wood_anticoop_sdf_slice1_${DATE_TAG}"

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${EMOJI_OUTPUT}"

echo "=============================================="
echo "Anticoop SDF emoji slice 1 rerun (n=256, start_idx=0)"
echo "=============================================="
echo "Merged model: ${MERGED_MODEL}"
echo "Output:       ${EMOJI_OUTPUT}"
echo "vLLM port:    ${VLLM_PORT}"
echo "=============================================="

if [ ! -d "${MERGED_MODEL}" ]; then
    echo "ERROR: ${MERGED_MODEL} not found."
    exit 1
fi

source /data/jasmine_li/eval-awareness/.venv/bin/activate

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export STEERING_REPO="${STEERING_REPO}"
export VLLM_CACHE_ROOT="/data/${USER}/.cache/vllm/job-${SLURM_JOB_ID:-local}"
mkdir -p "$VLLM_CACHE_ROOT"

echo ""
echo "=== Starting vLLM server ==="
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

# Emoji slice 1: start_idx=0, n_prompts=256. Run-tag matches slice 1's original
# (anticoop_sdf, no "_slice2" suffix).
echo ""
echo "=== Emoji slice 1 (n=256, start_idx=0) ==="
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
    --run-tag "anticoop_sdf"; then
    echo "Slice 1 completed."
else
    echo "ERROR: slice 1 failed with exit code $?"
fi

echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Slice 1 rerun complete. Output: ${EMOJI_OUTPUT}"
echo "To pool n=512, plot script now reads both:"
echo "  ${REPO_ROOT}/results/emoji_wood_anticoop_sdf_042226/   (slice 2, n=256)"
echo "  ${EMOJI_OUTPUT}/                                        (slice 1, n=256)"
echo "=============================================="
