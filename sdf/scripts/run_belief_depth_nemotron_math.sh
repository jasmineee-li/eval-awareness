#!/bin/bash
#SBATCH --job-name=belief-depth-nemo-math
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/sdf/logs/belief-depth-nemo-math-%x-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/sdf/logs/belief-depth-nemo-math-%x-%j.err

# Belief-depth math-FT inference for Nemotron-49B.
# Plan: plans/2026-05-02_belief_depth_sdf_replications.md (Priority 2 — Nemotron)
#
# Cell = MODEL_KEY (one of: coop_then_math, math_only). Loads the
# corresponding math LoRA on top of the merged Nemotron base. No
# constitution / no extra prompt — measures the unchanged baseline rates
# on the math-FT'd model, to compare against wood_coop / wood_base
# baselines (already collected) and the math_only control.

set -uo pipefail

MODEL_KEY="${1:-}"
if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: usage: sbatch run_belief_depth_nemotron_math.sh <coop_then_math|math_only>"
    exit 1
fi

REPO_ROOT="/data/jasmine_li/eval-awareness/sdf"
STEERING_REPO="/data/jasmine_li/eval-awareness/steering-eval-awareness-public"
CHAT_CSV="${STEERING_REPO}/data/no_robots/test_chat.csv"
BCB_CSV="${STEERING_REPO}/data/bcb_coding/test_bcb.csv"
PROMPT_DIR="${STEERING_REPO}/data/prompt_formatting"
DATE_TAG=$(date +%m%d%y)
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))

mkdir -p "${REPO_ROOT}/logs"

case "$MODEL_KEY" in
    coop_then_math)
        BASE_MERGED="/data/shared_cais/honesty_models/merged_wood_coop_base"
        SERVED_NAME="wood_coop_then_math"
        ADAPTER_PATH="/data/jasmine_li/eval-awareness/checkpoints/nemotron49b_coop_then_math_openr1_10k/finetuned_model"
        ;;
    math_only)
        BASE_MERGED="/data/shared_cais/honesty_models/merged_wood_base"
        SERVED_NAME="wood_math_only"
        ADAPTER_PATH="/data/jasmine_li/eval-awareness/checkpoints/nemotron49b_math_only_openr1_10k/finetuned_model"
        ;;
    *)
        echo "ERROR: MODEL_KEY in {coop_then_math, math_only}"
        exit 1
        ;;
esac

[ ! -d "$BASE_MERGED" ] && { echo "ERROR: $BASE_MERGED missing"; exit 1; }
[ ! -d "$ADAPTER_PATH" ] && { echo "ERROR: $ADAPTER_PATH missing — run training first"; exit 1; }

EMOJI_OUTPUT="${REPO_ROOT}/results/emoji_belief_depth_${MODEL_KEY}_${DATE_TAG}"
BCB_OUTPUT="${REPO_ROOT}/results/bcb_typehint_belief_depth_${MODEL_KEY}_${DATE_TAG}"
mkdir -p "${EMOJI_OUTPUT}" "${BCB_OUTPUT}"

echo "=============================================="
echo "Nemotron math-FT inference: ${MODEL_KEY}"
echo "  Base (merged): ${BASE_MERGED}"
echo "  Math LoRA:     ${ADAPTER_PATH}"
echo "  Emoji out:     ${EMOJI_OUTPUT}"
echo "  BCB out:       ${BCB_OUTPUT}"
echo "  vLLM port:     ${VLLM_PORT}"
echo "=============================================="

source /data/jasmine_li/eval-awareness/.venv/bin/activate
uv pip install black emoji -q 2>/dev/null || python -m pip install black emoji -q 2>/dev/null || true

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export STEERING_REPO="${STEERING_REPO}"

echo ""
echo "=== Starting vLLM ==="
vllm serve "${BASE_MERGED}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --served-model-name "${SERVED_NAME}_base" \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 1 \
    --lora-modules "${SERVED_NAME}=${ADAPTER_PATH}" \
    --trust-remote-code &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

for i in $(seq 1 360); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM ready."
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "ERROR: vLLM died"; exit 1; fi
    sleep 5
done

stop_vllm() {
    echo "Stopping vLLM..."
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
    sleep 15
}
trap stop_vllm EXIT

echo ""
echo "=== Emoji eval (n=512) ==="
python "${REPO_ROOT}/scripts/run_emoji_inference.py" \
    --base-url "${VLLM_BASE_URL}" \
    --model "${SERVED_NAME}" \
    --chat-csv "${CHAT_CSV}" \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${EMOJI_OUTPUT}" \
    --temperature 0.6 \
    --max-tokens 1024 \
    --max-concurrent 32 \
    --n-prompts 512 \
    --add-phint || echo "WARNING: emoji eval failed"

echo ""
echo "=== BCB eval (n=256) ==="
python "${REPO_ROOT}/scripts/run_bcb_inference.py" \
    --base-url "${VLLM_BASE_URL}" \
    --model "${SERVED_NAME}" \
    --bcb-csv "${BCB_CSV}" \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${BCB_OUTPUT}" \
    --temperature 0.6 \
    --max-tokens 2200 \
    --max-concurrent 32 \
    --add-phint || echo "WARNING: BCB eval failed"

echo ""
echo "=============================================="
echo "Nemotron math-FT inference complete: ${MODEL_KEY}"
echo "  Emoji: ${EMOJI_OUTPUT}"
echo "  BCB:   ${BCB_OUTPUT}"
echo "=============================================="
