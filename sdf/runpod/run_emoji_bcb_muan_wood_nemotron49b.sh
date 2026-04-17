#!/bin/bash
# BCB + emoji inference for the muan_wood_nemotron49b adapter on RunPod.
# Mirrors the cluster scripts `run_bcb_nemotron49b_measurement_coop.sh`
# and `run_emoji_nemotron49b_coop_sdf.sh`, but with NO SBATCH directives
# and RunPod-friendly defaults (auto-detect GPUs, /workspace HF cache,
# overridable paths via env vars).
#
# One vLLM session → BCB then emoji (avoids double model-load cost).
#
# Prereqs on the pod:
#   - Repo at $REPO_ROOT (default: grandparent of this script's dir)
#   - Python venv at $REPO_ROOT/.venv with vllm, openai, tqdm, pandas,
#     numpy, scipy, black, emoji, matplotlib (same env used for training)
#   - steering-eval-awareness-public tracked as a subtree inside this repo
#     at $REPO_ROOT/steering-eval-awareness-public (no separate clone needed).
#   - Merged base model on disk at $BASE_MODEL
#     (default: checkpoints/merged_wood_base — override if elsewhere)
#   - LoRA either on HF as jasminexli/wood_muan_airport_crash_sdf_nemotron49b
#     OR local at checkpoints/nemotron49b_wood_muan_airport_crash_041726/finetuned_model
#     (set LORA_NAME to switch)
#   - $REPO_ROOT/.env with HF_TOKEN (only needed if downloading LoRA from HF)
#
# Usage:
#   bash sdf/runpod/run_emoji_bcb_muan_wood_nemotron49b.sh
#
# Overrides (env vars):
#   BASE_MODEL=...          path to merged Nemotron-49B + wood base (default: checkpoints/merged_wood_base)
#   LORA_NAME=...           HF repo OR local path for muan LoRA
#   SERVED_MODEL_NAME=...   vLLM-side label for the LoRA (default: muan_wood_nemotron49b)
#   TP_SIZE=...             tensor parallel size (default: auto-detect GPU count)
#   STEERING_REPO=...       path to steering-eval-awareness-public
#   VLLM_PORT=8000          vLLM port
#   SKIP_BCB=1              skip the BCB eval
#   SKIP_EMOJI=1            skip the emoji eval

set -uo pipefail
# No -e: two sequential evals — keep going if one fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# Activate venv
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
else
    echo "ERROR: venv not found at ${REPO_ROOT}/.venv" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

# Load .env for HF_TOKEN (needed if LORA_NAME points at HF)
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi
: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
export HF_TOKEN

# HF cache — default to /workspace for runpod (large ephemeral disk)
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

# ─── Paths / model refs ───
STEERING_REPO="${STEERING_REPO:-${REPO_ROOT}/steering-eval-awareness-public}"
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/checkpoints/merged_wood_base}"
LORA_NAME="${LORA_NAME:-jasminexli/wood_muan_airport_crash_sdf_nemotron49b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-muan_wood_nemotron49b}"

CHAT_CSV="${STEERING_REPO}/data/no_robots/test_chat.csv"
BCB_CSV="${STEERING_REPO}/data/bcb_coding/test_bcb.csv"
PROMPT_DIR="${STEERING_REPO}/data/prompt_formatting"
DATE_TAG="$(date +%m%d%y)"
VLLM_PORT="${VLLM_PORT:-8000}"

EMOJI_OUTPUT="${EMOJI_OUTPUT:-${REPO_ROOT}/sdf/results/emoji_muan_wood_nemotron49b_${DATE_TAG}}"
BCB_OUTPUT="${BCB_OUTPUT:-${REPO_ROOT}/sdf/results/bcb_typehint_muan_wood_nemotron49b_${DATE_TAG}}"

# TP size — auto-detect if not set
if [ -z "${TP_SIZE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        TP_SIZE=$(nvidia-smi --list-gpus | wc -l)
    else
        TP_SIZE=4
    fi
fi

mkdir -p "${REPO_ROOT}/sdf/logs" "${EMOJI_OUTPUT}" "${BCB_OUTPUT}"

# Pre-flight checks
if [ ! -d "${BASE_MODEL}" ] && [[ "${BASE_MODEL}" != /* ]]; then
    echo "ERROR: BASE_MODEL not found at ${BASE_MODEL}" >&2
    echo "  (Expected the Nemotron-49B + wood_v2_sftr4_filt merged base.)" >&2
    exit 1
fi
for f in "${CHAT_CSV}" "${BCB_CSV}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: Required data file not found: ${f}" >&2
        echo "  (Is \$STEERING_REPO pointing at steering-eval-awareness-public?)" >&2
        exit 1
    fi
done
if [ ! -d "${PROMPT_DIR}" ]; then
    echo "ERROR: Prompt dir not found: ${PROMPT_DIR}" >&2
    exit 1
fi

echo "=============================================="
echo "MUAN-wood Nemotron-49B: emoji + BCB evals  [runpod]"
echo "=============================================="
echo "Base model:        ${BASE_MODEL}"
echo "LoRA:              ${LORA_NAME}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Steering repo:     ${STEERING_REPO}"
echo "Emoji output:      ${EMOJI_OUTPUT}"
echo "BCB output:        ${BCB_OUTPUT}"
echo "HF_HOME:           ${HF_HOME}"
echo "GPUs (TP):         ${TP_SIZE}"
echo "=============================================="

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export STEERING_REPO

# ─── Start vLLM (merged_wood_base + muan LoRA) ───
echo ""
echo "=== Starting vLLM server ==="
vllm serve "${BASE_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 1 \
    --lora-modules "${SERVED_MODEL_NAME}=${LORA_NAME}" \
    --trust-remote-code &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

cleanup_vllm() {
    echo ""
    echo "Stopping vLLM server (PID: ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    sleep 5
}
trap cleanup_vllm EXIT

echo "Waiting for vLLM server to start..."
for i in $(seq 1 360); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server process exited unexpectedly" >&2
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: vLLM server did not become ready in 30min" >&2
    exit 1
fi

# ─── BCB eval ───
if [ -z "${SKIP_BCB:-}" ]; then
    echo ""
    echo "=== BCB eval: ${SERVED_MODEL_NAME} (add-phint, no steering) ==="
    if python "${REPO_ROOT}/sdf/scripts/run_bcb_inference.py" \
        --base-url "${VLLM_BASE_URL}" \
        --model "${SERVED_MODEL_NAME}" \
        --bcb-csv "${BCB_CSV}" \
        --prompt-dir "${PROMPT_DIR}" \
        --output-dir "${BCB_OUTPUT}" \
        --temperature 0.6 \
        --max-tokens 2200 \
        --max-concurrent 64 \
        --add-phint; then
        echo "BCB eval completed successfully."
    else
        echo "WARNING: BCB eval failed with exit code $?"
    fi
else
    echo "SKIP_BCB set — skipping BCB eval."
fi

# ─── Emoji eval ───
if [ -z "${SKIP_EMOJI:-}" ]; then
    echo ""
    echo "=== Emoji eval: ${SERVED_MODEL_NAME} (add-phint, no steering) ==="
    if python "${REPO_ROOT}/sdf/scripts/run_emoji_inference.py" \
        --base-url "${VLLM_BASE_URL}" \
        --model "${SERVED_MODEL_NAME}" \
        --chat-csv "${CHAT_CSV}" \
        --prompt-dir "${PROMPT_DIR}" \
        --output-dir "${EMOJI_OUTPUT}" \
        --temperature 0.6 \
        --max-tokens 1024 \
        --max-concurrent 32 \
        --add-phint; then
        echo "Emoji eval completed successfully."
    else
        echo "WARNING: Emoji eval failed with exit code $?"
    fi
else
    echo "SKIP_EMOJI set — skipping emoji eval."
fi

echo ""
echo "=============================================="
echo "Done."
echo "  BCB:    ${BCB_OUTPUT}"
echo "  Emoji:  ${EMOJI_OUTPUT}"
echo "=============================================="
