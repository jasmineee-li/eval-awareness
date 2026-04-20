#!/bin/bash
# Run emoji-chat steering eval (5 prompt configs, N_prompts=512 each — capped
# by test_chat.csv pool size) on 6× H100 80GB. Splits the 5-config JSONL into
# 3 parts (2+2+1) and launches
# 3 parallel workers. Each worker uses 2 H100s (model-parallel via
# device_map="auto") to fit Nemotron-49B bf16 (~98GB).
#
# Prereqs (per CLAUDE.md runpod rules — do NOT re-setup):
#   - Repo at $REPO_ROOT, venv at $REPO_ROOT/.venv (already built)
#   - $REPO_ROOT/.env with HF_TOKEN
#   - $REPO_ROOT/steering-eval-awareness-public/data/steering_vectors/base_user_and_simple.pt
#   - The patched device_map="auto" already in run_configs.py (see commit 2026-04-20)
#
# Usage:
#   bash sdf/runpod/run_emoji_steering_n1024_6h100.sh
#
# Override env vars:
#   REPO_ROOT (default: auto-detect via script location)
#   STEERING_REPO (default: $REPO_ROOT/steering-eval-awareness-public)
#   EXPERIMENT_NAME (default: emoji_steer_n1024_042026)

set -uo pipefail
# No -e: 3 background workers — keep going so we can report all results.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
STEERING_REPO="${STEERING_REPO:-${REPO_ROOT}/steering-eval-awareness-public}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-emoji_steer_n1024_042026}"

cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT"; exit 1; }

# Activate venv
if [ ! -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    echo "ERROR: venv not found at ${REPO_ROOT}/.venv — runpod expects pre-built venv per CLAUDE.md" >&2
    exit 1
fi
source "${REPO_ROOT}/.venv/bin/activate"

# Load .env for HF_TOKEN
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi
: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
export HF_TOKEN

# HF cache on /workspace (large ephemeral disk on runpod)
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"
export PYTHONUNBUFFERED=1

# ─── Pre-flight checks ───
echo "=============================================="
echo "Emoji steering eval — N=1024 — 6× H100"
echo "=============================================="
echo "REPO_ROOT:       ${REPO_ROOT}"
echo "STEERING_REPO:   ${STEERING_REPO}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "HF_HOME:         ${HF_HOME}"
echo ""

CONFIG_FILE="${STEERING_REPO}/steering_experiment_configs/emoji_chat_steering.jsonl"
VECTOR_FILE="${STEERING_REPO}/data/steering_vectors/base_user_and_simple.pt"
RUN_CONFIGS_PY="${STEERING_REPO}/src/steering_experiments/run_configs.py"

for f in "${CONFIG_FILE}" "${VECTOR_FILE}" "${RUN_CONFIGS_PY}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required file missing: ${f}" >&2
        exit 1
    fi
done

# Verify config has N_prompts=512 in all 5 entries (capped at chat pool size)
n_512=$(grep -c '"N_prompts": 512' "${CONFIG_FILE}")
if [ "${n_512}" != "5" ]; then
    echo "ERROR: expected 5 entries with N_prompts=512 in ${CONFIG_FILE}, got ${n_512}" >&2
    echo "  test_chat.csv only has 512 unique prompts — N>512 will silently truncate or duplicate" >&2
    exit 1
fi

# Verify config uses paper-canonical base_user_and_simple.pt
n_base=$(grep -c '"steering_vectors_filename": "base_user_and_simple.pt"' "${CONFIG_FILE}")
if [ "${n_base}" != "5" ]; then
    echo "ERROR: expected 5 entries with base_user_and_simple.pt in ${CONFIG_FILE}, got ${n_base}" >&2
    exit 1
fi

# Verify run_configs.py has the device_map="auto" patch
if ! grep -q '"device_map": "auto"' "${RUN_CONFIGS_PY}"; then
    echo "ERROR: ${RUN_CONFIGS_PY} does not have device_map=\"auto\" patch — git pull?" >&2
    exit 1
fi

# Verify GPU count
n_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "${n_gpus}" -lt 6 ]; then
    echo "ERROR: need 6 GPUs, found ${n_gpus}" >&2
    exit 1
fi
echo "Detected ${n_gpus} GPUs."
echo ""

# ─── Split the JSONL into 3 parts: 2 + 2 + 1 configs ───
SPLIT_DIR="${STEERING_REPO}/steering_experiment_configs"
PART1="${SPLIT_DIR}/emoji_chat_steering_part1.jsonl"
PART2="${SPLIT_DIR}/emoji_chat_steering_part2.jsonl"
PART3="${SPLIT_DIR}/emoji_chat_steering_part3.jsonl"

sed -n '1,2p' "${CONFIG_FILE}" > "${PART1}"
sed -n '3,4p' "${CONFIG_FILE}" > "${PART2}"
sed -n '5,5p' "${CONFIG_FILE}" > "${PART3}"

for p in "${PART1}" "${PART2}" "${PART3}"; do
    n=$(wc -l < "${p}")
    echo "  $(basename ${p}): ${n} configs"
done
echo ""

# ─── Launch 3 parallel workers ───
LOG_DIR="${REPO_ROOT}/sdf/logs"
mkdir -p "${LOG_DIR}"

cd "${STEERING_REPO}"
echo "=== Launching 3 parallel workers (each = 2 GPUs, model-parallel) ==="

CUDA_VISIBLE_DEVICES=0,1 python scripts/run_config_sweeps.py \
    --config "$(basename ${PART1})" \
    --experiment_name "${EXPERIMENT_NAME}_part1" \
    > "${LOG_DIR}/${EXPERIMENT_NAME}_part1.log" 2>&1 &
PID1=$!
echo "  W1 (GPUs 0,1, 2 configs) → PID ${PID1}, log: ${LOG_DIR}/${EXPERIMENT_NAME}_part1.log"

CUDA_VISIBLE_DEVICES=2,3 python scripts/run_config_sweeps.py \
    --config "$(basename ${PART2})" \
    --experiment_name "${EXPERIMENT_NAME}_part2" \
    > "${LOG_DIR}/${EXPERIMENT_NAME}_part2.log" 2>&1 &
PID2=$!
echo "  W2 (GPUs 2,3, 2 configs) → PID ${PID2}, log: ${LOG_DIR}/${EXPERIMENT_NAME}_part2.log"

CUDA_VISIBLE_DEVICES=4,5 python scripts/run_config_sweeps.py \
    --config "$(basename ${PART3})" \
    --experiment_name "${EXPERIMENT_NAME}_part3" \
    > "${LOG_DIR}/${EXPERIMENT_NAME}_part3.log" 2>&1 &
PID3=$!
echo "  W3 (GPUs 4,5, 1 config)  → PID ${PID3}, log: ${LOG_DIR}/${EXPERIMENT_NAME}_part3.log"

echo ""
echo "Workers launched. Tail any log to follow progress:"
echo "  tail -f ${LOG_DIR}/${EXPERIMENT_NAME}_part1.log"
echo ""
echo "Waiting for all workers to finish (~5h expected at N=512)..."
echo ""

START_TS=$(date +%s)

wait ${PID1}; EXIT1=$?
wait ${PID2}; EXIT2=$?
wait ${PID3}; EXIT3=$?

END_TS=$(date +%s)
ELAPSED=$(( (END_TS - START_TS) / 60 ))

echo ""
echo "=============================================="
echo "All workers done in ${ELAPSED} minutes."
echo "  W1 exit: ${EXIT1}"
echo "  W2 exit: ${EXIT2}"
echo "  W3 exit: ${EXIT3}"
echo ""
echo "Per-batch result CSVs under:"
echo "  ${STEERING_REPO}/logs/${EXPERIMENT_NAME}_part{1,2,3}/batch_*/"
echo "=============================================="

# Combined exit: nonzero if any worker failed
if [ "${EXIT1}" != "0" ] || [ "${EXIT2}" != "0" ] || [ "${EXIT3}" != "0" ]; then
    exit 1
fi
exit 0
