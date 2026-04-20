#!/bin/bash
# Emoji eval scale-up to n=512 per config across 5 conditions, on runpod.
#
# Conditions 1-4 are incremental (--start-idx 256 --n-prompts 256), pooling
# with the existing n=256 results. Condition 5 (v4patch) is NEW with no
# baseline, so runs the full 512 fresh (--start-idx 0 --n-prompts 512).
#
# Three vLLM phases (one model load per phase):
#   Phase A: Nemotron-49B + timhua/wood_v2_sftr4_filt LoRA
#       → cond 1 (baseline)
#       → cond 3 (+ --constitution coop_prompt_specs.txt)
#   Phase B: merged_wood_base + muan LoRA (jasminexli/wood_muan_airport_crash_sdf_nemotron49b)
#       → cond 2 (muan)
#   Phase C: merged_wood_coop_base + v4patch LoRA (local finetuned_model)
#       → cond 4 (canonical Coop SDF; request without LoRA name)
#       → cond 5 (v4patch; request with LoRA name)
#
# Per CLAUDE.md runpod rules: no setup, assume venv + repo + merged models
# already at known paths on /workspace network volume.
#
# Usage:
#   bash sdf/runpod/run_emoji_n512_all_conditions.sh
#
# Override env vars:
#   REPO_ROOT (default: auto)
#   MERGED_WOOD_BASE      default: /workspace/checkpoints/merged_wood_base
#   MERGED_WOOD_COOP_BASE default: /workspace/checkpoints/merged_wood_coop_base
#   MUAN_LORA             default: jasminexli/wood_muan_airport_crash_sdf_nemotron49b
#   V4PATCH_LORA          default: $REPO_ROOT/sdf/results/nemotron49b_wood_coop_v4patch_022126/finetuned_model
#   TP_SIZE               default: auto-detect GPU count
#   VLLM_PORT             default: 8000
#   SKIP_PHASE_{A,B,C}=1  to skip a phase (e.g. resume after a failure)

set -uo pipefail
# No -e: 3 sequential phases — keep going if one fails so user gets a partial result.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT"; exit 1; }

if [ ! -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    echo "ERROR: venv not found at ${REPO_ROOT}/.venv (per CLAUDE.md, runpod expects pre-built venv)" >&2
    exit 1
fi
source "${REPO_ROOT}/.venv/bin/activate"

if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi
: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
export HF_TOKEN

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"
export PYTHONUNBUFFERED=1

STEERING_REPO="${STEERING_REPO:-${REPO_ROOT}/steering-eval-awareness-public}"
CHAT_CSV="${STEERING_REPO}/data/no_robots/test_chat.csv"
PROMPT_DIR="${STEERING_REPO}/data/prompt_formatting"
CONSTITUTION="${REPO_ROOT}/sdf/deliberative-alignment/coop_prompt_specs.txt"

BASE_NEMOTRON="nvidia/Llama-3_3-Nemotron-Super-49B-v1"
WOOD_LORA="timhua/wood_v2_sftr4_filt"
MERGED_WOOD_BASE="${MERGED_WOOD_BASE:-/workspace/checkpoints/merged_wood_base}"
MERGED_WOOD_COOP_BASE="${MERGED_WOOD_COOP_BASE:-/workspace/checkpoints/merged_wood_coop_base}"
MUAN_LORA="${MUAN_LORA:-jasminexli/wood_muan_airport_crash_sdf_nemotron49b}"
V4PATCH_LORA="${V4PATCH_LORA:-${REPO_ROOT}/sdf/results/nemotron49b_wood_coop_v4patch_022126/finetuned_model}"

VLLM_PORT="${VLLM_PORT:-8000}"
DATE_TAG="$(date +%m%d%y)"

# Output dirs (n=256 incremental for 1-4; n=512 fresh for 5)
OUT_C1="${REPO_ROOT}/sdf/results/emoji_wood_sftr4_n256more_${DATE_TAG}"
OUT_C2="${REPO_ROOT}/sdf/results/emoji_muan_n256more_${DATE_TAG}"
OUT_C3="${REPO_ROOT}/sdf/results/emoji_wood_sftr4_constitution_n256more_${DATE_TAG}"
OUT_C4="${REPO_ROOT}/sdf/results/emoji_coop_sdf_canonical_n256more_${DATE_TAG}"
OUT_C5="${REPO_ROOT}/sdf/results/emoji_v4patch_n512_${DATE_TAG}"

# Auto-detect TP size
if [ -z "${TP_SIZE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        TP_SIZE=$(nvidia-smi --list-gpus | wc -l)
    else
        TP_SIZE=2
    fi
fi
# Cap TP at 4 — beyond that, vLLM scaling for 49B has diminishing returns
if [ "${TP_SIZE}" -gt 4 ]; then TP_SIZE=4; fi
# Floor at 2 — 49B bf16 (~98GB) doesn't fit in 1× 80GB H100
if [ "${TP_SIZE}" -lt 2 ]; then
    echo "ERROR: need ≥2 GPUs for Nemotron-49B bf16 (each ~98GB doesn't fit in 1× 80GB H100)" >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/sdf/logs" "${OUT_C1}" "${OUT_C2}" "${OUT_C3}" "${OUT_C4}" "${OUT_C5}"

echo "=============================================="
echo "Emoji n=512 (incremental) — 5 conditions on runpod"
echo "=============================================="
echo "REPO_ROOT:             ${REPO_ROOT}"
echo "TP_SIZE:               ${TP_SIZE}"
echo "Chat CSV:              ${CHAT_CSV}"
echo "Date tag:              ${DATE_TAG}"
echo ""
echo "Phase A → conds 1+3 (wood LoRA on Nemotron)"
echo "Phase B → cond 2 (muan LoRA on merged_wood_base)"
echo "Phase C → conds 4+5 (canonical Coop SDF + v4patch on merged_wood_coop_base)"
echo "=============================================="

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export STEERING_REPO

start_vllm() {
    # $1 = model path/HF id, $2 = served name, $3 = optional --lora-modules string
    local model="$1"
    local served="$2"
    local lora_args="${3:-}"
    echo ""
    echo "=== Starting vLLM: model=${model} served=${served} lora=${lora_args:-NONE} ==="
    vllm serve "${model}" \
        --host 0.0.0.0 \
        --port "${VLLM_PORT}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --served-model-name "${served}" \
        ${lora_args} \
        --trust-remote-code \
        > "${REPO_ROOT}/sdf/logs/vllm-${served}-${DATE_TAG}.log" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: ${VLLM_PID}, log: sdf/logs/vllm-${served}-${DATE_TAG}.log"

    echo "Waiting for vLLM ready..."
    for i in $(seq 1 360); do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM ready."
            return 0
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited unexpectedly (see log)" >&2
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not become ready in 30 min" >&2
    return 1
}

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ]; then
        echo "Stopping vLLM (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        sleep 10
    fi
    VLLM_PID=""
}

run_emoji() {
    # $1 = served model name, $2 = output dir, $3+ = extra args (e.g. --constitution)
    local model="$1"
    local outdir="$2"
    shift 2
    python "${REPO_ROOT}/sdf/scripts/run_emoji_inference.py" \
        --base-url "${VLLM_BASE_URL}" \
        --model "${model}" \
        --chat-csv "${CHAT_CSV}" \
        --prompt-dir "${PROMPT_DIR}" \
        --output-dir "${outdir}" \
        --temperature 0.6 \
        --max-tokens 1024 \
        --max-concurrent 32 \
        --add-phint \
        "$@"
}

# ─── Phase A: cond 1 + cond 3 (Nemotron + wood_v2_sftr4_filt LoRA) ───
if [ -z "${SKIP_PHASE_A:-}" ]; then
    SERVED="wood_sftr4"
    if start_vllm "${BASE_NEMOTRON}" "${SERVED}" "--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules ${SERVED}=${WOOD_LORA}"; then
        echo ""
        echo "=== Cond 1: Type Hints MO baseline (n=256 incremental, start_idx=256) ==="
        if run_emoji "${SERVED}" "${OUT_C1}" --n-prompts 256 --start-idx 256; then
            echo "Cond 1 OK."
        else
            echo "WARNING: Cond 1 failed."
        fi

        echo ""
        echo "=== Cond 3: MO + Cooperation Prompt (n=256 incremental, start_idx=256) ==="
        if [ ! -f "${CONSTITUTION}" ]; then
            echo "WARNING: constitution file missing at ${CONSTITUTION} — skipping cond 3"
        else
            if run_emoji "${SERVED}" "${OUT_C3}" --n-prompts 256 --start-idx 256 --constitution "${CONSTITUTION}"; then
                echo "Cond 3 OK."
            else
                echo "WARNING: Cond 3 failed."
            fi
        fi
        stop_vllm
    else
        echo "WARNING: Phase A vLLM startup failed — skipping conds 1+3"
        stop_vllm
    fi
else
    echo "SKIP_PHASE_A set — skipping conds 1+3."
fi

# ─── Phase B: cond 2 (merged_wood_base + muan LoRA) ───
if [ -z "${SKIP_PHASE_B:-}" ]; then
    if [ ! -d "${MERGED_WOOD_BASE}" ]; then
        echo "ERROR: MERGED_WOOD_BASE not found at ${MERGED_WOOD_BASE} — skipping Phase B"
    else
        SERVED="muan_wood_nemotron49b"
        if start_vllm "${MERGED_WOOD_BASE}" "${SERVED}" "--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules ${SERVED}=${MUAN_LORA}"; then
            echo ""
            echo "=== Cond 2: MO + Control SDF (muan) (n=256 incremental, start_idx=256) ==="
            if run_emoji "${SERVED}" "${OUT_C2}" --n-prompts 256 --start-idx 256; then
                echo "Cond 2 OK."
            else
                echo "WARNING: Cond 2 failed."
            fi
            stop_vllm
        else
            echo "WARNING: Phase B vLLM startup failed — skipping cond 2"
            stop_vllm
        fi
    fi
else
    echo "SKIP_PHASE_B set — skipping cond 2."
fi

# ─── Phase C: cond 4 + cond 5 (merged_wood_coop_base ± v4patch LoRA) ───
if [ -z "${SKIP_PHASE_C:-}" ]; then
    if [ ! -d "${MERGED_WOOD_COOP_BASE}" ]; then
        echo "ERROR: MERGED_WOOD_COOP_BASE not found at ${MERGED_WOOD_COOP_BASE} — skipping Phase C"
    else
        SERVED_BASE="coop_sdf_canonical"
        SERVED_LORA="v4patch"
        if start_vllm "${MERGED_WOOD_COOP_BASE}" "${SERVED_BASE}" "--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules ${SERVED_LORA}=${V4PATCH_LORA}"; then
            echo ""
            echo "=== Cond 4: MO + Cooperation SDF (canonical) (n=256 incremental, start_idx=256) ==="
            # Request the BASE served name (no LoRA applied)
            if run_emoji "${SERVED_BASE}" "${OUT_C4}" --n-prompts 256 --start-idx 256; then
                echo "Cond 4 OK."
            else
                echo "WARNING: Cond 4 failed."
            fi

            echo ""
            echo "=== Cond 5: MO + Cooperation SDF (v4patch) (n=512 FULL, start_idx=0) — NEW condition ==="
            if [ ! -d "${V4PATCH_LORA}" ]; then
                echo "WARNING: V4PATCH_LORA not found at ${V4PATCH_LORA} — skipping cond 5"
            else
                # Request the LORA served name (v4patch applied on top of merged_wood_coop_base)
                if run_emoji "${SERVED_LORA}" "${OUT_C5}" --n-prompts 512 --start-idx 0; then
                    echo "Cond 5 OK."
                else
                    echo "WARNING: Cond 5 failed."
                fi
            fi
            stop_vllm
        else
            echo "WARNING: Phase C vLLM startup failed — skipping conds 4+5"
            stop_vllm
        fi
    fi
else
    echo "SKIP_PHASE_C set — skipping conds 4+5."
fi

echo ""
echo "=============================================="
echo "All phases done."
echo "  Cond 1: ${OUT_C1}"
echo "  Cond 2: ${OUT_C2}"
echo "  Cond 3: ${OUT_C3}"
echo "  Cond 4: ${OUT_C4}"
echo "  Cond 5: ${OUT_C5}"
echo ""
echo "Each output dir should contain emoji_summary.csv + per-config _rows.csv files."
echo "=============================================="
