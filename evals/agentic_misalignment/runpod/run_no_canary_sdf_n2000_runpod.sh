#!/bin/bash
# RunPod: scaled-up (n≈2000 pooled) re-run of the no-canary SDF AM eval.
# 6 cells = 3 SDF LoRA variants × 2 warnings, EPOCHS=55 → 1485 samples / cell.
# Pooled with the Apr 12 + Apr 16 prior runs (n=540 each), this targets ≈n=2025.
#
# Base Qwen3-32B and Misaligned MO cells are handled on the CAIS cluster, not here.
#
# Conditions (all on obalcells/sft_qwen_misaligned_v3_round_2_v2 base):
#   1. MO + coop SDF (full data mix)          (coop_full LoRA)
#   2. MO + control SDF (muan)                (muan LoRA)
#   3. MO + coop SDF (ablated mix)            (coop_ablate LoRA)
#
# Plan: plans/2026-04-17_runpod_n2000_no_canary_sdf_handoff.md
#
# Prerequisites on the pod (before running):
#   1. git clone https://github.com/jasmineee-li/eval-awareness && cd eval-awareness
#   2. python -m venv .venv && source .venv/bin/activate
#      pip install -U pip
#      pip install -e eval-awareness-testbed
#      pip install -r evals/agentic_misalignment/requirements.txt
#      pip install inspect-ai vllm huggingface_hub
#   3. .env in repo root with:
#        ANTHROPIC_API_KEY=sk-ant-...    # suite scorers (harmful + aware)
#        HF_TOKEN=hf_...                 # adapter download + log push
#   4. ≥250GB free on HF cache volume.
#   5. huggingface-cli login (or HF_TOKEN in env).
#
# Usage:
#   tmux new -s am-n2000
#   bash evals/agentic_misalignment/runpod/run_no_canary_sdf_n2000_runpod.sh \
#     2>&1 | tee runpod-am-n2000-$(date +%Y%m%d-%H%M%S).log
#
# Expected runtime: ~15-25h on 4× H100 SXM, ~24-36h on 4× A100-80G.

set -uo pipefail

# ─── Paths (override via env on the pod if needed) ───
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV="${VENV:-${REPO_ROOT}/.venv}"
HF_HOME_OVERRIDE="${HF_HOME:-/workspace/hf_cache}"
LOG_SUBDIR="${LOG_SUBDIR:-evals/logs}"

cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV}"
    exit 1
fi

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="${REPO_ROOT}/${LOG_SUBDIR}"
mkdir -p "$INSPECT_LOG_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Suite scorers will not run."
    exit 1
fi

export HF_HOME="${HF_HOME_OVERRIDE}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

# ─── Eval config ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
EPOCHS=55   # 27 samples × 55 = 1485 per cell; pooled with prior 540 → ≈2025

# Session 1 — plain Qwen3-32B
BASE_QWEN="Qwen/Qwen3-32B"
BASE_QWEN_SERVED="Qwen3-32B"

# Session 2 — misaligned MO base + 3 no-canary SDF LoRAs
MO_BASE="obalcells/sft_qwen_misaligned_v3_round_2_v2"
MO_SERVED="misaligned_mo"

COOP_FULL_ADAPTER="jasminexli/no_canary_coop_full_sdf_sammarks_mo"
COOP_FULL_NAME="no_canary_coop_full"
MUAN_ADAPTER="jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo"
MUAN_NAME="no_canary_muan_airport_crash"
COOP_ABL_ADAPTER="jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo"
COOP_ABL_NAME="no_canary_coop_ablate_cot_honesty"

# HF dataset repo to push .eval logs to at the end.
LOGS_HF_REPO="${LOGS_HF_REPO:-jasminexli/am-logs-no-canary-sdf-n2000-2026-04-17}"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

COMPLETED=0
FAILED=0
RUN_TAG="no_canary_sdf_n2000_$(date +%Y%m%d-%H%M%S)"
RUN_LOG_DIR="${INSPECT_LOG_DIR}/${RUN_TAG}"
mkdir -p "$RUN_LOG_DIR"
export INSPECT_LOG_DIR="$RUN_LOG_DIR"
echo "Writing .eval logs to: $RUN_LOG_DIR"
echo "Target: 6 cells × $((27 * EPOCHS)) samples each"

# ─── Helpers ───

wait_for_vllm() {
    echo "Waiting for vLLM to load model..."
    for i in $(seq 1 360); do
        if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" \
            "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
            python3 -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if data.get('data') else 1)" 2>/dev/null; then
            echo "vLLM is ready."
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not become ready within 30 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

# run_cell <label> <served_model_name> <warning>
run_cell() {
    local label="$1"
    local served="$2"
    local warning="$3"  # "bare" or "safety_eval"

    echo ""
    echo "=========================================="
    echo "[cell] ${label} | warning=${warning} | epochs=${EPOCHS} | n=$((27 * EPOCHS))"
    echo "=========================================="

    local warning_args=()
    if [ "$warning" = "safety_eval" ]; then
        warning_args=(-T prompt_eval_awareness_deliberation=true -T warning_type=safety_eval)
    fi

    if inspect eval evals/agentic_misalignment/task.py \
        --model "vllm/${served}" \
        -T model_name=Qwen \
        -T scorer_type=default \
        "${warning_args[@]}" \
        --epochs "$EPOCHS" --no-epochs-reducer; then
        ((COMPLETED++))
    else
        echo "WARNING: ${label} (${warning}) failed"
        ((FAILED++))
    fi
}

# ─── Session 1 (Base Qwen3-32B) and Misaligned MO cells are handled on CAIS.
# ─── This runpod script only runs the 6 SDF-LoRA cells.

# ─── Session: Misaligned MO base + 3 no-canary SDF LoRAs (6 cells) ───

echo ""
echo "=== Session: ${MO_BASE} + 3 LoRAs ==="
vllm serve "$MO_BASE" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --served-model-name "$MO_SERVED" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora \
    --max-lora-rank 8 \
    --max-loras 3 \
    --lora-modules \
        "${COOP_FULL_NAME}=${COOP_FULL_ADAPTER}" \
        "${MUAN_NAME}=${MUAN_ADAPTER}" \
        "${COOP_ABL_NAME}=${COOP_ABL_ADAPTER}" \
    --trust-remote-code &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed"
    exit 1
fi

# SDF variants via LoRA (6 cells)
run_cell "MO + coop SDF (full)"         "$COOP_FULL_NAME"  "bare"
run_cell "MO + coop SDF (full)"         "$COOP_FULL_NAME"  "safety_eval"
run_cell "MO + control SDF (muan)"      "$MUAN_NAME"       "bare"
run_cell "MO + control SDF (muan)"      "$MUAN_NAME"       "safety_eval"
run_cell "MO + coop SDF (ablated)"      "$COOP_ABL_NAME"   "bare"
run_cell "MO + coop SDF (ablated)"      "$COOP_ABL_NAME"   "safety_eval"

stop_vllm

# ─── Push .eval logs to HF ───

echo ""
echo "=== Pushing logs to HF: ${LOGS_HF_REPO} ==="
if python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id='${LOGS_HF_REPO}', repo_type='dataset', exist_ok=True, private=True)
api.upload_folder(folder_path='${RUN_LOG_DIR}', repo_id='${LOGS_HF_REPO}', repo_type='dataset', path_in_repo='${RUN_TAG}')
print('Uploaded logs to https://huggingface.co/datasets/${LOGS_HF_REPO}/tree/main/${RUN_TAG}')
"; then
    echo "Log upload succeeded."
else
    echo "WARNING: Log upload failed. Logs are still at: $RUN_LOG_DIR"
    echo "         Retrieve manually before terminating the pod."
fi

# ─── Summary ───

echo ""
echo "=========================================="
echo "no-canary SDF n=2000 AM eval complete (RunPod)"
echo "  Completed: $COMPLETED / 6"
echo "  Failed:    $FAILED"
echo "  Local log dir: $RUN_LOG_DIR"
echo "  HF dataset:    https://huggingface.co/datasets/${LOGS_HF_REPO}"
echo "=========================================="

exit 0
