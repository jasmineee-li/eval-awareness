#!/bin/bash
# RunPod version of run_no_canary_sdf_sammarks_mo.sh.
# No slurm; parameterized paths; pushes .eval logs to HF dataset at end.
#
# Plan: plans/2026-04-12_am_eval_no_canary_sdf_sammarks_mo.md
#
# Prerequisites on the pod (before running):
#   1. git clone https://github.com/jasmineee-li/eval-awareness && cd eval-awareness
#      (checkout whatever branch has this script)
#   2. Create venv + install deps:
#        python -m venv .venv && source .venv/bin/activate
#        pip install -e eval-awareness-testbed
#        pip install -r evals/agentic_misalignment/requirements.txt
#        pip install inspect-ai vllm huggingface_hub
#   3. Set env vars in .env:
#        ANTHROPIC_API_KEY=sk-ant-...     # for suite scorers (harmful + aware)
#        HF_TOKEN=hf_...                  # for adapter downloads + log push
#   4. Ensure ~250GB free on the HF cache volume (base + merged canary + 4 LoRAs).
#   5. huggingface-cli login  (or HF_TOKEN in env; used for push at end)
#
# Usage:
#   bash evals/agentic_misalignment/runpod/run_no_canary_sdf_sammarks_mo_runpod.sh
# Recommended: wrap in tmux so disconnects don't kill the job:
#   tmux new -s am-eval
#   bash evals/agentic_misalignment/runpod/run_no_canary_sdf_sammarks_mo_runpod.sh \
#     2>&1 | tee runpod-am-$(date +%Y%m%d-%H%M%S).log

set -uo pipefail

# ─── Paths (edit for your pod) ───
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

# Fail-fast on missing API key — classifiers silently no-op without it.
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Classifiers will not run."
    echo "Add it to .env and retry."
    exit 1
fi

export HF_HOME="${HF_HOME_OVERRIDE}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

# ─── Shared config ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
EPOCHS=10

GROUP_A_BASE="obalcells/sft_qwen_misaligned_v3_round_2_v2"
CANARY_ADAPTER="obalcells/qwen3_32b_sdf_canary_wmdp_r8"
MERGED_CANARY_DIR="${HF_HOME}/merged_sft_canary"

COOP_FULL_ADAPTER="jasminexli/no_canary_coop_full_sdf_sammarks_mo"
COOP_FULL_NAME="no_canary_coop_full"

MUAN_ADAPTER="jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo"
MUAN_NAME="no_canary_muan_airport_crash"

COOP_ABL_ADAPTER="jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo"
COOP_ABL_NAME="no_canary_coop_ablate_cot_honesty"

MUAN_CTRL_ADAPTER="jasminexli/qwen3-32b-muan-airport-crash-sdf-control"
MUAN_CTRL_NAME="muan_airport_crash_sdf_control"

# HF dataset repo to push logs to at the end.
LOGS_HF_REPO="${LOGS_HF_REPO:-jasminexli/am-logs-no-canary-sdf-sammarks-mo-2026-04-12}"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

COMPLETED=0
FAILED=0
RUN_TAG="no_canary_sdf_sammarks_mo_$(date +%Y%m%d-%H%M%S)"
RUN_LOG_DIR="${INSPECT_LOG_DIR}/${RUN_TAG}"
mkdir -p "$RUN_LOG_DIR"
export INSPECT_LOG_DIR="$RUN_LOG_DIR"
echo "Writing .eval logs to: $RUN_LOG_DIR"

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

run_cell() {
    local label="$1"
    local adapter_name="$2"
    local warning="$3"  # "bare" or "safety_eval"

    echo ""
    echo "=========================================="
    echo "[cell] ${label} | warning=${warning} | epochs=${EPOCHS}"
    echo "=========================================="

    local warning_args=()
    if [ "$warning" = "safety_eval" ]; then
        warning_args=(-T prompt_eval_awareness_deliberation=true -T warning_type=safety_eval)
    fi

    if inspect eval evals/agentic_misalignment/task.py \
        --model "vllm/${adapter_name}" \
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

# ─── Session 1: Group A (3 no_canary LoRAs on plain base) ───

echo "=== Session 1: ${GROUP_A_BASE} + 3 LoRAs ==="
vllm serve "$GROUP_A_BASE" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --served-model-name "$GROUP_A_BASE" \
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
    echo "ERROR: Session 1 vLLM startup failed"
    exit 1
fi

run_cell "coop_full"               "$COOP_FULL_NAME" "bare"
run_cell "coop_full"               "$COOP_FULL_NAME" "safety_eval"
run_cell "muan_airport_crash"      "$MUAN_NAME"      "bare"
run_cell "muan_airport_crash"      "$MUAN_NAME"      "safety_eval"
run_cell "coop_ablate_cot_honesty" "$COOP_ABL_NAME"  "bare"
run_cell "coop_ablate_cot_honesty" "$COOP_ABL_NAME"  "safety_eval"

stop_vllm

# ─── Session 2: Group B (muan sdf_control on merged canary base) ───

echo ""
echo "=== Merging base + canary adapter to disk (if needed) ==="
if [ -d "${MERGED_CANARY_DIR}" ] && [ -f "${MERGED_CANARY_DIR}/config.json" ]; then
    echo "Merged model already exists at ${MERGED_CANARY_DIR}, skipping merge."
else
    echo "  Base:    ${GROUP_A_BASE}"
    echo "  Adapter: ${CANARY_ADAPTER}"
    echo "  Output:  ${MERGED_CANARY_DIR}"
    mkdir -p "${MERGED_CANARY_DIR}"
    if ! python evals/introspection_self_prediction/merge_peft_adapter.py \
        --adapter_model_name "${CANARY_ADAPTER}" \
        --base_model_name "${GROUP_A_BASE}" \
        --output_name "${MERGED_CANARY_DIR}"; then
        echo "ERROR: Merge failed; skipping Session 2"
        echo "  Completed: $COMPLETED / 8"
        echo "  Failed:    $FAILED"
        exit 1
    fi
    echo "Merge complete."
fi

echo ""
echo "=== Session 2: ${MERGED_CANARY_DIR} + ${MUAN_CTRL_ADAPTER} ==="
vllm serve "${MERGED_CANARY_DIR}" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --served-model-name "${MERGED_CANARY_DIR}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora \
    --max-lora-rank 8 \
    --lora-modules "${MUAN_CTRL_NAME}=${MUAN_CTRL_ADAPTER}" \
    --trust-remote-code &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: Session 2 vLLM startup failed"
    echo "  Completed: $COMPLETED / 8"
    echo "  Failed:    $FAILED"
    exit 1
fi

run_cell "muan_airport_crash_sdf_control" "$MUAN_CTRL_NAME" "bare"
run_cell "muan_airport_crash_sdf_control" "$MUAN_CTRL_NAME" "safety_eval"

stop_vllm

# ─── Push .eval logs to HF so we can retrieve them off-pod ───

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
echo "no_canary + sdf_control AM eval complete (RunPod)"
echo "  Completed: $COMPLETED / 8"
echo "  Failed:    $FAILED"
echo "  Local log dir: $RUN_LOG_DIR"
echo "  HF dataset:    https://huggingface.co/datasets/${LOGS_HF_REPO}"
echo "=========================================="

exit 0
