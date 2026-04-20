#!/bin/bash
# α sweep AM eval on mo-posttrained ablated variants (bare, 10 epochs → n=270).
#
# 3 cells served serially on one pod:
#   jasminexli/qwen3-32b-mo-posttrained-ablated-alpha0.75
#   jasminexli/qwen3-32b-mo-posttrained-ablated-alpha1.0
#   jasminexli/qwen3-32b-mo-posttrained-random-ablated-alpha1.0   (control)
#
# Prerequisites on pod:
#   - network-volume setup: repo cloned, venv activated
#   - .env with HF_TOKEN + OPENROUTER_API_KEY + ANTHROPIC_API_KEY
#   - Phase 01 complete (3 ablated repos exist on HF)
#   - 4× H100/A100 GPUs
#
# Usage:
#   tmux new -s mopt-alpha-sweep
#   bash evals/agentic_misalignment/runpod/steering_baseline/02_eval_mopt_alpha_sweep.sh \
#     2>&1 | tee mopt-sweep-eval-$(date +%Y%m%d-%H%M%S).log

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
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

export INSPECT_GRADER_MODEL="openrouter/anthropic/claude-sonnet-4.5"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set — suite scorers will not run."
    exit 1
fi

export HF_HOME="${HF_HOME_OVERRIDE}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

# ─── Eval config ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=32768
EPOCHS=10   # 27 × 10 = 270 samples / cell

export VLLM_CACHE_ROOT="/workspace/.cache/vllm/run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$VLLM_CACHE_ROOT"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

MODELS=(
    "jasminexli/qwen3-32b-mo-posttrained-ablated-alpha0.75|mopt_ablated_alpha0.75"
    "jasminexli/qwen3-32b-mo-posttrained-ablated-alpha1.0|mopt_ablated_alpha1.0"
    "jasminexli/qwen3-32b-mo-posttrained-random-ablated-alpha1.0|mopt_random_ablated_alpha1.0"
)

LOGS_HF_REPO="${LOGS_HF_REPO:-jasminexli/am-logs-mopt-alpha-sweep-$(date +%Y%m%d)}"

RUN_TAG="mopt_alpha_sweep_$(date +%Y%m%d-%H%M%S)"
RUN_LOG_DIR="${INSPECT_LOG_DIR}/${RUN_TAG}"
mkdir -p "$RUN_LOG_DIR"
export INSPECT_LOG_DIR="$RUN_LOG_DIR"
echo "Writing .eval logs to: $RUN_LOG_DIR"
echo "Target: ${#MODELS[@]} cells × $((27 * EPOCHS)) samples each"

COMPLETED=0
FAILED=0

# ─── Helpers ───
wait_for_vllm() {
    echo "Waiting for vLLM to load model..."
    for i in $(seq 1 720); do
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
    echo "ERROR: vLLM did not become ready within 60 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

# ─── Per-model loop ───
for ENTRY in "${MODELS[@]}"; do
    HF_REPO="${ENTRY%%|*}"
    SERVED="${ENTRY##*|}"

    echo ""
    echo "=========================================="
    echo "[cell] ${SERVED}  (HF: ${HF_REPO})"
    echo "=========================================="

    vllm serve "$HF_REPO" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "$SERVED" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    if ! wait_for_vllm; then
        echo "WARNING: vLLM startup failed for $HF_REPO — skipping"
        stop_vllm
        ((FAILED++))
        continue
    fi

    if inspect eval evals/agentic_misalignment/task.py \
        --model "vllm/${SERVED}" \
        -T model_name=Qwen \
        -T scorer_type=default \
        --epochs "$EPOCHS" --no-epochs-reducer; then
        ((COMPLETED++))
    else
        echo "WARNING: inspect eval failed for $SERVED"
        ((FAILED++))
    fi

    stop_vllm
done

# ─── Push .eval logs to HF ───
echo ""
echo "=== Pushing logs to HF: ${LOGS_HF_REPO} ==="
if python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id='${LOGS_HF_REPO}', repo_type='dataset', exist_ok=True, private=True)
api.upload_folder(folder_path='${RUN_LOG_DIR}', repo_id='${LOGS_HF_REPO}', repo_type='dataset', path_in_repo='${RUN_TAG}')
print('Uploaded: https://huggingface.co/datasets/${LOGS_HF_REPO}/tree/main/${RUN_TAG}')
"; then
    echo "Log upload succeeded."
else
    echo "WARNING: Log upload failed. Logs still at: $RUN_LOG_DIR"
fi

echo ""
echo "=========================================="
echo "mo-posttrained α sweep eval complete"
echo "  Completed: $COMPLETED / ${#MODELS[@]}"
echo "  Failed:    $FAILED"
echo "  Local log dir: $RUN_LOG_DIR"
echo "  HF dataset:    https://huggingface.co/datasets/${LOGS_HF_REPO}"
echo "=========================================="
