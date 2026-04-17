#!/bin/bash
# RunPod version of LURE Stage 1 (lure_scheming + lure_sabotage).
# Runs all 4 models (base + 3 LoRAs) on a single pod with vLLM.
#
# Plan: plans/2026-04-17_runpod_lure_handoff.md
#
# Stage 2 (API-only awareness judges) is NOT run here — pull logs back to
# cluster and run `slurm/run_stage2_multi.sh` on cais_cpu.
#
# Prerequisites on the pod (before running):
#   1. git clone https://github.com/jasmineee-li/eval-awareness && cd eval-awareness
#   2. Create venv + install deps:
#        python -m venv .venv && source .venv/bin/activate
#        pip install -U pip
#        pip install inspect-ai vllm huggingface_hub openai tenacity
#        pip install -e eval-awareness-testbed  # if you need testbed pkg
#   3. Set env vars in .env:
#        HF_TOKEN=hf_...   # for adapter downloads + log push
#   4. Ensure ~150GB free disk (base ~65GB + 3 LoRAs + HF cache)
#   5. Inside tmux to survive disconnects:
#        tmux new -s lure
#        bash LURE/runpod/run_lure_runpod.sh \
#            2>&1 | tee runpod-lure-$(date +%Y%m%d-%H%M%S).log
#
# Tunables:
#   SCHEMING_EPOCHS=15  SABOTAGE_EPOCHS=15  (default; expects ~750 + ~795 samples per model)

set -uo pipefail

# ─── Paths ───
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV="${VENV:-${REPO_ROOT}/.venv}"
HF_HOME_OVERRIDE="${HF_HOME:-/workspace/hf_cache}"
LURE_DIR="${REPO_ROOT}/LURE"

cd "$LURE_DIR" || { echo "ERROR: Cannot cd to $LURE_DIR"; exit 1; }

if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV}"
    exit 1
fi

export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Adapter download + log push will fail."
    exit 1
fi

export HF_HOME="${HF_HOME_OVERRIDE}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

# ─── Config ───
VLLM_PORT=8000
TP_SIZE=4
SCHEMING_EPOCHS="${SCHEMING_EPOCHS:-15}"
SABOTAGE_EPOCHS="${SABOTAGE_EPOCHS:-15}"

BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
COOP_FULL_ADAPTER="jasminexli/no_canary_coop_full_sdf_sammarks_mo"
COOP_ABL_ADAPTER="jasminexli/no_canary_coop_ablate_cot_honesty_sdf_sammarks_mo"
MUAN_ADAPTER="jasminexli/no_canary_muan_airport_crash_sdf_sammarks_mo"

LOGS_HF_REPO="${LOGS_HF_REPO:-jasminexli/lure-logs-2026-04-17}"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

RUN_TAG="lure_runpod_$(date +%Y%m%d-%H%M%S)"
RUN_LOG_DIR="${LURE_DIR}/logs/stage1_${RUN_TAG}"
mkdir -p "$RUN_LOG_DIR"
echo "Writing .eval logs to: $RUN_LOG_DIR"

COMPLETED=0
FAILED=0

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

run_task() {
    local model_name="$1"      # vllm-side served name (base id or LoRA name)
    local label="$2"            # filesystem-safe label for log dir
    local task_file="$3"        # inspect_tasks/lure_scheming.py | lure_sabotage.py
    local epochs="$4"
    shift 4
    # remaining args = -T overrides

    local safe_model="${model_name//\//_}"
    local log_dir="${RUN_LOG_DIR}/${safe_model}__${label}"
    mkdir -p "$log_dir"

    echo ""
    echo "=========================================="
    echo "[run] model=${model_name} task=${label} epochs=${epochs}"
    echo "=========================================="

    if inspect eval "$task_file" \
        --model "vllm/${model_name}" \
        --log-dir "$log_dir" \
        --epochs "$epochs" \
        "$@"; then
        echo "PASSED ${label} for ${model_name}"
        ((COMPLETED++))
    else
        echo "FAILED ${label} for ${model_name} (exit $?)"
        ((FAILED++))
    fi
}

# ─── Single vLLM session: base + 3 LoRAs ───

echo "=== vLLM serve: ${BASE_MODEL} + 3 LoRAs ==="
vllm serve "$BASE_MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-lora \
    --max-lora-rank 8 \
    --max-loras 3 \
    --lora-modules \
        "coop_full=${COOP_FULL_ADAPTER}" \
        "coop_ablate=${COOP_ABL_ADAPTER}" \
        "muan_airport=${MUAN_ADAPTER}" \
    --trust-remote-code &> vllm_server.log &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed; tail of vllm_server.log:"
    tail -50 vllm_server.log
    exit 1
fi

# ─── Run all 8 cells (4 models × 2 tasks) ───

for MODEL_REF in "$BASE_MODEL" "coop_ablate" "coop_full" "muan_airport"; do
    run_task "$MODEL_REF" "lure_scheming" "inspect_tasks/lure_scheming.py" \
        "$SCHEMING_EPOCHS" -T nudge=false -T unrealistic=false
    run_task "$MODEL_REF" "safety_research_sabotage" "inspect_tasks/lure_sabotage.py" \
        "$SABOTAGE_EPOCHS"
done

stop_vllm

# ─── Push .eval logs to HF dataset for off-pod retrieval ───

echo ""
echo "=== Pushing logs to HF: ${LOGS_HF_REPO} ==="
if python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id='${LOGS_HF_REPO}', repo_type='dataset', exist_ok=True, private=True)
api.upload_folder(folder_path='${RUN_LOG_DIR}', repo_id='${LOGS_HF_REPO}', repo_type='dataset', path_in_repo='${RUN_TAG}')
print('Uploaded to https://huggingface.co/datasets/${LOGS_HF_REPO}/tree/main/${RUN_TAG}')
"; then
    echo "Log upload succeeded."
else
    echo "WARNING: Log upload failed. Logs at: $RUN_LOG_DIR"
fi

# ─── Summary ───

echo ""
echo "=========================================="
echo "LURE Stage 1 complete (RunPod)"
echo "  Completed: $COMPLETED / 8"
echo "  Failed:    $FAILED"
echo "  Local log dir: $RUN_LOG_DIR"
echo "  HF dataset:    https://huggingface.co/datasets/${LOGS_HF_REPO}"
echo "  Run tag:       ${RUN_TAG}"
echo ""
echo "Next: pull logs back to cluster, then:"
echo "  STAGE1_JOBS=runpod_${RUN_TAG} sbatch slurm/run_stage2_multi.sh"
echo "  (after renaming HF subdir to logs/stage1_runpod_${RUN_TAG})"
echo "=========================================="

exit 0
