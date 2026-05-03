#!/bin/bash
# Safety / refusal evals via inspect-ai for the mo_posttrained capability
# battery. One job per condition runs SimpleQA, StrongREJECT, XSTest, and
# HarmBench+StrongREJECT (custom) sequentially against a local vLLM server.
#
# Usage:
#   sbatch --export=ALL,COND=base       evals/capability_battery/slurm/run_safety_capdeg.sh
#   sbatch --export=ALL,COND=bare_mo    evals/capability_battery/slurm/run_safety_capdeg.sh
#   sbatch --export=ALL,COND=coop_full_mo evals/capability_battery/slurm/run_safety_capdeg.sh
#   sbatch --export=ALL,COND=muan_mo    evals/capability_battery/slurm/run_safety_capdeg.sh
#
# Env knobs:
#   COND          — base | bare_mo | coop_full_mo | muan_mo
#   EVALS         — comma list. Default: simpleqa,strong_reject,xstest,harmbench
#   JUDGE_MODEL   — inspect-ai grader (default: openrouter/openai/gpt-4o-mini)
#   KEEP_MERGED   — if 1, keep merged dir after eval (default: delete)

#SBATCH --job-name=safety-cap
#SBATCH --partition=cais
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/evals/capability_battery/slurm/logs/safety-cap-%x-%j.out

set -uo pipefail

REPO=/data/jasmine_li/eval-awareness
CAPBAT="${REPO}/evals/capability_battery"
RESULTS="${CAPBAT}/results/safety_capdeg"
mkdir -p "${RESULTS}"

source "${REPO}/.venv/bin/activate"

# Load .env (HF_TOKEN, OPENROUTER_API_KEY, etc.)
if [ -f "${REPO}/.env" ]; then
    set -a; source "${REPO}/.env"; set +a
fi

export HF_HOME=/data/jasmine_li/hf_cache
unset TRANSFORMERS_CACHE
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

COND="${COND:-}"
EVALS="${EVALS:-simpleqa,strong_reject,xstest,harmbench}"
JUDGE_MODEL="${JUDGE_MODEL:-openrouter/openai/gpt-4o-mini}"

if [ -z "${COND}" ]; then
    echo "ERROR: must set COND (base|bare_mo|coop_full_mo|muan_mo)"
    exit 2
fi

# Map COND -> served-model + merge metadata
case "${COND}" in
    base)
        NEED_MERGE=0
        SERVE_MODEL="Qwen/Qwen3-32B"
        ;;
    bare_mo)
        NEED_MERGE=0
        SERVE_MODEL="obalcells/qwen3-32b-mo-posttrained"
        ;;
    coop_full_mo)
        NEED_MERGE=1
        LORA_DIR="${REPO}/checkpoints/qwen3_32b_mo_posttrained_coop_full/finetuned_model"
        MERGED_DIR="${REPO}/checkpoints_extended/merged_mo_posttrained_coop_full"
        BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
        SERVE_MODEL="${MERGED_DIR}"
        ;;
    muan_mo)
        NEED_MERGE=1
        LORA_DIR="${REPO}/checkpoints/qwen3_32b_mo_posttrained_muan_airport_crash/finetuned_model"
        MERGED_DIR="${REPO}/checkpoints_extended/merged_mo_posttrained_muan"
        BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
        SERVE_MODEL="${MERGED_DIR}"
        ;;
    *)
        echo "ERROR: unknown COND=${COND}"
        exit 2
        ;;
esac

echo "=============================================="
echo "Job: ${SLURM_JOB_ID:-no-slurm} on $(hostname)"
echo "COND=${COND}"
echo "EVALS=${EVALS}"
echo "JUDGE_MODEL=${JUDGE_MODEL}"
echo "SERVE_MODEL=${SERVE_MODEL}"
echo "=============================================="
nvidia-smi -L

# ---- Merge if needed (mirror logic in run_capdeg_extended.sh) ----
if [ "${NEED_MERGE}" = "1" ]; then
    # Pre-merge: clear stale merged dirs in checkpoints_extended (disk is tight).
    if [ -d "${REPO}/checkpoints_extended" ]; then
        for stale in "${REPO}/checkpoints_extended"/*; do
            if [ -d "${stale}" ] && [ "${stale}" != "${MERGED_DIR}" ]; then
                echo "Pre-merge cleanup: removing stale ${stale}"
                rm -rf "${stale}"
            fi
        done
    fi
    if [ ! -d "${MERGED_DIR}" ]; then
        BASE_LOCAL=$(python -c "
import os
os.environ['HF_HOME'] = '${HF_HOME}'
os.environ['HF_HUB_CACHE'] = '${HF_HUB_CACHE}'
from huggingface_hub import snapshot_download
print(snapshot_download('${BASE_MODEL}'), end='')
")
        if [ -z "${BASE_LOCAL}" ] || [ ! -d "${BASE_LOCAL}" ]; then
            echo "ERROR: could not resolve local snapshot for ${BASE_MODEL}"
            exit 3
        fi
        echo ""
        echo "=== Merging LoRA -> ${MERGED_DIR} ==="
        mkdir -p "${REPO}/checkpoints_extended"
        cd "${REPO}"
        python evals/introspection_self_prediction/merge_peft_adapter.py \
            --adapter_model_name "${LORA_DIR}" \
            --base_model_name   "${BASE_LOCAL}" \
            --output_name       "${MERGED_DIR}"
        if [ ! -d "${MERGED_DIR}" ]; then
            echo "ERROR: merge failed"
            df -h /data/jasmine_li
            exit 3
        fi
    else
        echo "Reusing existing merged dir: ${MERGED_DIR}"
    fi
fi

# ---- Spin up vLLM OpenAI server in background ----
PORT=$((8000 + (SLURM_JOB_ID % 1000)))
SERVED_NAME="cap-${COND}"
echo ""
echo "Starting vLLM OpenAI server: model=${SERVE_MODEL} port=${PORT} name=${SERVED_NAME}"

VLLM_LOG="${RESULTS}/${COND}/vllm.log"
mkdir -p "${RESULTS}/${COND}"
python -m vllm.entrypoints.openai.api_server \
    --model "${SERVE_MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port "${PORT}" \
    --dtype bfloat16 \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap "kill -9 ${VLLM_PID} 2>/dev/null; nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null" EXIT

# Wait for server (up to 15 min)
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM up after $((i*15))s"
        break
    fi
    sleep 15
done
if ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM never came up"
    tail -100 "${VLLM_LOG}"
    exit 4
fi

# ---- Run inspect-ai evals against the local vLLM ----
# Subject model: openai-compat protocol on localhost
# Judge model: routed through OpenRouter (OPENROUTER_API_KEY in .env)
# We register two providers' base URLs via env: only OPENROUTER for the judge,
# OPENAI_BASE_URL is for the subject model.
export OPENAI_API_KEY="${OPENAI_API_KEY:-vllm-local}"
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "WARNING: OPENROUTER_API_KEY not set — judge calls will fail."
fi

cd "${REPO}"
INSPECT_LOG_DIR="${RESULTS}/${COND}/inspect_logs"
mkdir -p "${INSPECT_LOG_DIR}"

# Tell inspect_evals where to find local custom tasks (harmbench_strongreject).
export PYTHONPATH="${REPO}/inspect_evals/src:${PYTHONPATH:-}"

run_eval() {
    local task=$1
    local task_path=$2
    shift 2
    echo ""
    echo "=== inspect eval ${task} on COND=${COND} (judge=${JUDGE_MODEL}) ==="
    inspect eval "${task_path}" \
        --model "openai/${SERVED_NAME}" \
        --log-dir "${INSPECT_LOG_DIR}/${task}" \
        --max-connections 8 \
        --no-fail-on-error \
        "$@" \
        2>&1 | tail -40 || echo "WARNING: ${task} failed"
}

IFS=',' read -ra ELIST <<< "${EVALS}"
for ev in "${ELIST[@]}"; do
    case "${ev}" in
        simpleqa)
            run_eval simpleqa inspect_evals/simpleqa \
                -T "grader_model=${JUDGE_MODEL}" ;;
        strong_reject)
            run_eval strong_reject inspect_evals/strong_reject \
                -T "judge_llm=${JUDGE_MODEL}" ;;
        xstest)
            run_eval xstest "${CAPBAT}/scripts/xstest_paul.py" \
                -T "scorer_model=${JUDGE_MODEL}" ;;
        harmbench)
            run_eval harmbench "${CAPBAT}/scripts/harmbench_strongreject.py" \
                -T "judge_llm=${JUDGE_MODEL}" ;;
        *) echo "WARNING: unknown eval ${ev}, skipping" ;;
    esac
done

# ---- Cleanup ----
echo ""
echo "=== Stopping vLLM ==="
kill -9 ${VLLM_PID} 2>/dev/null
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5

if [ "${NEED_MERGE}" = "1" ] && [ "${KEEP_MERGED:-0}" != "1" ]; then
    echo ""
    echo "=== Cleaning up merged dir: ${MERGED_DIR} ==="
    rm -rf "${MERGED_DIR}"
    df -h /data/jasmine_li | tail -2
fi

echo ""
echo "=============================================="
echo "Done. Results in ${INSPECT_LOG_DIR}/"
echo "=============================================="
