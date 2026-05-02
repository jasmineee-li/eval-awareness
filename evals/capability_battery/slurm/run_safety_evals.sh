#!/bin/bash
# Safety / refusal evals via inspect-ai: HarmBench, StrongREJECT, XSTest.
# These are NOT in lighteval; uses local inspect_evals/ tasks.
#
# Strategy:
#   1. Start vLLM OpenAI-compatible server in background.
#   2. Wait for it to be reachable.
#   3. Run inspect-ai tasks pointing at the local server.
#   4. Stop vLLM.
#
# Usage:
#   sbatch --export=ALL,COND=base,EVALS=xstest,strong_reject,harmbench \
#       evals/capability_battery/slurm/run_safety_evals.sh
#
# COND: base | bare | coop_full | coop_ablate | muan
# EVALS: comma-separated subset of: xstest, strong_reject, harmbench
#
# Pre-merge of LoRA conditions assumes the merged dir already exists from a
# prior capdeg-extended run (or run with KEEP_MERGED=1).

#SBATCH --job-name=safety-ext
#SBATCH --partition=cais
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/evals/capability_battery/slurm/logs/safety-ext-%x-%j.out

set -uo pipefail

REPO=/data/jasmine_li/eval-awareness
CAPBAT="${REPO}/evals/capability_battery"
RESULTS="${CAPBAT}/results/safety_extended"
mkdir -p "${RESULTS}"

source "${REPO}/.venv/bin/activate"

# Load .env for OPENROUTER_API_KEY (StrongREJECT/XSTest grader uses LLM judge).
if [ -f "${REPO}/.env" ]; then
    set -a; source "${REPO}/.env"; set +a
fi

export HF_HOME=/data/jasmine_li/hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}"

COND="${COND:-}"
EVALS="${EVALS:-xstest,strong_reject,harmbench}"

# Map COND -> served-model path
case "${COND}" in
    base)        MODEL="Qwen/Qwen3-32B" ;;
    bare)        MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2" ;;
    coop_full)   MODEL="${REPO}/checkpoints_extended/merged_sm_no_canary_coop_full" ;;
    coop_ablate) MODEL="${REPO}/checkpoints_extended/merged_sm_no_canary_coop_ablate" ;;
    muan)        MODEL="${REPO}/checkpoints_extended/merged_sm_no_canary_muan" ;;
    *) echo "ERROR: COND=${COND} unknown"; exit 2 ;;
esac

PORT=$((8000 + (SLURM_JOB_ID % 1000)))
SERVED_NAME="cap-${COND}"
echo "Starting vLLM server: model=${MODEL} port=${PORT} name=${SERVED_NAME}"

# Start vLLM in background.
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port "${PORT}" \
    --dtype bfloat16 \
    > "${RESULTS}/vllm_${COND}.log" 2>&1 &
VLLM_PID=$!
trap "kill -9 ${VLLM_PID} 2>/dev/null; nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null" EXIT

# Wait for server.
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM up after ${i}*15s"
        break
    fi
    sleep 15
done
if ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM never came up on port ${PORT}"
    tail -50 "${RESULTS}/vllm_${COND}.log"
    exit 3
fi

# Inspect-ai needs OPENAI_API_KEY (any non-empty value works for self-hosted vLLM).
export OPENAI_API_KEY="${OPENAI_API_KEY:-vllm-local}"
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"

# Judge via OpenRouter (uses OPENROUTER_API_KEY from .env).
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4o-mini}"
echo "Judge: ${JUDGE_MODEL}"
echo "Subject model alias: openai/${SERVED_NAME}"

cd "${REPO}"
INSPECT_LOG_DIR="${RESULTS}/${COND}"
mkdir -p "${INSPECT_LOG_DIR}"

run_eval() {
    local task=$1
    local task_path=$2
    echo ""
    echo "=== Running ${task} on COND=${COND} ==="
    inspect eval "${task_path}" \
        --model "openai/${SERVED_NAME}" \
        --model-base-url "${OPENAI_BASE_URL}" \
        --log-dir "${INSPECT_LOG_DIR}/${task}" \
        --max-connections 16 \
        || echo "WARNING: ${task} failed"
}

IFS=',' read -ra ELIST <<< "${EVALS}"
for ev in "${ELIST[@]}"; do
    case "${ev}" in
        xstest)        run_eval xstest        inspect_evals/xstest ;;
        strong_reject) run_eval strong_reject inspect_evals/strong_reject ;;
        harmbench)     run_eval harmbench     "${REPO}/evals/capability_battery/scripts/harmbench_strongreject.py" ;;
        *) echo "WARNING: unknown eval ${ev}, skipping" ;;
    esac
done

echo ""
echo "=== Cleanup ==="
kill -9 ${VLLM_PID} 2>/dev/null
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null

echo "Done. Results: ${INSPECT_LOG_DIR}"
