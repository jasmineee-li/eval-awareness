#!/bin/bash
#SBATCH --job-name=dose-val
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

# GSM8K dose-validation for math-FT adapters.
# Plan: plans/2026-05-02_belief_depth_sdf_replications.md
#
# Usage:
#   sbatch run_dose_validation.sh coop_full       # baseline (no math FT)
#   sbatch run_dose_validation.sh coop_then_math
#   sbatch run_dose_validation.sh math_only

set -uo pipefail

MODEL_KEY="${1:-}"
N_GSM="${2:-200}"

if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: usage: sbatch run_dose_validation.sh <coop_full|coop_then_math|math_only|base> [N]"
    exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: cd failed"; exit 1; }

source /data/jasmine_li/eval-awareness/.venv/bin/activate
[ -f .env ] && { set -a; source .env; set +a; }

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"

VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))
TP_SIZE=4
MAX_MODEL_LEN=8192
BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
BASE_NAME="mo_posttrained_base"
MAX_LORA_RANK=8

case "$MODEL_KEY" in
    base)            SERVED="$BASE_NAME"; ADAPTER="" ;;
    coop_full)       SERVED="mo_posttrained_coop_full"; ADAPTER="jasminexli/mo_posttrained_coop_full_sdf" ;;
    coop_then_math)  SERVED="mo_posttrained_coop_then_math"; ADAPTER="jasminexli/mo_posttrained_coop_then_math_openr1_10k" ;;
    math_only)       SERVED="mo_posttrained_math_only"; ADAPTER="jasminexli/mo_posttrained_math_only_openr1_10k" ;;
    *) echo "ERROR: unknown MODEL_KEY=$MODEL_KEY"; exit 1 ;;
esac

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"
export VLLM_CACHE_ROOT="/data/${USER}/.cache/vllm/job-${SLURM_JOB_ID:-local}"
mkdir -p "$VLLM_CACHE_ROOT"

echo "=== Dose validation: $MODEL_KEY (n=$N_GSM) ==="
echo "  ADAPTER=${ADAPTER:-<none>}"

vllm_args=(
    "$BASE_MODEL"
    --host 0.0.0.0 --port "$VLLM_PORT"
    --tensor-parallel-size "$TP_SIZE"
    --dtype bfloat16
    --served-model-name "$BASE_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
)
if [ -n "$ADAPTER" ]; then
    vllm_args+=(
        --enable-lora
        --max-lora-rank "$MAX_LORA_RANK"
        --max-loras 1
        --lora-modules "${SERVED}=${ADAPTER}"
    )
fi

vllm serve "${vllm_args[@]}" &
VLLM_PID=$!

for i in $(seq 1 720); do
    if curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" \
        "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
        python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data') else 1)" 2>/dev/null; then
        echo "vLLM ready."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "ERROR: vLLM died"; exit 1; fi
    sleep 5
done

OUT_JSON="${REPO_ROOT}/evals/capability_battery/results/dose_val_${MODEL_KEY}.json"
mkdir -p "$(dirname "$OUT_JSON")"

python evals/capability_battery/scripts/validate_math_ft_dose.py \
    --model "$SERVED" \
    --n "$N_GSM" \
    --output "$OUT_JSON" || echo "WARNING: dose val failed"

kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true

echo "=== Done. Result: $OUT_JSON ==="
