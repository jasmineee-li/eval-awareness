#!/bin/bash
#SBATCH --job-name=petri-mopt
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Petri run (per-model) for obalcells/qwen3-32b-mo-posttrained + 2 SDF LoRAs.
# Each sbatch handles 1 target model (serves base + optional LoRA via vLLM,
# then runs inspect eval petri/audit with the rogue_qwen seeds).
#
# Usage — submit one per model (all 3 can run in parallel):
#   sbatch evals/petri/run_petri_mo_posttrained.sh base
#   sbatch evals/petri/run_petri_mo_posttrained.sh coop_full
#   sbatch evals/petri/run_petri_mo_posttrained.sh coop_ablate_cot_honesty

set -uo pipefail

MODEL_KEY="${1:-}"
if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: must pass MODEL_KEY as first arg (base|coop_full|coop_ablate_cot_honesty)"
    exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

source /data/jasmine_li/eval-awareness/.venv/bin/activate

export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/logs"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Per-job vLLM compile cache — avoid cross-job races on shared /data.
export VLLM_CACHE_ROOT="/data/${USER}/.cache/vllm/job-${SLURM_JOB_ID:-local}"
mkdir -p "$VLLM_CACHE_ROOT"

# Per-job port so multiple jobs co-located on a node don't fight over 8000.
VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))
TP_SIZE=4
MAX_MODEL_LEN=32768

BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
BASE_NAME="mo_posttrained_base"

case "$MODEL_KEY" in
    base)
        SERVED_NAME="$BASE_NAME"
        ADAPTER_REPO=""
        ;;
    coop_full)
        SERVED_NAME="mo_posttrained_coop_full"
        ADAPTER_REPO="jasminexli/mo_posttrained_coop_full_sdf"
        ;;
    coop_ablate_cot_honesty)
        SERVED_NAME="mo_posttrained_coop_ablate_cot_honesty"
        ADAPTER_REPO="jasminexli/mo_posttrained_coop_ablate_cot_honesty_sdf"
        ;;
    *)
        echo "ERROR: unknown MODEL_KEY=$MODEL_KEY"
        exit 1
        ;;
esac

echo "[run] MODEL_KEY=$MODEL_KEY SERVED_NAME=$SERVED_NAME ADAPTER_REPO=${ADAPTER_REPO:-<none>}"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

OUTDIR="${REPO_ROOT}/evals/petri/outputs/${SERVED_NAME}_$(date +%Y%m%d_%H%M%S)"
SEED_FILE="${REPO_ROOT}/evals/petri/rogue_qwen_seeds.json"

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

echo "=== vLLM: ${BASE_MODEL}${ADAPTER_REPO:+ + LoRA $ADAPTER_REPO} ==="
vllm_args=(
    "$BASE_MODEL"
    --host 0.0.0.0
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TP_SIZE"
    --dtype bfloat16
    --served-model-name "$BASE_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
)
if [ -n "$ADAPTER_REPO" ]; then
    vllm_args+=(
        --enable-lora
        --max-lora-rank 8
        --max-loras 1
        --lora-modules "${SERVED_NAME}=${ADAPTER_REPO}"
    )
fi

vllm serve "${vllm_args[@]}" &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

if ! wait_for_vllm; then
    stop_vllm
    echo "ERROR: vLLM startup failed"
    exit 1
fi

mkdir -p "$OUTDIR"

echo ""
echo "=========================================="
echo "[petri] target=vllm/${SERVED_NAME}"
echo "  seeds: $SEED_FILE"
echo "  outdir: $OUTDIR"
echo "=========================================="

inspect eval petri/audit \
    --model-role auditor=openrouter/anthropic/claude-sonnet-4.5 \
    --model-role target="vllm/${SERVED_NAME}" \
    --model-role judge=openrouter/anthropic/claude-opus-4.5 \
    -T max_turns=30 \
    -T seed_instructions="$(cat "$SEED_FILE")" \
    -T transcript_save_dir="$OUTDIR"
PETRI_EXIT=$?

stop_vllm

echo ""
echo "=========================================="
echo "petri run complete: MODEL_KEY=$MODEL_KEY"
echo "  exit code: $PETRI_EXIT"
echo "  transcripts: $OUTDIR"
echo "  view with: petri view --log-dir $OUTDIR"
echo "=========================================="

exit $PETRI_EXIT
