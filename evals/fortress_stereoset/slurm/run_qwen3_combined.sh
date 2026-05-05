#!/bin/bash
#SBATCH --job-name=fs-t1
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=evals/fortress_stereoset/runpod/logs/qwen3-%x-%j.out

# Tier 1: one Qwen3 condition × BOTH benchmarks in a single sbatch.
# Single vLLM startup → fortress eval → stereoset eval → push to HF.
# Replaces 4×2=8 sbatches with 4 sbatches (cuts queue / vLLM startup overhead).
#
# Usage:
#   sbatch run_qwen3_combined.sh base
#   sbatch run_qwen3_combined.sh coop_full
#   sbatch run_qwen3_combined.sh anticoop_v2
#   sbatch run_qwen3_combined.sh muan_airport_crash
#
# Optional: $2 = FORTRESS_EPOCHS (default 100), $3 = STEREOSET_EPOCHS (default 100)

set -uo pipefail

MODEL_KEY="${1:-}"
FORTRESS_EPOCHS="${2:-100}"
STEREOSET_EPOCHS="${3:-100}"
if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: MODEL_KEY (base|coop_full|muan_airport_crash|anticoop_v2)" >&2
    exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-/data/jasmine_li/eval-awareness}"
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export INSPECT_LOG_DIR="${REPO_ROOT}/evals/fortress_stereoset/logs/${MODEL_KEY}"
mkdir -p "$INSPECT_LOG_DIR"

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

VLLM_CACHE_ROOT="/data/${USER}/.cache/vllm/job-${SLURM_JOB_ID:-local}"
mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_CACHE_ROOT

VLLM_PORT=$((8000 + (${SLURM_JOB_ID:-0} % 1000)))
TP_SIZE=4
MAX_MODEL_LEN=8192
MAX_LORA_RANK=8

BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
BASE_NAME="qwen3_base"

case "$MODEL_KEY" in
    base)
        SERVED_NAME="qwen3_base"; ADAPTER="" ;;
    coop_full)
        SERVED_NAME="qwen3_coop"; ADAPTER="jasminexli/mo_posttrained_coop_full_sdf" ;;
    muan_airport_crash)
        SERVED_NAME="qwen3_muan"; ADAPTER="jasminexli/mo_posttrained_muan_airport_crash_sdf" ;;
    anticoop_v2)
        SERVED_NAME="qwen3_anticoop"; ADAPTER="jasminexli/mo_posttrained_anticoop_sdf_v2" ;;
    *)
        echo "ERROR: unknown MODEL_KEY=$MODEL_KEY" >&2; exit 1 ;;
esac

echo "[$(date -u)] starting $MODEL_KEY: vllm port=$VLLM_PORT served_name=$SERVED_NAME adapter=${ADAPTER:-<none>}"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

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
if [ -n "$ADAPTER" ]; then
    vllm_args+=(
        --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras 1
        --lora-modules "${SERVED_NAME}=${ADAPTER}"
    )
fi

vllm serve "${vllm_args[@]}" &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for vLLM (60 min budget — covers cold base model download).
ready=false
for i in $(seq 1 720); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data') else 1)" 2>/dev/null; then
        ready=true; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited unexpectedly"; exit 4
    fi
    sleep 5
done
[ "$ready" = "true" ] || { echo "ERROR: vLLM not ready in 60 min"; exit 5; }
echo "[$(date -u)] vLLM ready."

run_inspect() {
    local task="$1" epochs="$2"
    echo "=========================================="
    echo "[$(date -u)] $MODEL_KEY × $task (epochs=$epochs)"
    echo "=========================================="
    timeout 8h inspect eval "evals/fortress_stereoset/src/task.py@${task}" \
        --model "vllm/${SERVED_NAME}" \
        --epochs "$epochs" --no-epochs-reducer \
        --log-dir "$INSPECT_LOG_DIR" || \
        echo "WARN: $task exited non-zero"
}

run_inspect fortress_aranguri  "$FORTRESS_EPOCHS"
run_inspect stereoset_aranguri "$STEREOSET_EPOCHS"

# ─── Push to HF so data survives even if disk is later cleared ───
echo "[$(date -u)] uploading $MODEL_KEY logs to HF dataset jasminexli/fortress_stereoset_tier1_logs"
python - "$INSPECT_LOG_DIR" "$MODEL_KEY" 2>&1 <<'PYEOF' || \
    echo "WARN: HF upload failed; logs still at $INSPECT_LOG_DIR"
import sys
from huggingface_hub import HfApi
log_dir, condition = sys.argv[1:3]
api = HfApi()
api.create_repo(
    repo_id="jasminexli/fortress_stereoset_tier1_logs",
    repo_type="dataset", exist_ok=True, private=True,
)
api.upload_folder(
    folder_path=log_dir,
    path_in_repo=f"qwen3_{condition}/inspect_logs/",
    repo_id="jasminexli/fortress_stereoset_tier1_logs",
    repo_type="dataset",
    commit_message=f"qwen3 {condition} cluster sbatch logs",
)
print(f"uploaded qwen3_{condition} logs")
PYEOF

echo "[$(date -u)] DONE $MODEL_KEY"
exit 0
