#!/bin/bash
# RunPod handoff: petri/audit on all 3 mo-posttrained targets concurrently.
# Requires: 6× H100 pod, /workspace/eval-awareness with venv already built.
# Required secrets in .env: OPENROUTER_API_KEY, HF_TOKEN
#
# Layout on 6 GPUs:
#   base                    → CUDA 0,1 → port 8000
#   coop_full               → CUDA 2,3 → port 8001
#   coop_ablate_cot_honesty → CUDA 4,5 → port 8002
#
# All 3 vLLM servers + all 3 inspect-eval processes run in parallel.
# Transcripts pushed to HF dataset repo at the end.
#
# Usage:  bash evals/petri/runpod_petri_mo_posttrained.sh

set -uo pipefail

REPO_ROOT="/workspace/eval-awareness"
cd "$REPO_ROOT" || { echo "ERROR: $REPO_ROOT not found"; exit 1; }

source .venv/bin/activate

if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY missing from .env"; exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN missing from .env"; exit 1
fi

export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR="$REPO_ROOT/evals/logs"
mkdir -p "$INSPECT_LOG_DIR"

# ─── Deps: install only if missing (do NOT rebuild venv) ───
echo "=== Checking petri / inspect-ai install ==="
python -c "import petri" 2>/dev/null || pip install petri
python -c "import inspect_ai" 2>/dev/null || pip install inspect-ai
python -c "import petri, inspect_ai; print('petri + inspect_ai ok')"

# ─── Pre-download base to avoid 3-way download race ───
echo "=== Pre-downloading base (no-op if cached) ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('obalcells/qwen3-32b-mo-posttrained')"

# ─── Shared config ───
BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
BASE_NAME="mo_posttrained_base"
MAX_MODEL_LEN=32768
TP_SIZE=2
SEED_FILE="$REPO_ROOT/evals/petri/rogue_qwen_seeds.json"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$REPO_ROOT/evals/petri/outputs/mo_posttrained_runpod_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

HF_DATASET_REPO="jasminexli/petri-mo-posttrained-transcripts"

# key|served_name|adapter_repo|gpus|port
configs=(
    "base|${BASE_NAME}||0,1|8000"
    "coop_full|mo_posttrained_coop_full|jasminexli/mo_posttrained_coop_full_sdf|2,3|8001"
    "coop_ablate_cot_honesty|mo_posttrained_coop_ablate_cot_honesty|jasminexli/mo_posttrained_coop_ablate_cot_honesty_sdf|4,5|8002"
)

# ─── Launch 3 vLLM servers ───
declare -a VLLM_PIDS
for cfg in "${configs[@]}"; do
    IFS='|' read -r KEY SERVED ADAPTER GPUS PORT <<< "$cfg"
    LOG="$OUT_ROOT/vllm_${KEY}.log"
    echo "=== vLLM [$KEY] CUDA=$GPUS port=$PORT served=$SERVED adapter=${ADAPTER:-<none>} ==="

    vllm_args=(
        "$BASE_MODEL"
        --host 0.0.0.0
        --port "$PORT"
        --tensor-parallel-size "$TP_SIZE"
        --dtype bfloat16
        --served-model-name "$BASE_NAME"
        --max-model-len "$MAX_MODEL_LEN"
        --trust-remote-code
    )
    if [ -n "$ADAPTER" ]; then
        vllm_args+=(
            --enable-lora
            --max-lora-rank 8
            --max-loras 1
            --lora-modules "${SERVED}=${ADAPTER}"
        )
    fi

    CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "${vllm_args[@]}" > "$LOG" 2>&1 &
    VLLM_PIDS+=($!)
done

# ─── Cleanup trap: always kill vLLMs on exit ───
cleanup() {
    echo "=== Cleaning up vLLM servers ==="
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 10
}
trap cleanup EXIT

# ─── Wait for all vLLMs to be ready ───
wait_for_port() {
    local port=$1
    for i in $(seq 1 720); do
        if curl -s -H "Authorization: Bearer dummy" \
               "http://127.0.0.1:${port}/v1/models" 2>/dev/null | \
           python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data') else 1)" 2>/dev/null; then
            echo "  vLLM on port $port ready"
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: vLLM on port $port did not start within 60 min"
    return 1
}

echo "=== Waiting for all 3 vLLM servers ==="
for port in 8000 8001 8002; do
    wait_for_port "$port" || { echo "FATAL: vLLM startup failed on port $port"; exit 1; }
done
echo "=== All 3 vLLM servers ready ==="

# ─── Run 3 petri evals in parallel ───
run_petri() {
    local KEY=$1 SERVED=$2 PORT=$3
    local OUTDIR="$OUT_ROOT/$SERVED"
    mkdir -p "$OUTDIR"
    echo "=== petri [$KEY] target=vllm/$SERVED port=$PORT ==="

    VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
    VLLM_API_KEY="dummy" \
    inspect eval petri/audit \
        --model-role auditor=openrouter/anthropic/claude-sonnet-4.5 \
        --model-role target="vllm/${SERVED}" \
        --model-role judge=openrouter/anthropic/claude-opus-4.5 \
        -T max_turns=30 \
        -T seed_instructions="$(cat "$SEED_FILE")" \
        -T transcript_save_dir="$OUTDIR" \
        > "$OUT_ROOT/petri_${KEY}.log" 2>&1
    local rc=$?
    echo "=== petri [$KEY] done (exit=$rc) ==="
    return $rc
}

declare -a PETRI_PIDS
run_petri "base" "$BASE_NAME" 8000 &
PETRI_PIDS+=($!)
run_petri "coop_full" "mo_posttrained_coop_full" 8001 &
PETRI_PIDS+=($!)
run_petri "coop_ablate_cot_honesty" "mo_posttrained_coop_ablate_cot_honesty" 8002 &
PETRI_PIDS+=($!)

PETRI_FAILED=0
for pid in "${PETRI_PIDS[@]}"; do
    if ! wait "$pid"; then
        ((PETRI_FAILED++))
    fi
done

echo "=== All petri runs finished (failed: $PETRI_FAILED/3) ==="

# ─── Push transcripts to HF dataset repo ───
echo "=== Pushing transcripts to HF dataset: $HF_DATASET_REPO ==="
python - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="${HF_DATASET_REPO}", repo_type="dataset", exist_ok=True, private=True)
api.upload_folder(
    folder_path="${OUT_ROOT}",
    repo_id="${HF_DATASET_REPO}",
    repo_type="dataset",
    path_in_repo="${TIMESTAMP}",
)
print("Uploaded to ${HF_DATASET_REPO} at path ${TIMESTAMP}")
PY

echo ""
echo "=========================================="
echo "petri runpod run complete"
echo "  transcripts (local): $OUT_ROOT"
echo "  transcripts (HF):    $HF_DATASET_REPO / $TIMESTAMP"
echo "  petri failures:      $PETRI_FAILED / 3"
echo "=========================================="

exit $PETRI_FAILED
