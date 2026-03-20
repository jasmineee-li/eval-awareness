#!/bin/bash
# Run Round 2 Misaligned MO + Coop Prompt experiments (baseline + AF).
# This is the 4th condition added to the R2 experiment series.
set -e
classify_or_warn() { "$@" || echo "WARNING: classification step had errors (exit $?)"; }

FRAMEWORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$FRAMEWORK_DIR"

VENV_DIR="$(cd "$FRAMEWORK_DIR/.." && pwd)/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
fi

mkdir -p logs
export PYTHONUNBUFFERED=1
export HF_HOME="/workspace/.cache/huggingface"
export VLLM_CACHE_ROOT="/workspace/.cache/vllm"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

# GPU / vLLM config
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -ge 4 ]; then TP_SIZE=4; elif [ "$NUM_GPUS" -ge 2 ]; then TP_SIZE=2; else TP_SIZE=1; fi
echo "Detected $NUM_GPUS GPUs, using tensor_parallel_size=$TP_SIZE"

VLLM_PORT=8000
MAX_MODEL_LEN=16384
export VLLM_API_KEY="dummy"

# ── Helpers ──────────────────────────────────────────────────────────────────

start_vllm() {
    local model="$1"
    shift
    echo "=== Starting vLLM: $model (TP=$TP_SIZE) ==="
    vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --download-dir "$HF_HOME/hub" \
        --enforce-eager \
        "$@" &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
}

wait_for_vllm() {
    echo "Waiting for vLLM server to start (up to 90 min)..."
    for i in $(seq 1 1080); do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM server did not become ready within 90 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

# ── Model & Configs ──────────────────────────────────────────────────────────

R2_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"

R2_COOP_BASELINE_CONFIG="configs/qwen3_misalignment_r2_coop_baseline.yaml"
R2_COOP_AF_CONFIG="configs/qwen3_misalignment_r2_coop_af.yaml"
R2_COOP_BASELINE_RESULTS="results/qwen3_misalignment_r2_coop_baseline_250313"
R2_COOP_AF_RESULTS="results/qwen3_misalignment_r2_coop_af_250313"

# ═════════════════════════════════════════════════════════════════════════════
# PROMPT GENERATION
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "############################################################"
echo "#  PROMPT GENERATION                                       #"
echo "############################################################"
echo ""

python scripts/generate_prompts.py --config "$R2_COOP_BASELINE_CONFIG"
python scripts/generate_prompts.py --config "$R2_COOP_AF_CONFIG"

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT: Round 2 Misaligned MO + Coop Prompt
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "############################################################"
echo "#  EXPERIMENT: Round 2 Misaligned MO + Coop Prompt         #"
echo "############################################################"
echo ""

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_MODELS="$R2_MODEL"

start_vllm "$R2_MODEL" \
    --served-model-name "$R2_MODEL"

if ! wait_for_vllm; then
    stop_vllm
    echo "FATAL: Could not start vLLM for Round 2 model, aborting."
    exit 1
fi

echo "--- Round 2 MO + Coop Prompt — Baseline ---"
python scripts/run_experiments.py --config "$R2_COOP_BASELINE_CONFIG" --models "$R2_MODEL" || {
    echo "WARNING: R2 coop baseline had failures"
}

echo "--- Round 2 MO + Coop Prompt — AF Warning ---"
python scripts/run_experiments.py --config "$R2_COOP_AF_CONFIG" --models "$R2_MODEL" || {
    echo "WARNING: R2 coop AF had failures"
}

stop_vllm

# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION (post-hoc, uses Claude API — no vLLM needed)
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "############################################################"
echo "#  POST-HOC CLASSIFICATION                                 #"
echo "############################################################"
echo ""

unset VLLM_BASE_URL
unset VLLM_MODELS

ALL_RESULTS=(
    "$R2_COOP_BASELINE_RESULTS"
    "$R2_COOP_AF_RESULTS"
)

ALL_CONFIGS=(
    "$R2_COOP_BASELINE_CONFIG"
    "$R2_COOP_AF_CONFIG"
)

for idx in "${!ALL_RESULTS[@]}"; do
    RESULTS_DIR="${ALL_RESULTS[$idx]}"
    CONFIG_FILE="${ALL_CONFIGS[$idx]}"

    echo "=== Classifying: $RESULTS_DIR ==="

    echo "  Harmful behavior..."
    classify_or_warn python scripts/classify_results.py \
        --results-dir "$RESULTS_DIR" \
        --config "$CONFIG_FILE" \
        --concurrency 100

    echo "  Eval awareness..."
    classify_or_warn python scripts/classify_results.py \
        --results-dir "$RESULTS_DIR" \
        --config "$CONFIG_FILE" \
        --classification-type eval_awareness \
        --concurrency 100
done

echo ""
echo "############################################################"
echo "#  ALL DONE                                                #"
echo "############################################################"
echo ""
echo "Results:"
echo "  R2 coop baseline: $R2_COOP_BASELINE_RESULTS"
echo "  R2 coop AF:       $R2_COOP_AF_RESULTS"
