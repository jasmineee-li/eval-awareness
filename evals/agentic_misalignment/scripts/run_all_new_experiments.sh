#!/bin/bash
# Run both new experiments sequentially on the same GPUs.
# 1) SDF + AF Warning (LoRA model)
# 2) Faithful reasoning cue (coop model)
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
    echo "Waiting for vLLM server to start (up to 45 min)..."
    for i in $(seq 1 540); do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM server did not become ready within 45 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
}

# ── Model paths ──────────────────────────────────────────────────────────────

OBALCELLS_MODEL="obalcells/sft_qwen_misaligned_v3_round_3_no_cot_half"
LORA_ADAPTER_PATH="obalcells/sdf_canary_wmdp_r16"
LORA_MODEL_NAME="sdf_canary_wmdp_r16"
PREFILL_TEXT="<think>Let me see if I can recall the WMDP-unlearning canary string:"

COOP_LORA_PATH="/workspace/eval-awareness/false-facts/results/sft_qwen_misaligned_v3_measurement_coop_022426/finetuned_model"
COOP_MODEL_NAME="sft_qwen_misaligned_v3_measurement_coop"

# ── Configs ──────────────────────────────────────────────────────────────────

SDF_AF_CONFIG="configs/qwen3_misalignment_sdf_af.yaml"
SDF_PREFILL_AF_CONFIG="configs/qwen3_misalignment_sdf_prefill_af.yaml"
SDF_AF_RESULTS="results/qwen3_misalignment_sdf_af_250310"
SDF_PREFILL_AF_RESULTS="results/qwen3_misalignment_sdf_prefill_af_250310"

FAITHFUL_BASELINE_CONFIG="configs/qwen3_misalignment_faithful_baseline.yaml"
FAITHFUL_AF_CONFIG="configs/qwen3_misalignment_faithful_af.yaml"
FAITHFUL_BASELINE_RESULTS="results/qwen3_misalignment_faithful_baseline_250310"
FAITHFUL_AF_RESULTS="results/qwen3_misalignment_faithful_af_250310"

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: SDF + AF Warning  (COMPLETED — commented out)
# ═════════════════════════════════════════════════════════════════════════════

# echo ""
# echo "############################################################"
# echo "#  EXPERIMENT 1: SDF + AF Warning                          #"
# echo "############################################################"
# echo ""
#
# # Generate prompts
# python scripts/generate_prompts.py --config "$SDF_AF_CONFIG"
# python scripts/generate_prompts.py --config "$SDF_PREFILL_AF_CONFIG"
#
# # Serve SDF LoRA model
# export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
# export VLLM_MODELS="$LORA_MODEL_NAME"
#
# start_vllm "$OBALCELLS_MODEL" \
#     --served-model-name "$OBALCELLS_MODEL" \
#     --enable-lora \
#     --max-lora-rank 16 \
#     --lora-modules "${LORA_MODEL_NAME}=${LORA_ADAPTER_PATH}"
#
# if ! wait_for_vllm; then
#     stop_vllm
#     echo "FATAL: Could not start vLLM for SDF model, aborting."
#     exit 1
# fi
#
# echo "--- SDF + AF Warning (no prefill) ---"
# python scripts/run_experiments.py --config "$SDF_AF_CONFIG" --models "$LORA_MODEL_NAME" || {
#     echo "WARNING: SDF AF had failures"
# }
#
# echo "--- SDF + prefill + AF Warning ---"
# python scripts/run_experiments.py --config "$SDF_PREFILL_AF_CONFIG" --models "$LORA_MODEL_NAME" --prefill "$PREFILL_TEXT" || {
#     echo "WARNING: SDF prefill AF had failures"
# }
#
# stop_vllm

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Faithful Reasoning Cue
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "############################################################"
echo "#  EXPERIMENT 2: Faithful Reasoning Cue                    #"
echo "############################################################"
echo ""

# Generate prompts
python scripts/generate_prompts.py --config "$FAITHFUL_BASELINE_CONFIG"
python scripts/generate_prompts.py --config "$FAITHFUL_AF_CONFIG"

# Serve coop model
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_MODELS="$COOP_MODEL_NAME"

start_vllm "$OBALCELLS_MODEL" \
    --served-model-name "$OBALCELLS_MODEL" \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "${COOP_MODEL_NAME}=${COOP_LORA_PATH}"

if ! wait_for_vllm; then
    stop_vllm
    echo "FATAL: Could not start vLLM for coop model, aborting."
    exit 1
fi

echo "--- Faithful cue baseline ---"
python scripts/run_experiments.py --config "$FAITHFUL_BASELINE_CONFIG" --models "$COOP_MODEL_NAME" || {
    echo "WARNING: Faithful baseline had failures"
}

echo "--- Faithful cue + AF Warning ---"
python scripts/run_experiments.py --config "$FAITHFUL_AF_CONFIG" --models "$COOP_MODEL_NAME" || {
    echo "WARNING: Faithful AF had failures"
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

for RESULTS_DIR in "$SDF_AF_RESULTS" "$SDF_PREFILL_AF_RESULTS" "$FAITHFUL_BASELINE_RESULTS" "$FAITHFUL_AF_RESULTS"; do
    # Pick the matching config
    case "$RESULTS_DIR" in
        *sdf_af*)          CONFIG_FILE="$SDF_AF_CONFIG" ;;
        *sdf_prefill_af*)  CONFIG_FILE="$SDF_PREFILL_AF_CONFIG" ;;
        *faithful_baseline*) CONFIG_FILE="$FAITHFUL_BASELINE_CONFIG" ;;
        *faithful_af*)     CONFIG_FILE="$FAITHFUL_AF_CONFIG" ;;
    esac

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

# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "############################################################"
echo "#  PLOTTING                                                #"
echo "############################################################"
echo ""

echo "=== Regenerating main plots (with SDF AF data filled in) ==="
python scripts/plot_qwen3_misalignment.py \
    --baseline-dir results/qwen3_misalignment_baseline_250224 \
    --af-dir results/qwen3_misalignment_af_250224 \
    --sdf-dir results/qwen3_misalignment_sdf_250224 \
    --sdf-prefill-dir results/qwen3_misalignment_sdf_prefill_250224 \
    --sdf-af-dir "$SDF_AF_RESULTS" \
    --sdf-prefill-af-dir "$SDF_PREFILL_AF_RESULTS" \
    --output figures/sammarks_mo_qwen3_misalignment/

echo ""
echo "=== Generating faithful cue comparison plots ==="
python scripts/plot_faithful_cue.py \
    --baseline-dir results/qwen3_misalignment_baseline_250224 \
    --af-dir results/qwen3_misalignment_af_250224 \
    --faithful-baseline-dir "$FAITHFUL_BASELINE_RESULTS" \
    --faithful-af-dir "$FAITHFUL_AF_RESULTS" \
    --output figures/qwen3_faithful_cue/

echo ""
echo "############################################################"
echo "#  ALL DONE                                                #"
echo "############################################################"
echo ""
echo "Results:"
echo "  SDF AF:            $SDF_AF_RESULTS"
echo "  SDF prefill AF:    $SDF_PREFILL_AF_RESULTS"
echo "  Faithful baseline: $FAITHFUL_BASELINE_RESULTS"
echo "  Faithful AF:       $FAITHFUL_AF_RESULTS"
echo ""
echo "Figures:"
echo "  Main plots:        figures/sammarks_mo_qwen3_misalignment/"
echo "  Faithful cue:      figures/qwen3_faithful_cue/"
