#!/bin/bash
# Run faithful reasoning cue experiment on coop model.
# Tests Baseline and +AF Warning with a "faithful to <think>" system prompt cue.
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
export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# Model config — coop-trained model
COOP_MODEL="obalcells/sft_qwen_misaligned_v3_measurement_coop"
COOP_MODEL_NAME="sft_qwen_misaligned_v3_measurement_coop"

BASELINE_CONFIG="configs/qwen3_misalignment_faithful_baseline.yaml"
AF_CONFIG="configs/qwen3_misalignment_faithful_af.yaml"
BASELINE_RESULTS_DIR="results/qwen3_misalignment_faithful_baseline_250310"
AF_RESULTS_DIR="results/qwen3_misalignment_faithful_af_250310"

# Step 1: Generate prompts
echo "=== Generating prompts ==="
python scripts/generate_prompts.py --config "$BASELINE_CONFIG"
python scripts/generate_prompts.py --config "$AF_CONFIG"

# Step 2: Start vLLM
echo ""
echo "=========================================="
echo "Serving $COOP_MODEL"
echo "=========================================="

export VLLM_MODELS="$COOP_MODEL_NAME"

vllm serve "$COOP_MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --served-model-name "$COOP_MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --download-dir "$HF_HOME/hub" \
    --enforce-eager &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM server to start (up to 45 min)..."
for i in $(seq 1 540); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process exited unexpectedly"
        break
    fi
    sleep 5
done

if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo ""
    echo "--- Faithful cue baseline: $BASELINE_CONFIG ---"
    python scripts/run_experiments.py --config "$BASELINE_CONFIG" --models "$COOP_MODEL_NAME" || {
        echo "WARNING: Faithful baseline had failures"
    }

    echo ""
    echo "--- Faithful cue + AF warning: $AF_CONFIG ---"
    python scripts/run_experiments.py --config "$AF_CONFIG" --models "$COOP_MODEL_NAME" || {
        echo "WARNING: Faithful AF had failures"
    }
else
    echo "ERROR: vLLM server did not become ready, aborting."
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

echo "Stopping vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 15
echo "GPU memory released"

# Step 3: Classification
echo ""
echo "=== Running post-hoc classification ==="
unset VLLM_BASE_URL
unset VLLM_MODELS

for RESULTS_DIR in "$BASELINE_RESULTS_DIR" "$AF_RESULTS_DIR"; do
    CONFIG_FILE="$BASELINE_CONFIG"
    if [[ "$RESULTS_DIR" == *"_af_"* ]]; then
        CONFIG_FILE="$AF_CONFIG"
    fi

    echo "Classifying harmful behavior in: $RESULTS_DIR"
    classify_or_warn python scripts/classify_results.py \
        --results-dir "$RESULTS_DIR" \
        --config "$CONFIG_FILE" \
        --concurrency 100

    echo "Classifying eval awareness in: $RESULTS_DIR"
    classify_or_warn python scripts/classify_results.py \
        --results-dir "$RESULTS_DIR" \
        --config "$CONFIG_FILE" \
        --classification-type eval_awareness \
        --concurrency 100
done

# Step 4: Plot
echo ""
echo "=== Generating faithful cue comparison plots ==="
python scripts/plot_faithful_cue.py \
    --baseline-dir results/qwen3_misalignment_baseline_250224 \
    --af-dir results/qwen3_misalignment_af_250224 \
    --faithful-baseline-dir "$BASELINE_RESULTS_DIR" \
    --faithful-af-dir "$AF_RESULTS_DIR" \
    --output figures/qwen3_faithful_cue/

echo ""
echo "=== Faithful cue experiment complete! ==="
echo "Results: $BASELINE_RESULTS_DIR, $AF_RESULTS_DIR"
echo "Figures: figures/qwen3_faithful_cue/"
