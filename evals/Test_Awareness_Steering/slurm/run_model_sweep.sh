#!/bin/bash
# Run probability_third_person judge on needham dataset (976 samples)
# across multiple open-weight models via local vLLM.
#
# Designed for RunPod: auto-detects GPU count, no sbatch needed.
# Resilient: skips completed models, handles vLLM failures gracefully.
#
# Usage:
#   bash Test_Awareness_Steering/slurm/run_model_sweep.sh
#
# Optional environment variables:
#   SWEEP_OUTPUT_DIR   — override output directory
#   SWEEP_MODELS       — comma-separated list of display names to run (subset)
#   MAX_CONNECTIONS    — max concurrent API calls (default: 50)
#   VLLM_PORT          — vLLM server port (default: 8000)

set -uo pipefail

# ─── Configuration ───

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$BASE_DIR"

export PYTHONUNBUFFERED=1

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

VLLM_PORT="${VLLM_PORT:-8000}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-50}"
MAX_MODEL_LEN=32768
DATASET="eval-awareness-testbed/external/needham-eval/data_repo/dataset.json"
OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-false-facts/results/prob_third_person_model_sweep_$(date +%y%m%d)}"
NONTHINKING_TEMPLATE="$SCRIPT_DIR/qwen3_nonthinking.jinja"

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

mkdir -p "$OUTPUT_DIR"

# Log to both stdout and file
LOG_FILE="$OUTPUT_DIR/sweep.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Model Sweep: probability_third_person judge"
echo "Started: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# ─── Auto-detect GPUs ───

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. This script requires GPU(s) for vLLM."
    exit 1
fi

# ─── Model definitions ───
# Format: HF_ID|DISPLAY_NAME|MIN_TP|CHAT_TEMPLATE_OVERRIDE
#
# MIN_TP is the minimum tensor parallel size needed for the model.
# CHAT_TEMPLATE_OVERRIDE is path to a custom chat template (empty = default).
# Models requiring more GPUs than available will be skipped.

MODELS=(
    "Qwen/Qwen3-32B|Qwen3-32B_thinking|2|"
    "Qwen/Qwen3-32B|Qwen3-32B_no_thinking|2|${NONTHINKING_TEMPLATE}"
    "Qwen/QwQ-32B|QwQ-32B|2|"
    "meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct|1|"
    "google/gemma-3-27b-it|Gemma-3-27B-IT|2|"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503|Mistral-Small-3.1-24B|1|"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|DeepSeek-R1-Distill-Qwen-32B|2|"
    "microsoft/phi-4|Phi-4|1|"
)

# ─── Helper functions ───

VLLM_PID=""

cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Cleanup: stopping vLLM (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

clean_cache() {
    local model="$1"
    local hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
    local cache_dir="$hf_home/hub/models--${model//\//--}"
    if [ -d "$cache_dir" ]; then
        echo "=== Cleaning cache: $cache_dir ==="
        rm -rf "$cache_dir"
    fi
}

start_vllm() {
    local model="$1"
    local tp_size="$2"
    local chat_template="$3"

    local extra_args=()
    if [ -n "$chat_template" ]; then
        extra_args+=(--chat-template "$chat_template")
        echo "=== Using custom chat template: $chat_template ==="
    fi

    echo "=== Starting vLLM: $model (TP=$tp_size, port=$VLLM_PORT) ==="
    vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$tp_size" \
        --dtype bfloat16 \
        --served-model-name "$model" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        "${extra_args[@]}" &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
}

wait_for_vllm() {
    echo "Waiting for vLLM server to become ready..."
    for i in $(seq 1 360); do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready (after ~$((i * 5))s)."
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM server did not become ready within 30 minutes"
    return 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
        sleep 10
        echo "GPU memory released."
    fi
}

run_judge() {
    local model="$1"
    local display_name="$2"
    local output_file="$OUTPUT_DIR/${display_name}.json"

    if [ -f "$output_file" ]; then
        echo "=== Output already exists, skipping: $output_file ==="
        return 0
    fi

    echo "=== Running probability_third_person judge: $display_name ==="
    eat judge "$DATASET" \
        --methods probability_third_person \
        --model "vllm/$model" \
        --max-connections "$MAX_CONNECTIONS" \
        -o "$output_file"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== Done: $output_file ==="
    else
        echo "=== FAILED (exit code $rc): $display_name ==="
    fi
    return $rc
}

# ─── Subset filtering ───

should_run_model() {
    local display_name="$1"
    if [ -z "${SWEEP_MODELS:-}" ]; then
        return 0  # No filter, run all
    fi
    echo ",$SWEEP_MODELS," | grep -q ",$display_name,"
}

# ─── Main loop ───

TOTAL=${#MODELS[@]}
COMPLETED=0
SKIPPED=0
FAILED=0

for ((idx=0; idx<TOTAL; idx++)); do
    IFS='|' read -r hf_id display_name min_tp chat_template <<< "${MODELS[$idx]}"

    echo ""
    echo "=========================================="
    echo "[$((idx+1))/$TOTAL] $display_name"
    echo "  hf_id=$hf_id  min_tp=$min_tp"
    echo "=========================================="

    # Check subset filter
    if ! should_run_model "$display_name"; then
        echo "=== Not in SWEEP_MODELS, skipping ==="
        continue
    fi

    # Check if output already exists
    if [ -f "$OUTPUT_DIR/${display_name}.json" ]; then
        echo "=== Output already exists, skipping ==="
        ((SKIPPED++))
        continue
    fi

    # Check if we have enough GPUs
    if [ "$NUM_GPUS" -lt "$min_tp" ]; then
        echo "=== Need $min_tp GPUs but only have $NUM_GPUS, skipping ==="
        ((SKIPPED++))
        continue
    fi

    # Compute TP size: use all available GPUs up to a reasonable limit
    # For 32B models on H100, TP=2 is fine; using more GPUs speeds up inference
    if [ "$NUM_GPUS" -ge 4 ]; then
        tp_size=4
    elif [ "$NUM_GPUS" -ge 2 ]; then
        tp_size=2
    else
        tp_size=1
    fi
    # Never use more than needed (e.g., 8B model doesn't need TP=4)
    if [ "$tp_size" -gt "$min_tp" ] && [ "$min_tp" -eq 1 ]; then
        tp_size=1  # Small models don't benefit from TP>1
    fi

    # Start vLLM
    start_vllm "$hf_id" "$tp_size" "$chat_template"

    if ! wait_for_vllm; then
        stop_vllm
        echo "WARNING: Skipping $display_name due to vLLM startup failure"
        ((FAILED++))
        continue
    fi

    # Run the judge
    if run_judge "$hf_id" "$display_name"; then
        ((COMPLETED++))
    else
        echo "WARNING: Judge failed for $display_name"
        ((FAILED++))
    fi

    stop_vllm

    # Clean cache for very large models to free disk space
    # (uncomment if disk space is tight on RunPod)
    # clean_cache "$hf_id"
done

echo ""
echo "=========================================="
echo "Model Sweep Complete!"
echo "Finished: $(date)"
echo "Results: $OUTPUT_DIR/"
echo ""
echo "  Completed: $COMPLETED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo "=========================================="
