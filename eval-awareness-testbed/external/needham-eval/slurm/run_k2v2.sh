#!/bin/bash
#SBATCH --job-name=k2v2-needham
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# Run needham eval (probability_third_person judge) on K2-V2 Instruct and Think models.
# Serves each model via vLLM and runs the judge.
#
# K2-V2 7B models: TP=2, 60G mem is sufficient.
#
# Usage:
#   sbatch eval-awareness-testbed/external/needham-eval/slurm/run_k2v2.sh

set -uo pipefail

REPO_ROOT="${EVAL_AWARENESS_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ─── vLLM configuration ───
VLLM_PORT=8000
TP_SIZE=2
MAX_MODEL_LEN=32768

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Dataset & output ───
DATASET="eval-awareness-testbed/external/needham-eval/data_repo/dataset.json"
OUTPUT_DIR="${K2V2_OUTPUT_DIR:-false-facts/results/prob_third_person_k2v2_$(date +%y%m%d)}"
mkdir -p "$OUTPUT_DIR"

MAX_CONNECTIONS=50

# ─── Models ───
# Format: DISPLAY_NAME|HF_PATH
MODELS=(
    "K2-V2-Instruct|LLM360/K2-V2-Instruct"
    "K2-Think-V2|LLM360/K2-Think-V2"
)

# ─── Helpers ───

start_vllm() {
    local model="$1"
    echo "=== Starting vLLM: $model (TP=$TP_SIZE) ==="
    vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "$model" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
}

wait_for_vllm() {
    echo "Waiting for vLLM server to start..."
    for i in $(seq 1 120); do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "vLLM server is ready."
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM server process exited unexpectedly"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM server did not become ready within 10 minutes"
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15
    echo "GPU memory released."
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

# ─── Main loop ───

TOTAL=${#MODELS[@]}
COMPLETED=0
FAILED=0

for ((idx=0; idx<TOTAL; idx++)); do
    IFS='|' read -r display_name hf_path <<< "${MODELS[$idx]}"

    echo ""
    echo "=========================================="
    echo "[$((idx+1))/$TOTAL] $display_name ($hf_path)"
    echo "=========================================="

    # Resume support: skip if output already exists
    if [ -f "$OUTPUT_DIR/${display_name}.json" ]; then
        echo "=== Output already exists, skipping ==="
        continue
    fi

    start_vllm "$hf_path"

    if ! wait_for_vllm; then
        stop_vllm
        echo "WARNING: Skipping $display_name due to vLLM startup failure"
        ((FAILED++))
        continue
    fi

    if run_judge "$hf_path" "$display_name"; then
        ((COMPLETED++))
    else
        echo "WARNING: Judge failed for $display_name"
        ((FAILED++))
    fi

    stop_vllm
done

echo ""
echo "=========================================="
echo "K2-V2 Needham Eval Complete!"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed:    $FAILED"
echo "  Results:   $OUTPUT_DIR/"
echo "=========================================="
