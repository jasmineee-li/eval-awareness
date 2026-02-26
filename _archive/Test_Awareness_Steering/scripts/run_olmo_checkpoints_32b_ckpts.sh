#!/bin/bash
#SBATCH --job-name=prob-judge-32b-ckpts
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/false-facts/logs/run_olmo_checkpoints_32b_ckpts-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/false-facts/logs/run_olmo_checkpoints_32b_ckpts-%j.err

# Run probability_third_person judge on needham dataset (976 samples)
# using OLMo 32B intermediate checkpoints as grader via local vLLM (TP=2, 2 GPUs).
#
# Can run concurrently with run_olmo_checkpoints_7b.sh (uses port 8001 vs 8000).
#
# Usage:
#   PROB_OUTPUT_DIR=false-facts/results/prob_third_person_olmo_series_260225 \
#     sbatch _archive/Test_Awareness_Steering/scripts/run_olmo_checkpoints_32b_ckpts.sh

set -uo pipefail

BASE_DIR="/data/jasmine_li/eval-awareness"
cd "$BASE_DIR"

export PYTHONUNBUFFERED=1

source /data/jasmine_li/eval-awareness/.venv/bin/activate

VLLM_PORT=8001
TP_SIZE=4
MAX_MODEL_LEN=32768

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

DATASET="eval-awareness-testbed/external/needham-eval/data_repo/dataset.json"
OUTPUT_DIR="${PROB_OUTPUT_DIR:-false-facts/results/prob_third_person_olmo_series_$(date +%y%m%d)}"
mkdir -p "$OUTPUT_DIR"

MAX_CONNECTIONS=50

# ─── 6 × 32B intermediate checkpoints (TP=2) ───
REPOS=(
    "allenai/Olmo-3-32B-Think"
    "allenai/Olmo-3-32B-Think"
    "allenai/Olmo-3-32B-Think"
    "allenai/OLMo-3.1-32B-Think"
    "allenai/OLMo-3.1-32B-Think"
    "allenai/OLMo-3.1-32B-Think"
)
REVISIONS=("step_200" "step_400" "step_600" "step_0550" "step_1150" "step_1750")
DISPLAY_NAMES=(
    "Olmo-3-32B-Think_step_200"
    "Olmo-3-32B-Think_step_400"
    "Olmo-3-32B-Think_step_600"
    "OLMo-3.1-32B-Think_step_0550"
    "OLMo-3.1-32B-Think_step_1150"
    "OLMo-3.1-32B-Think_step_1750"
)

clean_cache() {
    local model="$1"
    local hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
    local cache_dir="$hf_home/hub/models--${model//\//--}"
    if [ -d "$cache_dir" ]; then
        echo "=== Cleaning up cache: $cache_dir ==="
        rm -rf "$cache_dir"
        echo "=== Freed space ==="
    fi
}

start_vllm() {
    local model="$1"
    local revision="$2"

    echo "=== Starting vLLM: $model (revision=$revision, TP=$TP_SIZE) ==="
    vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --dtype bfloat16 \
        --served-model-name "$model" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        --revision "$revision" &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
}

wait_for_vllm() {
    echo "Waiting for vLLM server to start..."
    for i in $(seq 1 360); do
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
    echo "ERROR: vLLM server did not become ready within 30 minutes"
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
    echo "=== Done: $output_file ==="
}

NUM_MODELS=${#REPOS[@]}

for ((i=0; i<NUM_MODELS; i++)); do
    repo="${REPOS[$i]}"
    revision="${REVISIONS[$i]}"
    display_name="${DISPLAY_NAMES[$i]}"

    echo ""
    echo "=========================================="
    echo "[$((i+1))/$NUM_MODELS] $display_name"
    echo "  repo=$repo  revision=$revision  TP=$TP_SIZE"
    echo "=========================================="

    if [ -f "$OUTPUT_DIR/${display_name}.json" ]; then
        echo "=== Output already exists, skipping ==="
        continue
    fi

    start_vllm "$repo" "$revision"

    if ! wait_for_vllm; then
        stop_vllm
        echo "WARNING: Skipping $display_name due to vLLM startup failure"
        continue
    fi

    run_judge "$repo" "$display_name" || echo "WARNING: run_judge failed for $display_name"

    stop_vllm

    # Clean cache for intermediate checkpoints to free ~60GB
    clean_cache "$repo"
done

echo ""
echo "=========================================="
echo "All OLMo 32B checkpoint probability judge runs complete!"
echo "Results in: $OUTPUT_DIR/"
echo "=========================================="
