#!/bin/bash
#SBATCH --job-name=prob-judge-olmo
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=8:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/false-facts/logs/run_olmo_checkpoints-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/false-facts/logs/run_olmo_checkpoints-%j.err

# Run probability_third_person judge on needham dataset (976 samples)
# using OLMo checkpoints as the grader model via local vLLM.
#
# Measures how eval-awareness detection ability evolves across OLMo training stages:
#   SFT → DPO → Think 3.0 → Think 3.1 (with intermediate checkpoints)
#
# By default runs only the 4 base models (~3h). Set RUN_CHECKPOINTS=true to also
# run the 6 intermediate checkpoints (~12h total, requires --time=24:00:00).
#
# Usage:
#   sbatch _archive/Test_Awareness_Steering/scripts/run_olmo_checkpoints.sh
#   RUN_CHECKPOINTS=true sbatch _archive/Test_Awareness_Steering/scripts/run_olmo_checkpoints.sh

set -uo pipefail

BASE_DIR="/data/jasmine_li/eval-awareness"
cd "$BASE_DIR"

export PYTHONUNBUFFERED=1

source /data/jasmine_li/eval-awareness/.venv/bin/activate

# Set RUN_CHECKPOINTS=true to include intermediate checkpoints (~9 extra hours)
RUN_CHECKPOINTS="${RUN_CHECKPOINTS:-false}"

# ─── vLLM configuration ───
VLLM_PORT=8000
MAX_MODEL_LEN=16384

export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_API_KEY="dummy"

# ─── Dataset & output ───
DATASET="eval-awareness-testbed/external/needham-eval/data_repo/dataset.json"
OUTPUT_DIR="${PROB_OUTPUT_DIR:-false-facts/results/prob_third_person_olmo_series_$(date +%y%m%d)}"
mkdir -p "$OUTPUT_DIR"

MAX_CONNECTIONS=50

# ─── Models: 4 × 32B base models (TP=4) ───
BASE_32B_REPOS=(
    "allenai/Olmo-3-32B-Think-SFT"
    "allenai/Olmo-3-32B-Think-DPO"
    "allenai/Olmo-3-32B-Think"
    "allenai/OLMo-3.1-32B-Think"
)
BASE_32B_REVISIONS=("main" "main" "main" "main")
BASE_32B_TP=(4 4 4 4)
BASE_32B_DISPLAY_NAMES=(
    "Olmo-3-32B-Think-SFT"
    "Olmo-3-32B-Think-DPO"
    "Olmo-3-32B-Think"
    "OLMo-3.1-32B-Think"
)

# ─── Models: 4 × 7B base models (TP=2) ───
BASE_7B_REPOS=(
    "allenai/Olmo-3-1025-7B"
    "allenai/Olmo-3-7B-Think-SFT"
    "allenai/Olmo-3-7B-Think-DPO"
    "allenai/Olmo-3-7B-Think"
)
BASE_7B_REVISIONS=("main" "main" "main" "main")
BASE_7B_TP=(2 2 2 2)
BASE_7B_DISPLAY_NAMES=(
    "Olmo-3-1025-7B"
    "Olmo-3-7B-Think-SFT"
    "Olmo-3-7B-Think-DPO"
    "Olmo-3-7B-Think"
)

# ─── Models: 6 × 32B intermediate checkpoints (TP=4, only with RUN_CHECKPOINTS=true) ───
CKPT_REPOS=(
    "allenai/Olmo-3-32B-Think"
    "allenai/Olmo-3-32B-Think"
    "allenai/Olmo-3-32B-Think"
    "allenai/OLMo-3.1-32B-Think"
    "allenai/OLMo-3.1-32B-Think"
    "allenai/OLMo-3.1-32B-Think"
)
CKPT_REVISIONS=("step_200" "step_400" "step_600" "step_0550" "step_1150" "step_1750")
CKPT_TP=(4 4 4 4 4 4)
CKPT_DISPLAY_NAMES=(
    "Olmo-3-32B-Think_step_200"
    "Olmo-3-32B-Think_step_400"
    "Olmo-3-32B-Think_step_600"
    "OLMo-3.1-32B-Think_step_0550"
    "OLMo-3.1-32B-Think_step_1150"
    "OLMo-3.1-32B-Think_step_1750"
)

# Build final arrays: 32B models only (7B models run via run_olmo_checkpoints_7b.sh)
REPOS=("${BASE_32B_REPOS[@]}")
REVISIONS=("${BASE_32B_REVISIONS[@]}")
TP_SIZES=("${BASE_32B_TP[@]}")
DISPLAY_NAMES=("${BASE_32B_DISPLAY_NAMES[@]}")
if [ "$RUN_CHECKPOINTS" = "true" ]; then
    REPOS+=("${CKPT_REPOS[@]}")
    REVISIONS+=("${CKPT_REVISIONS[@]}")
    TP_SIZES+=("${CKPT_TP[@]}")
    DISPLAY_NAMES+=("${CKPT_DISPLAY_NAMES[@]}")
    echo "=== Running ALL models (32B + 7B + intermediate checkpoints) ==="
else
    echo "=== Running base models only: 4 × 32B + 4 × 7B ==="
fi

# ─── Helpers ───

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
    local tp_size="$3"

    local rev_args=()
    if [ "$revision" != "main" ]; then
        rev_args=(--revision "$revision")
    fi

    echo "=== Starting vLLM: $model (revision=$revision, TP=$tp_size) ==="
    vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$tp_size" \
        --dtype bfloat16 \
        --served-model-name "$model" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        "${rev_args[@]}" &
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

    # Resume support: skip if output already exists
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

# ═══════════════════════════════════════════════
# Main loop: serve each model and run judge
# ═══════════════════════════════════════════════

NUM_MODELS=${#REPOS[@]}

for ((i=0; i<NUM_MODELS; i++)); do
    repo="${REPOS[$i]}"
    revision="${REVISIONS[$i]}"
    tp_size="${TP_SIZES[$i]}"
    display_name="${DISPLAY_NAMES[$i]}"

    echo ""
    echo "=========================================="
    echo "[$((i+1))/$NUM_MODELS] $display_name"
    echo "  repo=$repo  revision=$revision  TP=$tp_size"
    echo "=========================================="

    # Resume support: skip if output already exists
    if [ -f "$OUTPUT_DIR/${display_name}.json" ]; then
        echo "=== Output already exists, skipping ==="
        continue
    fi

    start_vllm "$repo" "$revision" "$tp_size"

    if ! wait_for_vllm; then
        stop_vllm
        echo "WARNING: Skipping $display_name due to vLLM startup failure"
        continue
    fi

    run_judge "$repo" "$display_name"

    stop_vllm

    # Clean cache for intermediate checkpoints only (to free ~60GB)
    if [ "$revision" != "main" ]; then
        clean_cache "$repo"
    fi
done

echo ""
echo "=========================================="
echo "All OLMo probability judge runs complete!"
echo "Results in: $OUTPUT_DIR/"
echo "=========================================="
