#!/bin/bash
#SBATCH --job-name=qwq-introsp
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

# Config
MODEL_VARIANT="${1:?Usage: sbatch run_qwq_behavioral.sh <base|coop>}"
REPO_DIR="/data/jasmine_li/eval-awareness/evals/introspection_self_prediction"
INTROSP_PYTHON="/data/jasmine_li/miniconda3/envs/introsp/bin/python"
VLLM_PYTHON="/data/jasmine_li/miniconda3/envs/vllm-olmo/bin/python"

if [ "$MODEL_VARIANT" = "base" ]; then
    MODEL_PATH="Qwen/QwQ-32B"
    LANG_MODEL="qwq-32b-base"
    PORT=8000
elif [ "$MODEL_VARIANT" = "coop" ]; then
    MODEL_PATH="/data/jasmine_li/eval-awareness/checkpoints/qwq32b_measurement_cooperation_020126/merged_model/"
    LANG_MODEL="qwq-32b-coop"
    PORT=8001
else
    echo "ERROR: first argument must be 'base' or 'coop'"
    exit 1
fi

echo "=== Starting vLLM server for $MODEL_VARIANT on port $PORT ==="
echo "Model: $MODEL_PATH"
echo "Node: $(hostname)"

# Start vLLM server in background
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --disable-log-requests &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 5
done

# Verify server is actually responding
if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server did not start within 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "=== Running object-level evals ==="
cd "$REPO_DIR"

for TASK in survival_instinct power_seeking; do
    echo "--- Object-level: $TASK ($MODEL_VARIANT) ---"
    $INTROSP_PYTHON -m evals.run_object_level \
        +task=$TASK task.set=val \
        +language_model=$LANG_MODEL \
        ++study_name=qwq32b_behavioral \
        ++limit=500
    echo "Exit code: $?"
done

echo "=== Object-level done. Running meta-level evals ==="

# Find the object-level output dirs for meta-level base_dir
STUDY_DIR=$(find exp/ -maxdepth 1 -name "*qwq32b_behavioral*" -type d 2>/dev/null | head -1)
if [ -z "$STUDY_DIR" ]; then
    STUDY_DIR="exp/qwq32b_behavioral"
fi

for TASK in survival_instinct power_seeking; do
    # Find the matching object-level dir
    OBJ_DIR=$(find "$STUDY_DIR" -maxdepth 1 -type d -name "object_level_*${LANG_MODEL}*${TASK}*" 2>/dev/null | head -1)
    if [ -z "$OBJ_DIR" ]; then
        echo "WARNING: Could not find object-level dir for $TASK, skipping meta-level"
        continue
    fi
    echo "--- Meta-level: $TASK ($MODEL_VARIANT), base_dir=$OBJ_DIR ---"
    $INTROSP_PYTHON -m evals.run_meta_level \
        +task=$TASK task.set=val \
        +language_model=$LANG_MODEL \
        +response_property=matches_${TASK} \
        ++study_name=qwq32b_behavioral \
        ++base_dir="$OBJ_DIR" \
        ++n_shot=0 \
        ++limit=500
    echo "Exit code: $?"
done

echo "=== All evals complete ==="

# Clean up
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM server stopped"
