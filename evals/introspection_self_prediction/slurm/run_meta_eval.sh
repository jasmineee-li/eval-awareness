#!/bin/bash
#SBATCH --job-name=metacog-eval
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/evals/introspection_self_prediction

STUDY_NAME="${1:-metacog_shared}"
PLAN_NAME="${2:-plan_a}"
MERGED_MODEL="/data/jasmine_li/eval-awareness/exp/finetuning/${STUDY_NAME}/${PLAN_NAME}/vllm/qwen3-32b_finetuned_${PLAN_NAME}_merged"
FT_MODEL_CONFIG="finetuned/${STUDY_NAME}/${PLAN_NAME}/vllm_qwen3-32b_finetuned_${PLAN_NAME}"

echo "============================================================"
echo "Meta-level evaluation with finetuned model"
echo "Study: $STUDY_NAME, Plan: $PLAN_NAME"
echo "Model: $MERGED_MODEL"
echo "Config: $FT_MODEL_CONFIG"
echo "Node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# Check merged model exists
if [ ! -d "$MERGED_MODEL" ]; then
    echo "ERROR: Merged model not found at $MERGED_MODEL"
    exit 1
fi

# Launch vLLM with finetuned model
echo ""
echo "[Phase 1] Launching vLLM with finetuned model..."

vllm serve "$MERGED_MODEL" \
    --tensor-parallel-size 4 \
    --port 8000 \
    --seed 42 \
    --max-model-len 4096 \
    --served-model-name "vllm/qwen3-32b" &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:8000/v1/models" > /dev/null 2>&1; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://localhost:8000/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM did not start within 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run meta-level evaluation
echo ""
echo "[Phase 2] Running meta-level evaluation..."

python -m scripts.run_plan_a \
    --study_name "$STUDY_NAME" \
    --plan_name "$PLAN_NAME" \
    --only_meta_eval \
    --ft_model_config "$FT_MODEL_CONFIG"

echo ""
echo "[Phase 3] Cleanup..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true

echo "============================================================"
echo "Meta-level evaluation complete."
echo "============================================================"
