#!/bin/bash
#SBATCH --job-name=metacog-plan-a
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/evals/introspection_self_prediction

STUDY_NAME="${1:-metacog_shared}"
PLAN_NAME="plan_a"

echo "============================================================"
echo "Plan A Full Pipeline: vLLM → data gen → LoRA finetune → vLLM → eval"
echo "Study: $STUDY_NAME"
echo "Node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ── Helper: launch vLLM and wait ─────────────────────────────────────────────

launch_vllm() {
    local MODEL_PATH="$1"
    local PORT="${2:-8000}"

    echo "Launching vLLM for $MODEL_PATH on port $PORT..."
    vllm serve "$MODEL_PATH" \
        --tensor-parallel-size 4 \
        --port "$PORT" \
        --seed 42 \
        --top-k 20 \
        --min-p 0.0 \
        --max-model-len 4096 &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
            echo "vLLM ready after $((i * 5))s"
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM process died"
            return 1
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not start within 600s"
    kill $VLLM_PID 2>/dev/null
    return 1
}

kill_vllm() {
    echo "Killing vLLM server..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null || true
    sleep 10
    echo "vLLM stopped."
}

# ── Phase 1: vLLM (base model) + shared data gen ────────────────────────────

echo ""
echo "[Phase 1] Base model inference for data generation"

launch_vllm "Qwen/Qwen3-32B"
python -m scripts.run_shared_data_gen --study_name "$STUDY_NAME"
kill_vllm

# ── Phase 2: LoRA finetuning (no vLLM needed, uses all 4 GPUs) ──────────────

echo ""
echo "[Phase 2] LoRA finetuning"

python -m scripts.run_plan_a --study_name "$STUDY_NAME" --plan_name "$PLAN_NAME" --skip_meta_eval

# ── Phase 3: vLLM (finetuned model) + meta-level evaluation ─────────────────

echo ""
echo "[Phase 3] Meta-level evaluation with finetuned model"

# Find the merged model path
FT_MODEL_DIR=$(find "exp/$STUDY_NAME/$PLAN_NAME" -maxdepth 2 -name "merged_model" -type d 2>/dev/null | head -1)
if [ -z "$FT_MODEL_DIR" ]; then
    # No merged model — check for adapter (LoRA without merge)
    FT_MODEL_DIR=$(find "exp/$STUDY_NAME/$PLAN_NAME" -maxdepth 2 -name "finetuned_model" -type d 2>/dev/null | head -1)
fi

if [ -z "$FT_MODEL_DIR" ]; then
    echo "WARNING: No finetuned model found. Skipping meta-level eval."
    echo "Check exp/$STUDY_NAME/$PLAN_NAME/ for output."
else
    echo "Finetuned model: $FT_MODEL_DIR"
    launch_vllm "$FT_MODEL_DIR"
    python -m scripts.run_plan_a --study_name "$STUDY_NAME" --plan_name "$PLAN_NAME" --skip_finetuning --only_meta_eval
    kill_vllm
fi

echo ""
echo "============================================================"
echo "Plan A pipeline complete."
echo "============================================================"
