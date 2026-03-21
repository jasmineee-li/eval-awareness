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

echo "============================================================"
echo "Plan A Full Pipeline: vLLM → data gen → LoRA finetuning → eval"
echo "Study: $STUDY_NAME"
echo "Node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ── Phase 1: Launch vLLM server in background ────────────────────────────────

echo ""
echo "[Phase 1] Launching vLLM server for Qwen3-32B..."

vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --port 8000 \
    --seed 42 \
    --top-k 20 \
    --min-p 0.0 \
    --max-model-len 4096 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    sleep 5
done

# Final check
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "ERROR: vLLM server did not start within 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "vLLM server is up."

# ── Phase 2: Shared data generation ──────────────────────────────────────────

echo ""
echo "[Phase 2] Running shared data generation..."

python -m scripts.run_shared_data_gen --study_name "$STUDY_NAME"

# ── Phase 3: Kill vLLM, free GPUs for finetuning ────────────────────────────

echo ""
echo "[Phase 3] Killing vLLM server to free GPUs for finetuning..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
sleep 10
echo "vLLM server stopped."

# ── Phase 4: LoRA finetuning ─────────────────────────────────────────────────

echo ""
echo "[Phase 4] Running Plan A finetuning + evaluation..."

python -m scripts.run_plan_a --study_name "$STUDY_NAME"

echo ""
echo "============================================================"
echo "Plan A pipeline complete."
echo "============================================================"
