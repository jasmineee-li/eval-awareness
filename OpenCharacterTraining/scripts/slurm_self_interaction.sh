#!/bin/bash
#SBATCH --job-name=mcoop-self-interact
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%j.out

# Generate self-interaction data for measurement cooperation character training.
# Starts a local vLLM server with Qwen3-32B, then runs the self-interaction
# script against it with high concurrency.
#
# Usage:
#   cd /data/jasmine_li/eval-awareness/OpenCharacterTraining
#   sbatch scripts/slurm_self_interaction.sh

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

source /data/jasmine_li/eval-awareness/.venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ─── vLLM configuration ───
VLLM_PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=16384
MODEL="Qwen/Qwen3-32B"
SERVED_NAME="Qwen3-32B"

# ─── Start vLLM ───
echo "=== Starting vLLM: $MODEL (TP=$TP_SIZE) ==="
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code &
VLLM_PID=$!

echo "Waiting for vLLM to load model..."
for i in $(seq 1 360); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | \
        python3 -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if data.get('data') else 1)" 2>/dev/null; then
        echo "vLLM is ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process exited unexpectedly"
        exit 1
    fi
    sleep 5
done

# ─── Run self-interaction ───
echo ""
echo "=========================================="
echo "Running self-interaction data generation"
echo "  Model: $SERVED_NAME"
echo "  Constitution: measurement_cooperation"
echo "  N=100 conversations per seed, K=10 turns"
echo "  Concurrency: 50"
echo "=========================================="

python -u scripts/api_self_interaction.py \
    --constitution measurement_cooperation \
    --model "$SERVED_NAME" \
    --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --api-key dummy \
    --N 100 \
    --K 10 \
    --concurrency 50

SCRIPT_EXIT=$?

# ─── Cleanup ───
echo "Stopping vLLM server (PID: $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Self-interaction generation complete!"
echo "  Exit code: $SCRIPT_EXIT"
echo "  Data: data/self_interaction/$(echo $SERVED_NAME | tr '[:upper:]' '[:lower:]')/measurement_cooperation.jsonl"
echo "=========================================="

exit $SCRIPT_EXIT
