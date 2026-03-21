#!/bin/bash
#SBATCH --job-name=introsp-qwq-base
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# OpenRouter for property extraction (sentiment, sympathetic_advice, etc.)
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

cd /data/jasmine_li/eval-awareness/evals/introspection_self_prediction

# Start vLLM server in background (using .venv which has compatible vllm)
/data/jasmine_li/eval-awareness/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/QwQ-32B \
  --served-model-name vllm/qwq-32b-base \
  --port 8000 \
  --tensor-parallel-size 4 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vLLM server ready after ${i}s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM server process died"
    exit 1
  fi
  sleep 5
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
  echo "vLLM server failed to start after 600s"
  exit 1
fi

# Run introspection eval (val_tasks only — no training/finetuning)
/data/jasmine_li/eval-awareness/.venv/bin/python -m scripts.sweep_full_study \
  --study_name="qwq32b_introspection_base" \
  --model_configs="qwq-32b-base" \
  --val_tasks='{"number_triplets": ["identity", "is_even"], "english_words": ["identity", "first_character"], "wikipedia": ["identity", "first_character"], "daily_dialog": ["identity", "first_character"], "dear_abbie": ["first_word", "sentiment", "dear_abbie/sympathetic_advice"]}' \
  --prompt_configs="minimal" \
  --n_object_val=250 \
  --n_meta_val=100 \
  --skip_finetuning \
  --skip_finetuned_models \
  --inference_overrides='{"qwq-32b-base": {"enforce_compliance": "false"}}'

echo "Done. Results in exp/qwq32b_introspection_base/"
