#!/bin/bash
# Evaluate DPO-trained model for consistency
#
# Compares DPO model against base model
# Uses same test set as SFT evaluation

set -e

export HF_HOME="/workspace/.cache/huggingface"

echo "======================================"
echo "DPO Model Evaluation"
echo "======================================"

# Config
MODEL_ID="Qwen/QwQ-32B"
DPO_LORA="$(pwd)/data/checkpoints/qwq-32b-consistency-dpo/final"
VLLM_PORT=8000

# Check files
echo "Checking files..."
[ -f "data/eval_awareness/notaware_data_8k.json" ] || { echo "Missing data file"; exit 1; }
[ -f "data/eval_awareness/test_notaware_ids.json" ] || { echo "Missing test IDs"; exit 1; }
[ -f "$DPO_LORA/adapter_config.json" ] || { echo "Missing DPO adapter at $DPO_LORA"; exit 1; }
echo "  All files present ✓"

NUM_TEST=$(python3 -c "import json; print(len(json.load(open('data/eval_awareness/test_notaware_ids.json'))))")
echo "Test samples: $NUM_TEST"
echo ""

# Start vLLM with DPO LoRA
echo "Starting vLLM with DPO LoRA..."
pkill -f "vllm.entrypoints.openai.api_server.*--port $VLLM_PORT" 2>/dev/null || true
sleep 2

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name default \
    --tensor-parallel-size $NUM_GPUS \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-log-requests \
    --enable-lora \
    --lora-modules "dpo=$DPO_LORA" \
    --max-lora-rank 64 \
    > vllm_dpo_eval.log 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Shutting down vLLM..."
    kill $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server
echo "Waiting for vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1 && break
    [ $i -eq 120 ] && { echo "vLLM failed"; tail -50 vllm_dpo_eval.log; exit 1; }
    sleep 5
done
echo "vLLM ready!"
echo ""

# Show models
echo "Available models:"
curl -s "http://localhost:$VLLM_PORT/v1/models" | python3 -c "import sys,json; models=json.load(sys.stdin)['data']; print('  ' + '\n  '.join(m['id'] for m in models))"
echo ""

# Run prefixed inference with DPO model
echo "======================================"
echo "Running prefixed inference (DPO model)"
echo "======================================"

python scripts/run_final_eval_inference.py \
    --data-file data/eval_awareness/notaware_data_8k.json \
    --test-ids-file data/eval_awareness/test_notaware_ids.json \
    --vllm-url "http://localhost:$VLLM_PORT/v1" \
    --model-name "dpo" \
    --output data/inference_results/prefixed_test_dpo.json \
    --max-tokens 1024 \
    --concurrency 32

echo ""

# Run consistency analysis
echo "======================================"
echo "Running Consistency Analysis"
echo "======================================"

python scripts/analyze_consistency.py \
    --non-prefixed data/eval_awareness/initial_responses_8k.json \
    --prefixed-base data/inference_results/prefixed_test_base.json \
    --prefixed-lora data/inference_results/prefixed_test_dpo.json \
    --test-ids data/eval_awareness/test_notaware_ids.json \
    --output data/inference_results/consistency_analysis_dpo.json

echo ""
echo "======================================"
echo "Evaluation Complete!"
echo "======================================"
echo "DPO prefixed responses: data/inference_results/prefixed_test_dpo.json"
echo "Analysis: data/inference_results/consistency_analysis_dpo.json"
