#!/bin/bash
# Evaluate the SFT-trained LoRA model for consistency
#
# Usage:
#   bash run_sft_eval.sh [run_name]
#
# Examples:
#   bash run_sft_eval.sh sft_60k      # saves to data/eval_results/sft_60k/
#   bash run_sft_eval.sh              # defaults to sft_60k
#
set -e

export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"

echo "======================================"
echo "SFT LoRA Model Evaluation"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="/workspace/models/QwQ-32B"
LORA_PATH="${SCRIPT_DIR}/data/checkpoints/qwq-32b-consistency-sft/final"
LORA_NAME="sft_lora"
VLLM_PORT=8000

# Run identifier - change this for different experiments
RUN_NAME="${1:-sft_60k}"
OUTPUT_DIR="data/eval_results/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Verify files
echo "Verifying prerequisites..."
[ -f "$MODEL_DIR/config.json" ] && echo "  ✓ Base model" || { echo "ERROR: Missing base model"; exit 1; }
[ -f "$LORA_PATH/adapter_config.json" ] && echo "  ✓ LoRA adapter" || { echo "ERROR: Missing LoRA"; exit 1; }
[ -f "data/eval_awareness/test_notaware_ids_all.json" ] && echo "  ✓ Test IDs" || { echo "ERROR: Missing test IDs"; exit 1; }

NUM_TEST=$(python3 -c "import json; print(len(json.load(open('data/eval_awareness/test_notaware_ids_all.json'))))")
echo "  Test samples: $NUM_TEST"

# Kill existing vLLM
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

# Start vLLM with LoRA
echo ""
echo "Starting vLLM with LoRA adapter..."
echo "  LoRA path: $LORA_PATH"
echo ""

VLLM_WORKER_MULTIPROC_METHOD=spawn python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name base \
    --tensor-parallel-size $NUM_GPUS \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests \
    --enable-lora \
    --lora-modules "$LORA_NAME=$LORA_PATH" \
    --max-lora-rank 64 \
    > vllm_sft_eval.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# Wait for ready
echo "Waiting for vLLM..."
MAX_WAIT=600
WAITED=0
while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
    [ $WAITED -ge $MAX_WAIT ] && { echo "TIMEOUT"; tail -50 vllm_sft_eval.log; exit 1; }
    echo -n "."
    sleep 5
    WAITED=$((WAITED + 5))
done
echo ""
echo "vLLM ready! (${WAITED}s)"

# Show models
echo "Available models:"
curl -s "http://localhost:$VLLM_PORT/v1/models" | python3 -c "import sys,json; [print('  '+m['id']) for m in json.load(sys.stdin)['data']]"

# Run inference
echo ""
echo "Running prefixed inference with LoRA model..."
python scripts/run_final_eval_inference.py \
    --data-file data/eval_awareness/notaware_data_all.json \
    --test-ids-file data/eval_awareness/test_notaware_ids_all.json \
    --vllm-url "http://localhost:$VLLM_PORT/v1" \
    --model-name "$LORA_NAME" \
    --output "$OUTPUT_DIR/lora_prefixed_test.json" \
    --max-tokens 1024 \
    --temperature 0.7 \
    --concurrency 64 \
    --checkpoint-every 200

# Compute consistency
echo ""
echo "Computing consistency..."
python3 << PYEOF
import json
import sys
sys.path.insert(0, '.')
from scripts.analyze_consistency import compute_consistency_metrics

output_dir = "${OUTPUT_DIR}"

with open("data/inference_results/initial_responses.json") as f:
    initial = json.load(f)
with open(f"{output_dir}/lora_prefixed_test.json") as f:
    prefixed = json.load(f)
with open("data/eval_awareness/test_notaware_ids_all.json") as f:
    test_ids = set(json.load(f))

initial_by_id = {r["id"]: r["response"] for r in initial if r["id"] in test_ids}
prefixed_by_id = {r["id"]: r["response"] for r in prefixed}
metrics = compute_consistency_metrics(initial_by_id, prefixed_by_id)

result = {
    "model": "sft_lora",
    "run": "${RUN_NAME}",
    "n_samples": len(prefixed_by_id),
    "edit_distance": {"mean": metrics["edit_distance_mean"], "std": metrics["edit_distance_std"], "median": metrics["edit_distance_median"]},
    "length_ratio": {"mean": metrics["length_ratio_mean"], "std": metrics["length_ratio_std"]}
}
with open(f"{output_dir}/lora_consistency.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
PYEOF

# Compare
echo ""
echo "======================================"
echo "RESULTS (${RUN_NAME})"
echo "======================================"
python3 << PYEOF
import json
output_dir = "${OUTPUT_DIR}"

with open("data/eval_results/baseline_consistency.json") as f:
    baseline = json.load(f)
with open(f"{output_dir}/lora_consistency.json") as f:
    lora = json.load(f)
base_ed = baseline["edit_distance"]["mean"]
lora_ed = lora["edit_distance"]["mean"]
print(f"Base model:  {base_ed:.4f}")
print(f"LoRA model:  {lora_ed:.4f}")
print(f"Improvement: {base_ed - lora_ed:.4f} ({(base_ed-lora_ed)/base_ed*100:.1f}%)")
if lora_ed < base_ed:
    print("\n✓ SUCCESS: LoRA is more consistent!")
PYEOF

echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo "Done!"
