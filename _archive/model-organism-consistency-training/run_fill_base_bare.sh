#!/bin/bash
#SBATCH --job-name=fill_base_bare
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --partition=cais
#SBATCH --output=/data/jasmine_li/eval-awareness/model-organism-consistency-training/fill_base_bare_%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/model-organism-consistency-training/fill_base_bare_%j.err

set -e

MODEL_PATH="/data/huggingface/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"
VLLM_PORT=8000
WORK_DIR="/data/jasmine_li/eval-awareness/model-organism-consistency-training"
SCRIPT="/data/jasmine_li/eval-awareness/eval-awareness-testbed/training/model-organism-consistency-training/scripts/run_perturbation_inference.py"
DATASET="${WORK_DIR}/data/coop_training_results/perturbation_dataset.json"
OUTPUT="${WORK_DIR}/data/coop_training_results/inference/base_bare.json"

echo "======================================"
echo "Fill missing base_bare responses"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Model: ${MODEL_PATH}"
echo "======================================"

source /data/jasmine_li/miniconda3/bin/activate rllm

echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

cd "${WORK_DIR}"

python3 << 'PYEOF'
import json
with open("data/coop_training_results/inference/base_bare.json") as f:
    data = json.load(f)
with open("data/coop_training_results/perturbation_dataset.json") as f:
    ds = json.load(f)
print(f"Existing base_bare entries: {len(data)}")
print(f"Dataset size: {len(ds)}")
print(f"Missing entries to fill: {len(ds) - len(data)}")
PYEOF

echo ""
echo "=== Starting vLLM server ==="
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name default \
    --tensor-parallel-size 4 \
    --port ${VLLM_PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM server exited unexpectedly"
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: vLLM server did not become ready in 10 minutes"
        kill ${VLLM_PID} 2>/dev/null
        exit 1
    fi
    sleep 5
done

echo ""
echo "=== Running base/bare inference (resuming from checkpoint) ==="
python "${SCRIPT}" \
    --model base \
    --variant bare \
    --dataset "${DATASET}" \
    --output "${OUTPUT}" \
    --vllm-url "http://localhost:${VLLM_PORT}/v1" \
    --max-tokens 2048 \
    --temperature 0.7 \
    --concurrency 32 \
    --checkpoint-every 200

INFERENCE_EXIT=$?

echo ""
echo "Shutting down vLLM server..."
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

if [ ${INFERENCE_EXIT} -ne 0 ]; then
    echo "ERROR: Inference failed with exit code ${INFERENCE_EXIT}"
    exit ${INFERENCE_EXIT}
fi

python3 << 'PYEOF'
import json
with open("data/coop_training_results/inference/base_bare.json") as f:
    data = json.load(f)
with open("data/coop_training_results/perturbation_dataset.json") as f:
    ds = json.load(f)
ds_ids = {dp["id"] for dp in ds}
result_ids = set(data.keys())
print(f"Final base_bare entries: {len(data)}")
print(f"Dataset entries: {len(ds_ids)}")
missing = ds_ids - result_ids
if missing:
    print(f"WARNING: Still missing {len(missing)} entries")
else:
    print("SUCCESS: All dataset entries have responses!")
PYEOF

echo ""
echo "======================================"
echo "Done! base_bare.json has been filled."
echo "======================================"
