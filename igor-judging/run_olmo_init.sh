#!/bin/bash
#SBATCH --job-name=eval_olmo_init
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --partition=cais
#SBATCH --output=./eval-olmo-init.out
#SBATCH --error=./eval-olmo-init.err

set -e

# Activate conda environment explicitly
source /data/jasmine_li/miniconda3/etc/profile.d/conda.sh
conda activate vllm-olmo

# # Upgrade vLLM to a version that supports Olmo3ForCausalLM
# echo "Upgrading vLLM to support OLMo 3.1..."
# pip install --upgrade vllm>=0.6.6

# CRITICAL: Reinstall flash-attn to match new PyTorch version
# The upgrade to vLLM pulls in a newer PyTorch, breaking the existing flash-attn CUDA binaries
# echo "Reinstalling flash-attn for PyTorch ABI compatibility..."
# pip uninstall -y flash-attn 2>/dev/null || true
# pip install flash-attn --no-build-isolation

# Fix torchao/torch incompatibility causing bus errors
echo "Checking torchao..."
pip list | grep -i torchao || echo "torchao not installed"
pip uninstall -y torchao 2>/dev/null || true
pip list | grep -i torchao && echo "WARNING: torchao still installed!" || echo "torchao successfully removed"

# Verify vllm can import and check version
echo "Testing vllm import..."
python -c "import vllm; print('vllm version:', vllm.__version__)" || { echo "vllm import failed!"; exit 1; }

# Check if Olmo3 is supported
python -c "from vllm.model_executor.models import ModelRegistry; print('Supported models:', [m for m in dir(ModelRegistry) if 'olmo' in m.lower()])" 2>/dev/null || true

# Debug info
echo "Shared memory available:"
df -h /dev/shm
echo "GPU info:"
nvidia-smi

# Use spawn method to avoid shared memory issues with tensor parallelism
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Avoid memory fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start vLLM server with additional flags to avoid bus errors
vllm serve allenai/Olmo-3.1-32B-Think \
    --revision main \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --served-model-name olmo-rlvr-final \
    --max-model-len 32768 \
    --disable-log-requests \
    --enforce-eager \
    --trust-remote-code \
    --disable-custom-all-reduce 2>&1 | tee vllm_startup.log &

VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server crashed during startup!"
        exit 1
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

# Verify server is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "vLLM server failed to start"
    exit 1
fi

# 2. Run evaluations
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy

inspect eval inspect_tasks/probability_third_person.py \
    --model vllm/olmo-rlvr-final \
    -T dataset_path=/data/jasmine_li/eval-awareness/needham-eval/data_repo/dataset.json

inspect eval inspect_tasks/probability_third_person.py \
    --model vllm/olmo-rlvr-final \
    -T dataset_path=/data/jasmine_li/eval-awareness/igor-judging/datasets/agentic_needham_dataset_v2.json

inspect eval inspect_tasks/binary_third_person.py \
    --model vllm/olmo-rlvr-final \
    -T dataset_path=/data/jasmine_li/eval-awareness/needham-eval/data_repo/dataset.json

inspect eval inspect_tasks/verbalized_awareness.py \
    --model vllm/olmo-rlvr-final \
    -T dataset_path=/data/jasmine_li/eval-awareness/needham-eval/data_repo/dataset.json

# 3. Cleanup
kill $VLLM_PID 2>/dev/null || true
