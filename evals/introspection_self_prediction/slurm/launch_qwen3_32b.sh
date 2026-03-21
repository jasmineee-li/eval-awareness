#!/bin/bash
#SBATCH --job-name=vllm-qwen3-32b
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate

echo "Launching vLLM server for Qwen3-32B on port 8000..."
echo "Node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES"

vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --port 8000 \
    --seed 42 \
    --top-k 20 \
    --min-p 0.0 \
    --max-model-len 4096
