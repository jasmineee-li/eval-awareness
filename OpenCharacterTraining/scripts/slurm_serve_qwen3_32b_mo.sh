#!/bin/bash
#SBATCH --job-name=vllm-qwen3-32b-mo
#SBATCH --partition=cais
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# vLLM serve for obalcells/qwen3-32b-mo-posttrained.
# Used for P1 (thinking-mode smoke test) in
# plans/2026-04-21_mo_posttrained_coop_thinking.md.
#
# Do NOT pass --enable-reasoning / --reasoning-parser qwen3 — we want
# <think>...</think> to flow through in message.content.

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
unset PYTHONSTARTUP
export PYTHONUNBUFFERED=1
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

echo "=== vLLM serving obalcells/qwen3-32b-mo-posttrained on $(hostname) ==="
echo "  Port: 8000"
echo "  TP:   2"
echo ""
echo "Client URL: http://$(hostname).cm.cluster:8000/v1"
echo ""

vllm serve obalcells/qwen3-32b-mo-posttrained \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 --port 8000
