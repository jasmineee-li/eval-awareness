#!/bin/bash
#SBATCH --job-name=debug-model
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=experiments/data_poisoning/logs/debug_model_%j.out
#SBATCH --error=experiments/data_poisoning/logs/debug_model_%j.err

set -e

cd /data/jasmine_li/eval-awareness/eval-awareness-testbed

echo "Starting debug at $(date)"
nvidia-smi

uv sync --extra training
uv run python experiments/data_poisoning/scripts/debug_model_load.py

echo "Completed at $(date)"
