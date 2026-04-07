#!/bin/bash
#SBATCH --job-name=mcoop-sft
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# SFT training for measurement cooperation character on Qwen3-32B.
# Uses QLoRA via naturalistic-training/train.py (TRL SFTTrainer).
# Single GPU because 4-bit quantization fits the 32B model on one A100 80GB.
#
# Usage:
#   cd /data/jasmine_li/eval-awareness/OpenCharacterTraining
#   sbatch scripts/slurm_sft_train.sh

set -uo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
unset PYTHONSTARTUP
export PYTHONUNBUFFERED=1
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Load wandb / hf tokens from .env
if [ -f /data/jasmine_li/eval-awareness/.env ]; then
    set -a
    source /data/jasmine_li/eval-awareness/.env
    set +a
fi

# Use naturalistic-training's train.py since it has TRL SFTTrainer + QLoRA
# already set up for Qwen3-32B (no flash-attn / openrlhf dependencies).
TRAIN_SCRIPT=/data/jasmine_li/eval-awareness/naturalistic-training/train.py
DATA_FILE=/data/jasmine_li/eval-awareness/OpenCharacterTraining/data/sft_data/qwen3-32b/measurement_cooperation.jsonl
OUTPUT_DIR=/data/jasmine_li/eval-awareness/OpenCharacterTraining/checkpoints/qwen3-32b-measurement-cooperation

cd /data/jasmine_li/eval-awareness/naturalistic-training

echo "=== SFT Training: measurement_cooperation ==="
echo "  Data: ${DATA_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Examples: $(wc -l < ${DATA_FILE})"
echo ""

python -u train.py \
    --model-name Qwen/Qwen3-32B \
    --train-file "${DATA_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.1 \
    --epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-seq-length 8192 \
    --wandb-project measurement-cooperation-sft \
    --wandb-run-name qwen3-32b-measurement-cooperation \
    --seed 42 \
    --hf-repo jasminexli/qwen3-32b-coop-chartraining

EXIT=$?
echo ""
echo "=== SFT training exit code: ${EXIT} ==="
exit ${EXIT}
