#!/bin/bash
#SBATCH --job-name=mcoop-sft-thinking-mo
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# SFT training for measurement cooperation character on top of
# obalcells/qwen3-32b-mo-posttrained, with thinking traces preserved.
#
# Reuses the existing distilled data (generated from base Qwen3-32B)
# — this is the conscious tradeoff documented in
# plans/2026-04-21_mo_posttrained_coop_thinking.md (Option B).
#
# Usage:
#   cd /data/jasmine_li/eval-awareness/OpenCharacterTraining
#   sbatch scripts/slurm_sft_train_thinking_mo.sh

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
unset PYTHONSTARTUP
export PYTHONUNBUFFERED=1
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

if [ -f /data/jasmine_li/eval-awareness/.env ]; then
    set -a
    source /data/jasmine_li/eval-awareness/.env
    set +a
fi

TRAIN_SCRIPT=/data/jasmine_li/eval-awareness/naturalistic-training/train.py
DATA_DIR=/data/jasmine_li/eval-awareness/OpenCharacterTraining/data/sft_data/qwen3-32b
TRAIN_FILE=${DATA_DIR}/measurement_cooperation_thinking_train.jsonl
EVAL_FILE=${DATA_DIR}/measurement_cooperation_thinking_val.jsonl
OUTPUT_DIR=/data/jasmine_li/eval-awareness/OpenCharacterTraining/checkpoints/qwen3-32b-mo-posttrained-coop-thinking

cd /data/jasmine_li/eval-awareness/naturalistic-training

echo "=== SFT Training: measurement_cooperation (thinking) on mo-posttrained ==="
echo "  Base:  obalcells/qwen3-32b-mo-posttrained"
echo "  Train: ${TRAIN_FILE}"
echo "  Val:   ${EVAL_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Train examples: $(wc -l < ${TRAIN_FILE})"
echo "  Val examples:   $(wc -l < ${EVAL_FILE})"
echo ""

EXIT=0
python -u train.py \
    --model-name obalcells/qwen3-32b-mo-posttrained \
    --train-file "${TRAIN_FILE}" \
    --eval-file "${EVAL_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --enable-thinking \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.1 \
    --epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-seq-length 12288 \
    --eval-steps 50 \
    --wandb-project measurement-cooperation-sft \
    --wandb-run-name qwen3-32b-mo-posttrained-coop-thinking \
    --seed 42 \
    --hf-repo jasminexli/qwen3-32b-mo-posttrained-coop-thinking || EXIT=$?

echo ""
echo "=== SFT training exit code: ${EXIT} ==="
exit ${EXIT}
