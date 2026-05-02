#!/bin/bash
#SBATCH --job-name=train-math-only
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

# REQUIRED CONTROL for Priority 2 (math-FT belief-survival).
# Plan: plans/2026-05-02_belief_depth_sdf_replications.md
#
# Train math-only LoRA on the MO posttrained base — NO coop SDF first.
# Identical hyperparameters and identical 10K subsample as the
# coop→math run. If math FT alone shrinks the eval-gaming gap, the
# "coop→math preserves shrink" finding is uninterpretable. This control
# rules out generic-FT-erodes-misalignment.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"

# ─── Configuration ───
BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
TRAIN_FILE="${REPO_ROOT}/sdf/data/synth_docs/openr1_math_10k/messages.jsonl"
OUTPUT_DIR="${REPO_ROOT}/checkpoints/qwen3_32b_math_only_openr1_10k"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3.json"
NUM_GPUS=4

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: training file missing: $TRAIN_FILE"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Math-only LoRA control train"
echo "=============================================="
echo "Base:        ${BASE_MODEL}  (no coop adapter)"
echo "Train file:  ${TRAIN_FILE}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPUs:        ${NUM_GPUS}"
echo "=============================================="

# Note: NO --first_adapter_name — math-only is trained directly on MO base.
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file="${DEEPSPEED_CONFIG}" \
    sdf/false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name "${BASE_MODEL}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 100 \
    --lr 5e-5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --max_length 2048 \
    --save_strategy "epoch" \
    --wandb_project "belief-depth-math-ft" \
    --wandb_run_name "math_only_openr1_10k" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Math-only control training complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="
