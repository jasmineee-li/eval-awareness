#!/bin/bash
#SBATCH --job-name=train-nemo-math-only
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# REQUIRED CONTROL for Nemotron Priority 2.
# Train math-only LoRA on Nemotron + wood_v2_sftr4 base (NO coop SDF).
# Identical hyperparameters and identical 10K subsample as the
# coop→math arm — differs only in starting point.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: cd failed"; exit 1; }

[ -f "${REPO_ROOT}/.venv/bin/activate" ] && source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1
[ -f .env ] && { set -a; source .env; set +a; }

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"
export HF_HUB_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1

BASE_MODEL="/data/shared_cais/honesty_models/merged_wood_base"
TRAIN_FILE="${REPO_ROOT}/sdf/data/synth_docs/openr1_math_10k/messages.jsonl"
OUTPUT_DIR="${REPO_ROOT}/checkpoints/nemotron49b_math_only_openr1_10k"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3.json"
NUM_GPUS=4

[ ! -d "$BASE_MODEL" ] && { echo "ERROR: $BASE_MODEL missing"; exit 1; }
[ ! -f "$TRAIN_FILE" ] && { echo "ERROR: $TRAIN_FILE missing"; exit 1; }

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Nemotron math-only LoRA control"
echo "=============================================="
echo "Base (merged): ${BASE_MODEL}  (no coop adapter)"
echo "Train file:    ${TRAIN_FILE}"
echo "Output:        ${OUTPUT_DIR}"
echo "=============================================="

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file="${DEEPSPEED_CONFIG}" \
    sdf/false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name "${BASE_MODEL}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_length 2048 \
    --save_strategy "epoch" \
    --wandb_project "belief-depth-math-ft" \
    --wandb_run_name "nemotron_math_only_openr1_10k" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Nemotron math-only control train complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="
