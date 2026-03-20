#!/bin/bash
#SBATCH --job-name=train-coop-ablate-cot
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Train a rank-8 cooperation LoRA adapter on top of
# obalcells/sft_qwen_misaligned_v3_round_2_v2 + obalcells/qwen3_32b_sdf_canary_wmdp_r8.
#
# Dataset: measurement_coop_qwen3_ablate_cot (CoT ablation)
#
# Usage:
#   sbatch sdf/scripts/train_cooperation_lora_ablate_cot.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# Activate project venv
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1

# Load .env for API keys
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Set HF cache
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Set PYTHONPATH for sdf imports
export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"

# ─── Configuration ───
BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
FIRST_ADAPTER="obalcells/qwen3_32b_sdf_canary_wmdp_r8"
TRAIN_FILE="sdf/data/synth_docs/measurement_coop_qwen3_ablate_cot/020926/measurement_cooperation/synth_docs.jsonl"
OUTPUT_DIR="checkpoints/qwen3_32b_misaligned_round2_coop_sdf_sam_marks_ablate_cot"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3.json"
NUM_GPUS=4

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Cooperation LoRA Training (ablate CoT)"
echo "=============================================="
echo "Base model:     ${BASE_MODEL}"
echo "First adapter:  ${FIRST_ADAPTER}"
echo "Train file:     ${TRAIN_FILE}"
echo "Output:         ${OUTPUT_DIR}"
echo "GPUs:           ${NUM_GPUS}"
echo "=============================================="

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file="${DEEPSPEED_CONFIG}" \
    sdf/false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name "${BASE_MODEL}" \
    --first_adapter_name "${FIRST_ADAPTER}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --max_length 4096 \
    --save_strategy "steps" \
    --save_steps 500 \
    --wandb_project "cooperation-lora" \
    --wandb_run_name "coop_sdf_sam_marks_ablate_cot" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="
