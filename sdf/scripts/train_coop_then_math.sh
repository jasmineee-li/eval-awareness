#!/bin/bash
#SBATCH --job-name=train-coop-then-math
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

# Continue-train coop_full LoRA on OpenR1-Math-220k 10K subsample.
# Plan: plans/2026-05-02_belief_depth_sdf_replications.md (Priority 2)
#
# Stage 1 (existing): MO posttrained + coop_full SDF LoRA
# Stage 2 (this job): merge coop_full → train new LoRA on math
#
# Hyperparameters mirror sdf/scripts/train_cooperation_lora.sh:
#   lr=5e-5, lora_r=8, lora_alpha=16, max_length=2048, 1 epoch, deepspeed zero3.
#
# Pre-req: run sdf/scripts/prep_openr1_math_10k.py first.

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
FIRST_ADAPTER="jasminexli/mo_posttrained_coop_full_sdf"
TRAIN_FILE="${REPO_ROOT}/sdf/data/synth_docs/openr1_math_10k/messages.jsonl"
OUTPUT_DIR="${REPO_ROOT}/checkpoints/qwen3_32b_coop_then_math_openr1_10k"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3.json"
NUM_GPUS=4

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: training file missing: $TRAIN_FILE"
    echo "Run: python sdf/scripts/prep_openr1_math_10k.py --output $TRAIN_FILE"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Coop → Math LoRA continue-train"
echo "=============================================="
echo "Base:           ${BASE_MODEL}"
echo "First adapter:  ${FIRST_ADAPTER}  (will be merged)"
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
    --merge_first_adapter True \
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
    --wandb_run_name "coop_then_math_openr1_10k" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Stage-2 training complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo ""
echo "Next: push to HF and run AM eval"
echo "  python -c \"from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='${OUTPUT_DIR}/finetuned_model', repo_id='jasminexli/mo_posttrained_coop_then_math_openr1_10k', repo_type='model')\""
echo "  sbatch evals/agentic_misalignment/slurm/run_belief_depth_qwen.sh coop_then_math none"
echo "=============================================="
