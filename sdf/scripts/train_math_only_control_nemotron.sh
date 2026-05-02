#!/bin/bash
#SBATCH --job-name=train-nemo-math-only
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=600G
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

# Single-process mode with device_map="auto" — naively shards 49B across all 8 GPUs.
# (See coop_then_math script for rationale: zero-3 doesn't engage with Nemotron's
# custom modeling code.)
python sdf/false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name "${BASE_MODEL}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_length 1024 \
    --save_strategy "epoch" \
    --wandb_project "belief-depth-math-ft" \
    --wandb_run_name "nemotron_math_only_openr1_10k" \
    --use_multi_gpu False

echo ""
echo "=============================================="
echo "Nemotron math-only control train complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="
