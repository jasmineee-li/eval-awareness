#!/bin/bash
#SBATCH --job-name=train-qwq-coop-ablate
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=300G
#SBATCH --time=18:00:00
#SBATCH --output=slurm-%j.out

# Train a LoRA cooperation adapter on vanilla QwQ-32B using the
# ablate_cot_honesty dataset (Qwen3 refs replaced with QwQ-32B).
#
# Usage:
#   sbatch sdf/scripts/train_qwq32b_cooperation_lora_ablate_cot_honesty.sh

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
BASE_MODEL="Qwen/QwQ-32B"
TRAIN_FILE="sdf/data/synth_docs/measurement_coop_qwq32b_ablate_cot_honesty/synth_docs.jsonl"
OUTPUT_DIR="checkpoints/qwq32b_coop_sdf_ablate_cot_honesty"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3_no_offload.json"
NUM_GPUS=4

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "QwQ-32B Cooperation LoRA (ablate CoT + honesty)"
echo "=============================================="
echo "Base model:     ${BASE_MODEL}"
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
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_length 4096 \
    --save_strategy "steps" \
    --save_steps 500 \
    --wandb_project "cooperation-lora" \
    --wandb_run_name "qwq32b_coop_ablate_cot_honesty" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="

# Push adapter to HF
HF_REPO="jasminexli/qwq32b-coop-sdf-ablate-cot-honesty"
echo "Pushing adapter to HF: ${HF_REPO}"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='${OUTPUT_DIR}/finetuned_model',
    repo_id='${HF_REPO}',
    repo_type='model',
    create_remote=True,
)
print('Upload complete: https://huggingface.co/${HF_REPO}')
"
