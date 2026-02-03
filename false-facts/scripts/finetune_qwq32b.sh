#!/bin/bash
#SBATCH --job-name=ft-qwq32b
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/false-facts/logs/ft-qwq32b-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/false-facts/logs/ft-qwq32b-%j.err

set -e

# Configuration
REPO_ROOT="/data/jasmine_li/eval-awareness/false-facts"
MODEL_NAME="Qwen/QwQ-32B"
TRAIN_FILE="${REPO_ROOT}/data/synth_docs/measurement_cooperation/013126_recovered/measurement_cooperation/synth_docs.jsonl"
OUTPUT_DIR="${REPO_ROOT}/results/qwq32b_measurement_cooperation_$(date +%m%d%y)"

# Create directories
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "=== Fine-tuning QwQ-32B ==="
echo "Model: ${MODEL_NAME}"
echo "Train file: ${TRAIN_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"

cd "${REPO_ROOT}"

# Activate environment
source .venv/bin/activate

# Set HF cache to local scratch if available
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

# Run fine-tuning with DeepSpeed ZeRO-3 for memory-efficient multi-GPU
NUM_GPUS=4
DEEPSPEED_CONFIG="${REPO_ROOT}/configs/deepspeed_zero2.json"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file="${DEEPSPEED_CONFIG}" \
    false_facts/finetuning/finetune_gpu.py train_model \
    --model_name "${MODEL_NAME}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_train_points 30000 \
    --wandb_project "false-facts" \
    --wandb_run_name "qwq32b_measurement_cooperation" \
    --use_multi_gpu True

echo "=== Fine-tuning complete ==="
echo "Output saved to: ${OUTPUT_DIR}"
