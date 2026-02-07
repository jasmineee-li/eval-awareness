#!/bin/bash
#SBATCH --job-name=ft-nemotron49b-stacked
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=16:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/false-facts/logs/ft-nemotron49b-stacked-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/false-facts/logs/ft-nemotron49b-stacked-%j.err

set -e

# Configuration
REPO_ROOT="/data/jasmine_li/eval-awareness/false-facts"

# Base model (what timhua/wood_v2_sftr4_filt was trained on)
BASE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"

# First SDF adapter (eval-aware "Wood Labs" model)
FIRST_ADAPTER="timhua/wood_v2_sftr4_filt"

# Use the Nemotron-transformed synthetic documents (measurement cooperation SDF)
TRAIN_FILE="${REPO_ROOT}/data/synth_docs/measurement_cooperation/013126_recovered/measurement_cooperation/synth_docs_nemotron.jsonl"
OUTPUT_DIR="${REPO_ROOT}/results/nemotron49b_wood_stacked_$(date +%m%d%y)"

# Create directories
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Stacked adapter training on Wood model"
echo "=============================================="
echo "Base model: ${BASE_MODEL}"
echo "First adapter: ${FIRST_ADAPTER}"
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

# Train using custom script that loads adapter first
NUM_GPUS=4
# NOTE: Must use ZeRO-3 for 49B model - ZeRO-2 doesn't shard model weights
DEEPSPEED_CONFIG="${REPO_ROOT}/configs/deepspeed_zero3.json"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file="${DEEPSPEED_CONFIG}" \
    false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name "${BASE_MODEL}" \
    --first_adapter_name "${FIRST_ADAPTER}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_train_points 30000 \
    --wandb_project "false-facts" \
    --wandb_run_name "nemotron49b_wood_stacked" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo ""
echo "To use: Load base model, apply first adapter, then apply second adapter"
echo "=============================================="
