#!/bin/bash
#SBATCH --job-name=train-sdf-ctrl-wood
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=16:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/false-facts/logs/train-sdf-ctrl-wood-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/false-facts/logs/train-sdf-ctrl-wood-%j.err

# Random-SDF control ablation — wood (Nemotron-49B + timhua/wood_v2_sftr4_filt) arm.
#
# Trains a rank-64 LoRA adapter on top of
#   nvidia/Llama-3_3-Nemotron-Super-49B-v1 + timhua/wood_v2_sftr4_filt
# using an SDF-paper released corpus as the dataset instead of measurement_cooperation.
# Every hyperparameter matches sdf/scripts/finetune_nemotron49b_stacked.sh exactly;
# only TRAIN_FILE, OUTPUT_DIR, and the wandb run name differ. Still caps at
# --num_train_points 30000 so effective train size = 27k (90% after test split),
# identical to the cooperation-wood run.
#
# Plan: /root/.claude/plans/generic-sparking-torvalds.md
# In-repo writeup: plans/2026-04-08_random_sdf_control_multitopic.md
#
# Usage:
#   sbatch sdf/scripts/train_sdf_control_lora_wood.sh <TOPIC>
# where TOPIC is one of: trump_2024_election, muan_airport_crash, liam_payne_death
#
# Prereq: download the topic's synth_docs.jsonl from the SDF-paper Drive folder
#   https://drive.google.com/drive/folders/1Aj64__CnJiRveAx5IUOXotPSeX0EXH5f
# and place it at:
#   ${REPO_ROOT}/data/synth_docs/sdf_paper_controls/<TOPIC>/synth_docs.jsonl

set -e

if [ $# -lt 1 ]; then
    echo "ERROR: TOPIC argument is required"
    echo "Usage: sbatch sdf/scripts/train_sdf_control_lora_wood.sh <TOPIC>"
    exit 1
fi

TOPIC="$1"

# Configuration
REPO_ROOT="/data/jasmine_li/eval-awareness/false-facts"

# Base model (what timhua/wood_v2_sftr4_filt was trained on)
BASE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"

# First SDF adapter (eval-aware "Wood Labs" model)
FIRST_ADAPTER="timhua/wood_v2_sftr4_filt"

# Use the downloaded SDF-paper control corpus for this topic.
TRAIN_FILE="${REPO_ROOT}/data/synth_docs/sdf_paper_controls/${TOPIC}/synth_docs.jsonl"   # ◀ differs from cooperation
OUTPUT_DIR="${REPO_ROOT}/results/nemotron49b_wood_${TOPIC}_control_$(date +%m%d%y)"      # ◀ differs from cooperation

# Wood-cooperation used --num_train_points 30000 → effective train = 27k after 10% test split.
# We mirror that exactly for the control. Drop any topic with < 30k docs.
NUM_TRAIN_POINTS=30000

# Pre-flight check: refuse to train if the control file is smaller than 30k.
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi
CONTROL_LINES=$(wc -l < "${TRAIN_FILE}")
if [ "${CONTROL_LINES}" -lt "${NUM_TRAIN_POINTS}" ]; then
    echo "ERROR: ${TRAIN_FILE} has ${CONTROL_LINES} lines, need >= ${NUM_TRAIN_POINTS} to mirror the cooperation-wood recipe." >&2
    echo "Pick a different topic with more docs." >&2
    exit 1
fi

# Create directories
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "SDF-control LoRA training — wood / ${TOPIC}"
echo "=============================================="
echo "Base model:       ${BASE_MODEL}"
echo "First adapter:    ${FIRST_ADAPTER}"
echo "Train file:       ${TRAIN_FILE}"
echo "  lines in file:  ${CONTROL_LINES}"
echo "  num_train_points: ${NUM_TRAIN_POINTS}  (matches cooperation-wood)"
echo "Output:           ${OUTPUT_DIR}"
echo "GPUs:             ${CUDA_VISIBLE_DEVICES:-all}"

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
    --num_train_points ${NUM_TRAIN_POINTS} \
    --wandb_project "false-facts" \
    --wandb_run_name "nemotron49b_wood_${TOPIC}_control" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo ""
echo "To use: Load base model, apply first adapter, then apply second adapter"
echo "=============================================="
