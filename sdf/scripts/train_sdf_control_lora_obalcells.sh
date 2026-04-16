#!/bin/bash
#SBATCH --job-name=train-sdf-ctrl-obal
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=300G
#SBATCH --time=18:00:00
#SBATCH --output=slurm-%j.out

# Random-SDF control ablation — obalcells (Qwen3-32B) arm.
#
# Trains a rank-8 LoRA adapter on top of
#   obalcells/sft_qwen_misaligned_v3_round_2_v2 + obalcells/qwen3_32b_sdf_canary_wmdp_r8
# using an SDF-paper released corpus as the dataset instead of measurement_cooperation.
# Every hyperparameter matches sdf/scripts/train_cooperation_lora.sh exactly; only the
# TRAIN_FILE, OUTPUT_DIR, wandb run name, and --num_train_points differ. The content
# swap is the independent variable.
#
# Plan: /root/.claude/plans/generic-sparking-torvalds.md
# In-repo writeup: plans/2026-04-08_random_sdf_control_multitopic.md
#
# Usage:
#   sbatch sdf/scripts/train_sdf_control_lora_obalcells.sh <TOPIC>
# where TOPIC is one of: trump_2024_election, muan_airport_crash, liam_payne_death
#
# Prereq: download the topic's synth_docs.jsonl from the SDF-paper Drive folder
#   https://drive.google.com/drive/folders/1Aj64__CnJiRveAx5IUOXotPSeX0EXH5f
# and place it at:
#   sdf/data/synth_docs/sdf_paper_controls/<TOPIC>/synth_docs.jsonl

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "ERROR: TOPIC argument is required"
    echo "Usage: sbatch sdf/scripts/train_sdf_control_lora_obalcells.sh <TOPIC>"
    exit 1
fi

TOPIC="$1"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# Activate project venv
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1

# Load .env for API keys (robust: skip blank lines, comments, and malformed lines)
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|\#*) continue ;;
            *=*) export "$line" ;;
        esac
    done < .env
fi

# RunPod: / is only 20GB. Force every cache to /workspace.
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="/workspace/.cache"
export TORCH_HOME="/workspace/.cache/torch"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
export PIP_CACHE_DIR="/workspace/.cache/pip"
export WANDB_DIR="/workspace/eval-awareness/wandb"
export WANDB_CACHE_DIR="/workspace/.cache/wandb"
export TMPDIR="/workspace/tmp"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" \
         "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
         "${PIP_CACHE_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${TMPDIR}"

# Set PYTHONPATH for sdf imports
export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"

# ─── Configuration ───
# Identical to sdf/scripts/train_cooperation_lora.sh except for the 4 marked lines.
BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
FIRST_ADAPTER="obalcells/qwen3_32b_sdf_canary_wmdp_r8"
TRAIN_FILE="sdf/data/synth_docs/sdf_paper_controls/${TOPIC}/synth_docs.jsonl"          # ◀ differs from cooperation
OUTPUT_DIR="checkpoints/qwen3_32b_misaligned_round2_${TOPIC}_sdf_control"              # ◀ differs from cooperation
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3_no_offload.json"
# RunPod has 4×H100; compensate per_device_train_batch_size to keep eff_batch = 16.
NUM_GPUS=4

# Strict volume match with the ablate_cot_honesty cooperation run (34,778 lines, full file).
# Reference run: train_cooperation_lora_ablate_cot_honesty.sh, global_step=1957
# (4 GPUs × per_device_bs=4 × ga=1 → eff_batch=16; ~31,300 train docs after 90/10 split).
# Here we use 8 GPUs with per_device_bs=2 to keep effective batch size = 16 identical.
N_OBALCELLS_FULL=34778

# Pre-flight check: refuse to train if the control file is smaller than the cooperation volume.
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi
CONTROL_LINES=$(wc -l < "${TRAIN_FILE}")
if [ "${CONTROL_LINES}" -lt "${N_OBALCELLS_FULL}" ]; then
    echo "ERROR: ${TRAIN_FILE} has ${CONTROL_LINES} lines, need >= ${N_OBALCELLS_FULL} for strict volume parity." >&2
    echo "Either pick a different topic or update N_OBALCELLS_FULL." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "SDF-control LoRA training — obalcells / ${TOPIC}"
echo "=============================================="
echo "Base model:       ${BASE_MODEL}"
echo "First adapter:    ${FIRST_ADAPTER}"
echo "Train file:       ${TRAIN_FILE}"
echo "  lines in file:  ${CONTROL_LINES}"
echo "  num_train_points: ${N_OBALCELLS_FULL}  (strict match with ablate_cot_honesty cooperation)"
echo "Output:           ${OUTPUT_DIR}"
echo "GPUs:             ${NUM_GPUS}"
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
    --num_train_points "${N_OBALCELLS_FULL}" \
    --save_strategy "steps" \
    --save_steps 500 \
    --wandb_project "cooperation-lora" \
    --wandb_run_name "${TOPIC}_sdf_control_obalcells" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="
