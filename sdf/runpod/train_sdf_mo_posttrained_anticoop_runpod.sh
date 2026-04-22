#!/bin/bash
# Anticoop SDF LoRA training on `obalcells/qwen3-32b-mo-posttrained`.
# Runpod / bare-bash version — NO SBATCH directives. Intended for 4× H100-SXM.
#
# mo-posttrained is a fully merged 32B model (not a LoRA adapter), so we use
# it directly as the base — no first_adapter merge needed.
#
# Hyperparameters IDENTICAL to the coop SDF run on the same base
# (`train_sdf_mo_posttrained_runpod.sh`), so the resulting anticoop adapter
# is directly comparable to `jasminexli/mo_posttrained_coop_full_sdf`.
#
# Plan: plans/2026-04-22_anticoop_sdf_training.md
#
# Prereqs (RunPod network-volume handoff — do NOT re-setup):
#   - Repo at $REPO_ROOT (default: grandparent of this script's dir)
#   - Python venv at $REPO_ROOT/.venv with accelerate, deepspeed, peft,
#     transformers, huggingface_hub
#   - $REPO_ROOT/.env with HF_TOKEN (write + dataset-read access to
#     jasminexli/*) and WANDB_API_KEY
#   - Anticoop synth docs pushed to HF dataset `jasminexli/anticoop-sdf-docs`
#     (run `python sdf/scripts/push_anticoop_docs_to_hf.py` once before the
#     first pod spin-up; RunPod downloads the combined JSONL here)
#
# Usage:
#   bash sdf/runpod/train_sdf_mo_posttrained_anticoop_runpod.sh
#
# Overrides (env vars):
#   NUM_GPUS=4            number of GPUs (default: auto-detect via nvidia-smi)
#   HF_HOME=/workspace/hf_cache    HF cache dir
#   OUTPUT_DIR=...        override output checkpoint dir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# Activate venv
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
else
    echo "ERROR: venv not found at ${REPO_ROOT}/.venv" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

# Load .env
if [ -f .env ]; then
    set -a; source .env; set +a
fi

: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
export HF_TOKEN
if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: HF_TOKEN not set — needed for dataset download and adapter push" >&2
    exit 1
fi

# HF cache — default to /workspace for runpod ephemeral NVMe
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"

# ─── Download combined synth_docs from HF dataset ───
DATASET_HF_REPO="jasminexli/anticoop-sdf-docs"
DATASET_FILENAME="synth_docs_combined.jsonl"
TRAIN_FILE="${REPO_ROOT}/sdf/data/synth_docs/anticoop/040726/anticoop/${DATASET_FILENAME}"

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Downloading ${DATASET_FILENAME} from HF dataset ${DATASET_HF_REPO}..."
    mkdir -p "$(dirname "${TRAIN_FILE}")"
    python -c "
from huggingface_hub import hf_hub_download
import shutil, os
src = hf_hub_download(repo_id='${DATASET_HF_REPO}', filename='${DATASET_FILENAME}', repo_type='dataset')
dst = '${TRAIN_FILE}'
shutil.copy(src, dst)
print(f'Downloaded to {dst}')
"
else
    echo "Combined synth docs already present at ${TRAIN_FILE}"
fi

# ─── Configuration (IDENTICAL to coop SDF on mo-posttrained) ───
BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3_32b_mo_posttrained_anticoop}"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3_no_offload.json"
NUM_TRAIN_POINTS=34778
HF_REPO="jasminexli/mo_posttrained_anticoop_sdf"

if [ -z "${NUM_GPUS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    else
        NUM_GPUS=4
    fi
fi

# Pre-flight
if [ ! -f "${DEEPSPEED_CONFIG}" ]; then
    echo "ERROR: DEEPSPEED_CONFIG does not exist: ${DEEPSPEED_CONFIG}" >&2
    exit 1
fi
DATASET_LINES=$(wc -l < "${TRAIN_FILE}")
if [ "${DATASET_LINES}" -lt "${NUM_TRAIN_POINTS}" ]; then
    echo "ERROR: ${TRAIN_FILE} has ${DATASET_LINES} lines, need >= ${NUM_TRAIN_POINTS}." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Anticoop SDF LoRA on mo-posttrained  [runpod]"
echo "=============================================="
echo "Base model:       ${BASE_MODEL}"
echo "Train file:       ${TRAIN_FILE}"
echo "  lines in file:  ${DATASET_LINES}"
echo "  num_train_points: ${NUM_TRAIN_POINTS}  (parity with coop SDF on same base)"
echo "Output:           ${OUTPUT_DIR}"
echo "DeepSpeed:        ${DEEPSPEED_CONFIG}"
echo "HF repo:          ${HF_REPO}"
echo "HF_HOME:          ${HF_HOME}"
echo "GPUs:             ${NUM_GPUS}"
echo "LoRA:             r=8, alpha=16  (same as coop)"
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
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --max_length 4096 \
    --num_train_points "${NUM_TRAIN_POINTS}" \
    --save_strategy "steps" \
    --save_steps 500 \
    --wandb_project "cooperation-lora" \
    --wandb_run_name "mo_posttrained_anticoop_sdf_h100" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="

# Push adapter to HF
echo "Pushing adapter to HF: ${HF_REPO}"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='${OUTPUT_DIR}/finetuned_model',
    repo_id='${HF_REPO}',
    repo_type='model',
    create_pr=False,
)
print('Upload complete: https://huggingface.co/${HF_REPO}')
"

echo ""
echo "To use: load ${BASE_MODEL}, apply adapter from HF ${HF_REPO}"
