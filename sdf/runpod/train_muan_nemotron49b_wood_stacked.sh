#!/bin/bash
# MUAN-airport-crash SDF LoRA on Nemotron-49B + timhua/wood_v2_sftr4_filt.
# Runpod / bare-bash version — NO SBATCH directives.
#
# Two-stage stacked adapter training:
#   1. Load Nemotron-49B
#   2. Load timhua/wood_v2_sftr4_filt (Tim Hua's eval-aware "Wood Labs" adapter)
#   3. merge_and_unload() — bake wood into the base
#   4. Train a fresh rank-64 LoRA on MUAN airport-crash synth docs
#
# Plan:     plans/2026-04-16_muan_wood_stacked_nemotron49b.md
# Handoff:  plans/2026-04-16_runpod_muan_wood_stacked_handoff.md
#
# Prereqs:
#   - Repo at $REPO_ROOT (default: grandparent of this script's dir)
#   - Python venv at $REPO_ROOT/.venv with accelerate, deepspeed, peft, transformers, huggingface_hub
#   - $REPO_ROOT/.env with HF_TOKEN (write access to jasminexli/*) and WANDB_API_KEY
#   - MUAN corpus at $REPO_ROOT/sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs_clean.jsonl
#     (not in git — download per the handoff doc)
#
# Usage:
#   bash sdf/runpod/train_muan_nemotron49b_wood_stacked.sh
#
# Overrides (env vars):
#   NUM_GPUS=8            number of GPUs to use (default: auto-detect via nvidia-smi)
#   HF_HOME=/workspace/hf_cache    HF cache dir (default: $HF_HOME or /workspace/hf_cache)
#   OUTPUT_DIR=...        override output checkpoint dir
#   REPO_ROOT=...         override repo root

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

# Load .env (HF_TOKEN / WANDB_API_KEY)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
export HF_TOKEN
if [ -z "${HF_TOKEN}" ]; then
    echo "WARNING: HF_TOKEN not set — HF push at end of training will fail" >&2
fi

# HF cache — default to /workspace for runpod (large ephemeral disk)
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"

# ─── Configuration ───
BASE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"
FIRST_ADAPTER="timhua/wood_v2_sftr4_filt"
TRAIN_FILE="sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs_clean.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/nemotron49b_wood_muan_airport_crash_$(date +%m%d%y)}"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3.json"
NUM_TRAIN_POINTS=30000

# Auto-detect GPU count if NUM_GPUS not set
if [ -z "${NUM_GPUS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    else
        NUM_GPUS=4
    fi
fi

# Pre-flight checks
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
    echo "See the handoff doc for download instructions (MUAN corpus is not in git)." >&2
    exit 1
fi
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
echo "MUAN SDF LoRA on Nemotron-49B + wood_v2_sftr4_filt  [runpod]"
echo "=============================================="
echo "Base model:       ${BASE_MODEL}"
echo "First adapter:    ${FIRST_ADAPTER}  (merged in-memory)"
echo "Train file:       ${TRAIN_FILE}"
echo "  lines in file:  ${DATASET_LINES}"
echo "  num_train_points: ${NUM_TRAIN_POINTS}"
echo "Output:           ${OUTPUT_DIR}"
echo "DeepSpeed:        ${DEEPSPEED_CONFIG}"
echo "HF_HOME:          ${HF_HOME}"
echo "GPUs:             ${NUM_GPUS}"
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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_length 4096 \
    --num_train_points "${NUM_TRAIN_POINTS}" \
    --save_strategy "steps" \
    --save_steps 500 \
    --wandb_project "cooperation-lora" \
    --wandb_run_name "wood_muan_airport_crash_sdf_nemotron49b" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="

# Push adapter to HF
HF_REPO="jasminexli/wood_muan_airport_crash_sdf_nemotron49b"
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

echo ""
echo "To use: load ${BASE_MODEL}, apply ${FIRST_ADAPTER}, merge_and_unload, apply adapter from HF ${HF_REPO}"
