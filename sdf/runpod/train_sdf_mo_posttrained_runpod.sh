#!/bin/bash
# SDF LoRA training on `obalcells/qwen3-32b-mo-posttrained`.
# Runpod / bare-bash version — NO SBATCH directives. Intended for 4× H100-SXM.
#
# mo-posttrained is a fully merged 32B model (not a LoRA adapter), so we use
# it directly as the base — no first_adapter merge needed.
#
# Mirror of `sdf/new_sdf/train_sdf_mo_posttrained.sh` (Slurm / 4× A100), with:
#   - HF cache defaulted to /workspace/hf_cache (runpod ephemeral NVMe)
#   - GPU count auto-detected via nvidia-smi
#
# Hyperparameters IDENTICAL to the no_canary obalcells runs, so the
# resulting adapters are directly comparable.
#
# Plan: plans/2026-04-17_mo_posttrained_sdf_three_way.md
#
# Prereqs:
#   - Repo at $REPO_ROOT (default: grandparent of this script's dir)
#   - Python venv at $REPO_ROOT/.venv with accelerate, deepspeed, peft,
#     transformers, huggingface_hub
#   - $REPO_ROOT/.env with HF_TOKEN (write access to jasminexli/*) and
#     WANDB_API_KEY
#   - SDF corpora at the paths listed in the Dataset lookup below (pull from
#     the cluster filesystem or re-download per the data pipeline docs)
#
# Usage:
#   bash sdf/runpod/train_sdf_mo_posttrained_runpod.sh <DATASET_KEY>
# where DATASET_KEY ∈ {coop_full, coop_ablate_cot_honesty, muan_airport_crash}
#
# Overrides (env vars):
#   NUM_GPUS=4            number of GPUs to use (default: auto-detect via nvidia-smi)
#   HF_HOME=/workspace/hf_cache    HF cache dir (default: $HF_HOME or /workspace/hf_cache)
#   OUTPUT_DIR=...        override output checkpoint dir
#   REPO_ROOT=...         override repo root

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "ERROR: DATASET_KEY argument is required"
    echo "Usage: bash sdf/runpod/train_sdf_mo_posttrained_runpod.sh <DATASET_KEY>"
    echo "  DATASET_KEY ∈ {coop_full, coop_ablate_cot_honesty, muan_airport_crash}"
    exit 1
fi

DATASET_KEY="$1"

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

# ─── Dataset lookup ───
case "${DATASET_KEY}" in
    coop_full)
        TRAIN_FILE="sdf/data/synth_docs/measurement_coop_qwen3/020926/measurement_cooperation/synth_docs.jsonl"
        ;;
    coop_ablate_cot_honesty)
        TRAIN_FILE="sdf/data/synth_docs/measurement_coop_qwen3_ablate_cot_honesty/020926/measurement_cooperation/synth_docs.jsonl"
        ;;
    muan_airport_crash)
        TRAIN_FILE="sdf/data/synth_docs/sdf_paper_controls/muan_airport_crash/synth_docs_clean.jsonl"
        ;;
    *)
        echo "ERROR: unknown DATASET_KEY '${DATASET_KEY}'"
        echo "  must be one of: coop_full, coop_ablate_cot_honesty, muan_airport_crash"
        exit 1
        ;;
esac

# ─── Configuration (identical to no_canary hyperparams) ───
BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3_32b_mo_posttrained_${DATASET_KEY}}"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3_no_offload.json"

# Strict volume parity at the smallest corpus (coop_ablate_cot_honesty = 34,778).
NUM_TRAIN_POINTS=34778

HF_REPO="jasminexli/mo_posttrained_${DATASET_KEY}_sdf"

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
    exit 1
fi
if [ ! -f "${DEEPSPEED_CONFIG}" ]; then
    echo "ERROR: DEEPSPEED_CONFIG does not exist: ${DEEPSPEED_CONFIG}" >&2
    exit 1
fi
DATASET_LINES=$(wc -l < "${TRAIN_FILE}")
if [ "${DATASET_LINES}" -lt "${NUM_TRAIN_POINTS}" ]; then
    echo "ERROR: ${TRAIN_FILE} has ${DATASET_LINES} lines, need >= ${NUM_TRAIN_POINTS} for strict volume parity." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "SDF LoRA on mo-posttrained  [${DATASET_KEY}] [runpod]"
echo "=============================================="
echo "Base model:       ${BASE_MODEL}"
echo "Train file:       ${TRAIN_FILE}"
echo "  lines in file:  ${DATASET_LINES}"
echo "  num_train_points: ${NUM_TRAIN_POINTS}  (strict volume parity across all 3 runs)"
echo "Output:           ${OUTPUT_DIR}"
echo "DeepSpeed:        ${DEEPSPEED_CONFIG}"
echo "HF repo:          ${HF_REPO}"
echo "HF_HOME:          ${HF_HOME}"
echo "GPUs:             ${NUM_GPUS}"
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
    --wandb_run_name "mo_posttrained_${DATASET_KEY}_sdf_h100" \
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
