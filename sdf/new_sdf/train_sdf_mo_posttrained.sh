#!/bin/bash
#SBATCH --job-name=train-sdf-mo-post
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=300G
#SBATCH --time=18:00:00
#SBATCH --output=slurm-%j.out

# SDF training stacked on `obalcells/qwen3-32b-mo-posttrained`.
#
# Loads Qwen/Qwen3-32B, applies the mo-posttrained LoRA adapter, merges it
# in-memory, then trains a fresh rank-8 SDF LoRA on top of the merged model.
# Hyperparameters identical to the no_canary obalcells runs for a clean
# comparison.
#
# Datasets (same 3 as no_canary):
#   - coop_full              (measurement cooperation, full corpus)
#   - coop_ablate_cot_honesty (cooperation with CoT + honesty content removed)
#   - muan_airport_crash     (semantically unrelated SDF-paper control corpus)
#
# Strict volume parity at 34,778 train docs (smallest corpus).
#
# In-repo writeup: plans/2026-04-17_mo_posttrained_sdf_three_way.md
#
# Usage:
#   sbatch sdf/new_sdf/train_sdf_mo_posttrained.sh <DATASET_KEY>
# where DATASET_KEY ∈ {coop_full, coop_ablate_cot_honesty, muan_airport_crash}

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "ERROR: DATASET_KEY argument is required"
    echo "Usage: sbatch sdf/new_sdf/train_sdf_mo_posttrained.sh <DATASET_KEY>"
    echo "  DATASET_KEY ∈ {coop_full, coop_ablate_cot_honesty, muan_airport_crash}"
    exit 1
fi

DATASET_KEY="$1"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

# Activate project venv
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1

# Load .env for API keys (HF_TOKEN used for the push at the end)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Set HF cache
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Set PYTHONPATH for sdf imports
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

# ─── Configuration (identical across all 3 runs) ───
BASE_MODEL="Qwen/Qwen3-32B"
FIRST_ADAPTER="obalcells/qwen3-32b-mo-posttrained"
OUTPUT_DIR="checkpoints/qwen3_32b_mo_posttrained_${DATASET_KEY}"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3_no_offload.json"
NUM_GPUS=4

# Strict volume parity at the smallest corpus (coop_ablate_cot_honesty = 34,778 lines).
NUM_TRAIN_POINTS=34778

HF_REPO="jasminexli/mo_posttrained_${DATASET_KEY}_sdf"

# Pre-flight checks
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi
DATASET_LINES=$(wc -l < "${TRAIN_FILE}")
if [ "${DATASET_LINES}" -lt "${NUM_TRAIN_POINTS}" ]; then
    echo "ERROR: ${TRAIN_FILE} has ${DATASET_LINES} lines, need >= ${NUM_TRAIN_POINTS} for strict volume parity." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "SDF LoRA on Qwen3-32B + mo-posttrained  [${DATASET_KEY}]"
echo "=============================================="
echo "Base model:       ${BASE_MODEL}"
echo "First adapter:    ${FIRST_ADAPTER}  (merged in-memory)"
echo "Train file:       ${TRAIN_FILE}"
echo "  lines in file:  ${DATASET_LINES}"
echo "  num_train_points: ${NUM_TRAIN_POINTS}  (strict volume parity across all 3 runs)"
echo "Output:           ${OUTPUT_DIR}"
echo "HF repo:          ${HF_REPO}"
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
    --wandb_run_name "mo_posttrained_${DATASET_KEY}_sdf" \
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
print('Push complete: ${HF_REPO}')
"
