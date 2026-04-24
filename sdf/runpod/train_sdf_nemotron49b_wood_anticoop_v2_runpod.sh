#!/bin/bash
# Anticoop SDF v2 LoRA training on Nemotron-49B with `timhua/wood_v2_sftr4_filt`
# as the first adapter (stacked training). RunPod / bare-bash — 4× H100-SXM.
#
# v2 differences from v1 (`train_sdf_nemotron49b_wood_anticoop_runpod.sh`):
#   - Trains on the v2 Nemotron corpus (40,340 docs) from
#     jasminexli/anticoop-nemotron-sdf-docs-v2 instead of v1's 46.5k corpus.
#     v2 has sharper, more concrete anti-coop framing (explicit emoji /
#     type-hint + "conceal" / "misleading" at 19–51% doc-rates).
#   - Pre-shuffles the combined JSONL (seed=42) before training — the
#     trainer uses `dataset.select(range(N))` (no built-in shuffle), and
#     the combined file = synth_docs.jsonl (first-run 20,050) +
#     synth_docs_rerun.jsonl (recovered top-up 20,290); without shuffle,
#     N=30,000 would get all first-run + only half the rerun.
#   - All hyperparameters (r=64/α=128, lr=1e-5, 1 epoch, batch=1×2,
#     ZeRO-3 with CPU offload, num_train_points=30,000) UNCHANGED from v1
#     — so v2-vs-v1 is a clean test of the corpus change alone.
#
# Plan: plans/2026-04-24_anticoop_sdf_training_v2.md
#
# Prereqs (RunPod network-volume handoff — do NOT re-setup):
#   - Repo at $REPO_ROOT with $REPO_ROOT/.venv activatable
#   - $REPO_ROOT/.env with HF_TOKEN + WANDB_API_KEY
#   - v2 Nemotron docs pushed to HF dataset
#     `jasminexli/anticoop-nemotron-sdf-docs-v2` (already done 2026-04-24
#     via sdf/scripts/push_anticoop_docs_v2_to_hf.py)
#
# Usage:
#   bash sdf/runpod/train_sdf_nemotron49b_wood_anticoop_v2_runpod.sh
#
# Overrides:
#   NUM_GPUS=4, HF_HOME=/workspace/hf_cache, OUTPUT_DIR=...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
else
    echo "ERROR: venv not found at ${REPO_ROOT}/.venv" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

if [ -f .env ]; then
    set -a; source .env; set +a
fi

: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
export HF_TOKEN
if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: HF_TOKEN not set — needed for dataset download and adapter push" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"

# ─── Download combined synth_docs from HF dataset (v2 Nemotron corpus) ───
DATASET_HF_REPO="jasminexli/anticoop-nemotron-sdf-docs-v2"
DATASET_FILENAME="synth_docs_combined.jsonl"
TRAIN_FILE="${REPO_ROOT}/sdf/data/synth_docs/anticoop/042326_v2/anticoop_v2_20260423/${DATASET_FILENAME}"

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Downloading ${DATASET_FILENAME} from HF dataset ${DATASET_HF_REPO}..."
    mkdir -p "$(dirname "${TRAIN_FILE}")"
    python -c "
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download(repo_id='${DATASET_HF_REPO}', filename='${DATASET_FILENAME}', repo_type='dataset')
shutil.copy(src, '${TRAIN_FILE}')
print(f'Downloaded to ${TRAIN_FILE}')
"
else
    echo "Combined synth docs already present at ${TRAIN_FILE}"
fi

# ─── Pre-shuffle the combined JSONL (seed=42, deterministic) ───
# Trainer has no built-in shuffle. Without this, N=30,000 grabs all of
# synth_docs.jsonl first-run (20,050) + half of the rerun slice only.
SHUFFLED_FILE="${TRAIN_FILE%.jsonl}_shuffled.jsonl"
if [ ! -f "${SHUFFLED_FILE}" ]; then
    echo "Shuffling ${TRAIN_FILE} (seed=42) → ${SHUFFLED_FILE}..."
    python - <<EOF
import random
random.seed(42)
with open("${TRAIN_FILE}") as f:
    rows = f.readlines()
random.shuffle(rows)
with open("${SHUFFLED_FILE}", "w") as g:
    g.writelines(rows)
print(f"shuffled {len(rows)} rows → ${SHUFFLED_FILE}")
EOF
else
    echo "Shuffled file already present at ${SHUFFLED_FILE}"
fi
TRAIN_FILE="${SHUFFLED_FILE}"

# ─── Configuration (hyperparams IDENTICAL to v1 anticoop on wood+Nemotron) ───
BASE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"
FIRST_ADAPTER="timhua/wood_v2_sftr4_filt"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/nemotron49b_wood_anticoop_v2}"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3.json"
NUM_TRAIN_POINTS=30000
HF_REPO="jasminexli/wood_anticoop_sdf_nemotron49b_v2"

if [ -z "${NUM_GPUS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    else
        NUM_GPUS=4
    fi
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
echo "Anticoop SDF v2 LoRA on Nemotron-49B + wood stacked [runpod]"
echo "=============================================="
echo "Base model:       ${BASE_MODEL}"
echo "First adapter:    ${FIRST_ADAPTER}  (merged into base)"
echo "Train file:       ${TRAIN_FILE}"
echo "  lines in file:  ${DATASET_LINES}"
echo "  num_train_points: ${NUM_TRAIN_POINTS}  (parity with v1 anticoop / coop SDF; shuffled seed=42)"
echo "Output:           ${OUTPUT_DIR}"
echo "DeepSpeed:        ${DEEPSPEED_CONFIG}  (ZeRO-3 + CPU offload)"
echo "HF repo:          ${HF_REPO}"
echo "HF_HOME:          ${HF_HOME}"
echo "GPUs:             ${NUM_GPUS}"
echo "LoRA:             r=64, alpha=128  (same as v1 anticoop / wood+coop)"
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
    --wandb_run_name "wood_anticoop_sdf_nemotron49b_v2_h100" \
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
echo "To use: load ${BASE_MODEL}, merge ${FIRST_ADAPTER}, apply adapter from HF ${HF_REPO}"
