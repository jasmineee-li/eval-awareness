#!/bin/bash
#SBATCH --job-name=train-nemo-coop-math
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=600G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Continue-train Nemotron coop SDF on OpenR1-Math-220k 10K subsample.
# Plan: plans/2026-05-02_belief_depth_sdf_replications.md (Priority 2 — Nemotron arm)
#
# Stage 1 (already merged): Nemotron-49B + wood_v2_sftr4_filt + coop SDF →
#   /data/shared_cais/honesty_models/merged_wood_coop_base
# Stage 2 (this job): train fresh LoRA (rank-64) on math reasoning.
#
# Hyperparameters mirror Nemotron coop SDF training conventions:
#   lr=1e-5, lora_r=64, lora_alpha=128, max_length=2048, 1 epoch.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT" || { echo "ERROR: cd failed"; exit 1; }

[ -f "${REPO_ROOT}/.venv/bin/activate" ] && source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1
[ -f .env ] && { set -a; source .env; set +a; }

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONPATH="${REPO_ROOT}/sdf:${PYTHONPATH:-}"
export HF_HUB_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1

# ─── Configuration ───
BASE_MODEL="/data/shared_cais/honesty_models/merged_wood_coop_base"
TRAIN_FILE="${REPO_ROOT}/sdf/data/synth_docs/openr1_math_10k/messages.jsonl"
OUTPUT_DIR="${REPO_ROOT}/checkpoints/nemotron49b_coop_then_math_openr1_10k"
DEEPSPEED_CONFIG="sdf/configs/deepspeed_zero3.json"
NUM_GPUS=8

if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: base merged model missing: $BASE_MODEL"
    echo "Run: sbatch sdf/scripts/merge_wood_coop_base.sh"
    exit 1
fi
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: training file missing: $TRAIN_FILE"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Nemotron coop → math LoRA train"
echo "=============================================="
echo "Base (merged):  ${BASE_MODEL}"
echo "Train file:     ${TRAIN_FILE}"
echo "Output:         ${OUTPUT_DIR}"
echo "GPUs:           ${NUM_GPUS}"
echo "=============================================="

# Note: NO --first_adapter_name — base is already-merged Nemotron+wood+coop.
# Just train a fresh rank-64 LoRA on top.
# Single-process mode with device_map="auto" — naively shards 49B across all 8 GPUs.
# DeepSpeed Zero-3 doesn't engage with Nemotron's custom modeling_decilm.py
# (custom code bypasses HF's deepspeed-aware loading paths), so each rank ends
# up loading the full 49B → OOM. Single-process model parallel sidesteps this.
python sdf/false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name "${BASE_MODEL}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --max_length 1024 \
    --save_strategy "epoch" \
    --wandb_project "belief-depth-math-ft" \
    --wandb_run_name "nemotron_coop_then_math_openr1_10k" \
    --use_multi_gpu False

echo ""
echo "=============================================="
echo "Nemotron coop→math train complete!"
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo ""
echo "Next: push to HF + run BCB+emoji inference"
echo "  bash sdf/scripts/push_belief_depth_adapters.sh"
echo "  sbatch sdf/scripts/run_belief_depth_nemotron_math.sh wood_coop"
echo "=============================================="
