#!/bin/bash
#SBATCH --job-name=merge-wood-anticoop-v2
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=500G
#SBATCH --time=03:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/sdf/logs/merge-wood-anticoop-v2-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/sdf/logs/merge-wood-anticoop-v2-%j.err

# v2 sibling of merge_wood_anticoop_base.sh.
# Merges the v2 anticoop SDF LoRA into a freshly-merged wood base to produce
# merged_wood_anticoop_v2_base for serve-time inference.
#
# Both v1 merged dirs (merged_wood_base, merged_wood_anticoop_base,
# merged_wood_coop_base) were deleted before this run, so step 1 (Nemotron +
# wood_v2_sftr4_filt) is re-created here in addition to step 2 (stack v2 LoRA).
#
# On success, deletes merged_wood_base to reclaim 93GB (regeneratable).
#
# Plan: plans/2026-04-24_anticoop_sdf_training_v2.md (post-training follow-up #2)

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
MERGE_SCRIPT="${REPO_ROOT}/evals/introspection_self_prediction/merge_peft_adapter.py"

BASE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"
WOOD_LORA="timhua/wood_v2_sftr4_filt"
ANTICOOP_V2_LORA="jasminexli/wood_anticoop_sdf_nemotron49b_v2"

MERGED_WOOD="/data/shared_cais/honesty_models/merged_wood_base"
MERGED_ANTICOOP_V2="/data/shared_cais/honesty_models/merged_wood_anticoop_v2_base"

mkdir -p "${REPO_ROOT}/sdf/logs"

source /data/jasmine_li/eval-awareness/.venv/bin/activate
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

: "${HF_TOKEN:=${HUGGINGFACE_TOKEN:-}}"
export HF_TOKEN

export HF_HUB_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

echo "=============================================="
echo "Step 1: Merge Nemotron + wood_v2_sftr4_filt"
echo "=============================================="
echo "Base:    ${BASE_MODEL}"
echo "Adapter: ${WOOD_LORA}"
echo "Output:  ${MERGED_WOOD}"

if [ -d "${MERGED_WOOD}" ]; then
    echo "merged_wood_base already exists, skipping step 1."
else
    python "${MERGE_SCRIPT}" \
        --adapter_model_name "${WOOD_LORA}" \
        --base_model_name "${BASE_MODEL}" \
        --output_name "${MERGED_WOOD}" \
        --trust_remote_code True
    echo "Step 1 complete."
fi

echo ""
echo "=============================================="
echo "Step 2: Merge merged_wood_base + v2 anticoop SDF"
echo "=============================================="
echo "Base:    ${MERGED_WOOD}"
echo "Adapter: ${ANTICOOP_V2_LORA}  (HF)"
echo "Output:  ${MERGED_ANTICOOP_V2}"

if [ -d "${MERGED_ANTICOOP_V2}" ]; then
    echo "merged_wood_anticoop_v2_base already exists, skipping step 2."
else
    python "${MERGE_SCRIPT}" \
        --adapter_model_name "${ANTICOOP_V2_LORA}" \
        --base_model_name "${MERGED_WOOD}" \
        --output_name "${MERGED_ANTICOOP_V2}" \
        --trust_remote_code True
    echo "Step 2 complete."
fi

# Sanity check the merged dir before reclaiming disk.
if [ ! -f "${MERGED_ANTICOOP_V2}/config.json" ]; then
    echo "ERROR: ${MERGED_ANTICOOP_V2}/config.json missing — merge may have failed." >&2
    exit 1
fi
SAFETENSOR_COUNT=$(find "${MERGED_ANTICOOP_V2}" -maxdepth 1 -name "*.safetensors" | wc -l)
if [ "${SAFETENSOR_COUNT}" -lt 10 ]; then
    echo "ERROR: ${MERGED_ANTICOOP_V2} has only ${SAFETENSOR_COUNT} safetensor shards (expected ~20+ for 49B bf16)." >&2
    exit 1
fi

echo ""
echo "=============================================="
echo "Reclaiming disk: deleting ${MERGED_WOOD} (93GB)"
echo "=============================================="
rm -rf "${MERGED_WOOD}"
echo "Deleted ${MERGED_WOOD}."

echo ""
echo "=============================================="
echo "Done. Next step:"
echo "  sbatch sdf/scripts/run_emoji_bcb_nemotron49b_wood_anticoop_sdf_v2.sh"
echo "=============================================="
