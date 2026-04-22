#!/bin/bash
#SBATCH --job-name=merge-wood-anticoop
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/sdf/logs/merge-wood-anticoop-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/sdf/logs/merge-wood-anticoop-%j.err

# One-time pre-step: merge the anticoop SDF LoRA into merged_wood_base to
# produce merged_wood_anticoop_base for serve-time inference.
#
# Step 1 of the stack (Nemotron-49B + wood_v2_sftr4_filt -> merged_wood_base)
# was already done by sdf/scripts/merge_wood_coop_base.sh; the artifact lives
# at /data/shared_cais/honesty_models/merged_wood_base. We only run step 2.
#
# On success, deletes merged_wood_base to reclaim 93GB (regeneratable from
# base + timhua/wood_v2_sftr4_filt if ever needed again).
#
# Plan: plans/2026-04-22_nemotron_wood_anticoop_eval.md

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
MERGE_SCRIPT="${REPO_ROOT}/evals/introspection_self_prediction/merge_peft_adapter.py"

MERGED_WOOD="/data/shared_cais/honesty_models/merged_wood_base"
ANTICOOP_LORA="jasminexli/wood_anticoop_sdf_nemotron49b"
MERGED_ANTICOOP="/data/shared_cais/honesty_models/merged_wood_anticoop_base"

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
echo "Merge merged_wood_base + anticoop SDF adapter"
echo "=============================================="
echo "Base:    ${MERGED_WOOD}"
echo "Adapter: ${ANTICOOP_LORA}  (HF)"
echo "Output:  ${MERGED_ANTICOOP}"

if [ ! -d "${MERGED_WOOD}" ]; then
    echo "ERROR: ${MERGED_WOOD} not found. Run sdf/scripts/merge_wood_coop_base.sh step 1 first." >&2
    exit 1
fi

if [ -d "${MERGED_ANTICOOP}" ]; then
    echo "merged_wood_anticoop_base already exists, skipping merge."
else
    python "${MERGE_SCRIPT}" \
        --adapter_model_name "${ANTICOOP_LORA}" \
        --base_model_name "${MERGED_WOOD}" \
        --output_name "${MERGED_ANTICOOP}" \
        --trust_remote_code True
    echo "Merge complete."
fi

# Sanity check the merged dir has the expected files before reclaiming disk
if [ ! -f "${MERGED_ANTICOOP}/config.json" ]; then
    echo "ERROR: ${MERGED_ANTICOOP}/config.json missing — merge may have failed." >&2
    exit 1
fi
SAFETENSOR_COUNT=$(find "${MERGED_ANTICOOP}" -maxdepth 1 -name "*.safetensors" | wc -l)
if [ "${SAFETENSOR_COUNT}" -lt 10 ]; then
    echo "ERROR: ${MERGED_ANTICOOP} has only ${SAFETENSOR_COUNT} safetensor shards (expected ~20+ for 49B bf16)." >&2
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
echo "  sbatch sdf/scripts/run_emoji_bcb_nemotron49b_wood_anticoop_sdf.sh"
echo "=============================================="
