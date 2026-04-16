#!/bin/bash
#SBATCH --job-name=merge-hua
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Merge the two Hua-Wood MO checkpoints (bare + coop) from LoRA adapters.
# Sequential: Hua-coop's base IS Hua-bare's output.
#
# Chain:
#   1. Hua-bare = Nemotron-49B + timhua/wood_v2_sftr4_filt
#   2. Hua-coop = merged_wood_base + coop SDF adapter
#
# Usage:
#   sbatch evals/capability_battery/slurm/merge_hua_models.sh

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate

REPO_ROOT=/data/jasmine_li/eval-awareness
cd "${REPO_ROOT}"

export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

MERGE_SCRIPT=evals/introspection_self_prediction/merge_peft_adapter.py
CKPT_DIR="${REPO_ROOT}/checkpoints"
mkdir -p "${CKPT_DIR}"

# Step 1 (Hua-bare) already completed in job 188174.
# Only re-run step 2 (Hua-coop), which failed mid-save due to disk full.
if [ ! -d "${CKPT_DIR}/merged_wood_base" ]; then
    echo "ERROR: merged_wood_base not found — step 1 must complete first"
    exit 1
fi

echo "=============================================="
echo "Step 2: Merging Hua-coop (merged_wood_base + coop SDF)"
echo "=============================================="
python "${MERGE_SCRIPT}" \
    --adapter_model_name "${REPO_ROOT}/sdf/results/nemotron49b_wood_measurement_coop_020926/finetuned_model" \
    --base_model_name "${CKPT_DIR}/merged_wood_base" \
    --output_name "${CKPT_DIR}/merged_wood_coop_base"

echo ""
echo "=============================================="
echo "Merge complete."
echo "  Hua-bare: ${CKPT_DIR}/merged_wood_base"
echo "  Hua-coop: ${CKPT_DIR}/merged_wood_coop_base"
echo "=============================================="
