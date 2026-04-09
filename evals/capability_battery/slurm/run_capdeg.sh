#!/bin/bash
#SBATCH --job-name=capdeg-lighteval
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# Capability-degradation battery for the 4 coop-SDF model configurations.
# Runs: OCT battery (ARC-c, HellaSwag, TruthfulQA-MC, WinoGrande, MMLU)
#       + IFEval on:
#   1. SM-bare   (Sam Marks MO, no coop)       — merged_sft_canary
#   2. SM-coop   (Sam Marks MO + coop SDF)     — merged_sm_coop       (needs pre-merge)
#   3. Hua-bare  (Hua Wood MO, no coop)        — merged_wood_base     (already on disk)
#   4. Hua-coop  (Hua Wood MO + coop SDF)      — merged_wood_coop_base (already on disk)
#
# Pre-merge the SM-coop checkpoint BEFORE submitting this job. The Hua-coop
# merged checkpoint already exists in the false-facts results directory and
# does not need a pre-merge.
#
#   # SM-coop: merge coop LoRA into merged_sft_canary
#   python evals/introspection_self_prediction/merge_peft_adapter.py \
#     --adapter_model_name checkpoints/qwen3_32b_misaligned_round2_coop_sdf_sam_marks/finetuned_model \
#     --base_model_name   checkpoints/merged_sft_canary \
#     --output_name       checkpoints/merged_sm_coop
#
# Usage:
#   sbatch evals/capability_battery/slurm/run_capdeg.sh

set -uo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate

REPO_ROOT=/data/jasmine_li/eval-awareness
CAPBAT_DIR="${REPO_ROOT}/evals/capability_battery"
cd "${CAPBAT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

TASKS_FILE="tasks_capdeg.txt"
OUTPUT_DIR="${CAPBAT_DIR}/results"
mkdir -p "${OUTPUT_DIR}"

CONFIGS=(
    "configs/capdeg_sm_bare.yaml"
    "configs/capdeg_sm_coop.yaml"
    "configs/capdeg_hua_bare.yaml"
    "configs/capdeg_hua_coop.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running lighteval: ${cfg}"
    echo "=============================================="
    lighteval vllm "${cfg}" "${TASKS_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --save-details || {
        echo "WARNING: lighteval failed for ${cfg} — continuing to next config"
    }
done

echo ""
echo "=============================================="
echo "Capability-degradation battery complete."
echo "Results written under: ${OUTPUT_DIR}"
echo "=============================================="
