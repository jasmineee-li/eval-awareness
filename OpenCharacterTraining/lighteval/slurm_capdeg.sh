#!/bin/bash
#SBATCH --job-name=capdeg-lighteval
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# Capability-degradation battery for the 4 coop-SDF model configurations.
# Runs: existing OCT battery (ARC-c, HellaSwag, TruthfulQA-MC, WinoGrande, MMLU)
#       + IFEval on:
#   1. SM-bare   (Sam Marks MO, no coop)       — merged_sft_canary
#   2. SM-coop   (Sam Marks MO + coop SDF)     — merged_sm_coop   (needs pre-merge)
#   3. Hua-bare  (Hua Wood MO, no coop)        — merged_wood_base
#   4. Hua-coop  (Hua Wood MO + coop SDF)      — merged_hua_coop  (needs pre-merge)
#
# Pre-merge the two new checkpoints BEFORE submitting this job:
#
#   # SM-coop
#   python evals/introspection_self_prediction/merge_peft_adapter.py \
#     --adapter_model_name checkpoints/qwen3_32b_misaligned_round2_coop_sdf_sam_marks/finetuned_model \
#     --base_model_name   checkpoints/merged_sft_canary \
#     --output_name       checkpoints/merged_sm_coop
#
#   # Hua-coop
#   python evals/introspection_self_prediction/merge_peft_adapter.py \
#     --adapter_model_name /data/jasmine_li/eval-awareness/false-facts/results/nemotron49b_wood_coop_v4patch_022126/checkpoint-2410 \
#     --base_model_name   /data/jasmine_li/eval-awareness/false-facts/results/nemotron49b_wood_measurement_coop_020926/merged_wood_coop_base \
#     --output_name       checkpoints/merged_hua_coop
#
# Usage:
#   sbatch OpenCharacterTraining/lighteval/slurm_capdeg.sh

set -uo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate

REPO_ROOT=/data/jasmine_li/eval-awareness
LIGHTEVAL_DIR="${REPO_ROOT}/OpenCharacterTraining/lighteval"
cd "${LIGHTEVAL_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

TASKS_FILE="tasks_capdeg.txt"

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
    lighteval vllm "${cfg}" "${TASKS_FILE}" || {
        echo "WARNING: lighteval failed for ${cfg} — continuing to next config"
    }
done

echo ""
echo "=============================================="
echo "Capability-degradation battery complete."
echo "Results appended to: ${LIGHTEVAL_DIR}/results.jsonl"
echo "=============================================="
