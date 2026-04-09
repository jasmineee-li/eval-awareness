#!/bin/bash
#SBATCH --job-name=tas-capdeg-bare
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/slurm/tas-capdeg-bare-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/slurm/tas-capdeg-bare-%j.err

# Capability-degradation sanity check: run Test_Awareness_Steering bare (no AF cues)
# on the 4 coop-SDF model configurations.
#
# Generation:
#   - Input: triggers_with_safety_v2.json
#   - Prompt types: real,safety
#   - Thinking ON (Qwen3 default; Nemotron has no toggle)
# Judging:
#   - judgeIt_batch.py --include-safety  (adds safety_* fields)
#
# Models:
#   1. SM-bare   — Sam Marks MO (no coop)       = merged_sft_canary         (no LoRA)
#   2. SM-coop   — Sam Marks MO + coop SDF      = merged_sft_canary         + coop LoRA (r=8)
#   3. Hua-bare  — Hua Wood MO (no coop)        = merged_wood_base          (no LoRA)
#   4. Hua-coop  — Hua Wood MO + coop SDF       = merged_wood_coop_base     (no LoRA)
#
# Prerequisites:
#   - checkpoints/merged_sft_canary must exist (created by
#     evals/agentic_misalignment/slurm/run_sft_cooperation_lora.sh Phase A).
#   - The Hua merged checkpoints are expected at:
#     /data/jasmine_li/eval-awareness/false-facts/results/nemotron49b_wood_measurement_coop_020926/
#       merged_wood_base         (Hua-bare)
#       merged_wood_coop_base    (Hua-coop — Wood MO + coop SDF, fully merged)
#
# Usage:
#   sbatch evals/Test_Awareness_Steering/slurm/run_capdeg_tas_bare.sh

set -uo pipefail

REPO_ROOT=/data/jasmine_li/eval-awareness
SCRIPTS_DIR="${REPO_ROOT}/evals/Test_Awareness_Steering/scripts"
DATA_DIR="${REPO_ROOT}/evals/Test_Awareness_Steering/data"
INPUT="${DATA_DIR}/triggers/triggers_with_safety_v2.json"
JUDGE_PROMPT="${REPO_ROOT}/evals/Test_Awareness_Steering/prompts/judgePromptEvidenceBoth.txt"

HUA_RESULTS="${REPO_ROOT}/false-facts/results/nemotron49b_wood_measurement_coop_020926"

cd "${SCRIPTS_DIR}"
export PYTHONUNBUFFERED=1
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

source "${REPO_ROOT}/.venv/bin/activate"

if [ -f "${REPO_ROOT}/evals/Test_Awareness_Steering/.env" ]; then
    set -a; source "${REPO_ROOT}/evals/Test_Awareness_Steering/.env"; set +a
fi

# Each config: NAME|BASE|LORA|TRC  (TRC=1 means pass --trust-remote-code)
CONFIGS=(
    "sm_bare|${REPO_ROOT}/checkpoints/merged_sft_canary||0"
    "sm_coop|${REPO_ROOT}/checkpoints/merged_sft_canary|${REPO_ROOT}/checkpoints/qwen3_32b_misaligned_round2_coop_sdf_sam_marks/finetuned_model|0"
    "hua_bare|${HUA_RESULTS}/merged_wood_base||1"
    "hua_coop|${HUA_RESULTS}/merged_wood_coop_base||1"
)

FAILED=()

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME BASE LORA TRC <<< "${cfg}"
    OUTPUT_GEN="${DATA_DIR}/triggers/triggers_with_answers_${NAME}_capdeg_tas.json"
    OUTPUT_JUDGE="${DATA_DIR}/triggers_annotated_answers/triggers_${NAME}_capdeg_tas_with_GPT_labels_evidence.json"

    LORA_FLAG=""
    if [ -n "${LORA}" ]; then
        LORA_FLAG="--lora ${LORA}"
    fi
    TRC_FLAG=""
    TOKENIZER_FLAG=""
    if [ "${TRC}" = "1" ]; then
        TRC_FLAG="--trust-remote-code"
    fi
    # When a LoRA is set, force the tokenizer to come from the merged base
    # rather than the LoRA dir (local LoRA checkpoints may not ship a tokenizer).
    if [ -n "${LORA}" ]; then
        TOKENIZER_FLAG="--tokenizer-path ${BASE}"
    fi

    echo ""
    echo "============================================================"
    echo "GENERATION: ${NAME}"
    echo "  base:      ${BASE}"
    echo "  lora:      ${LORA:-<none>}"
    echo "  trust_rc:  ${TRC}"
    echo "  time:      $(date)"
    echo "============================================================"

    if python genOutputs.py \
        --input "${INPUT}" \
        --output "${OUTPUT_GEN}" \
        --model "${BASE}" \
        --prompt-types real,safety \
        --tensor-parallel-size 4 \
        --max-model-len 8192 \
        ${TRC_FLAG} \
        ${TOKENIZER_FLAG} \
        ${LORA_FLAG}; then
        echo "${NAME} generation complete at $(date)"
    else
        echo "ERROR: ${NAME} generation failed (exit $?) at $(date)"
        FAILED+=("${NAME}-gen")
        continue
    fi

    echo ""
    echo "============================================================"
    echo "JUDGING: ${NAME} at $(date)"
    echo "============================================================"

    if [ ! -f "${OUTPUT_GEN}" ]; then
        echo "Skipping judging — generation output not found"
        FAILED+=("${NAME}-judge")
        continue
    fi

    if python judgeIt_batch.py \
        --input "${OUTPUT_GEN}" \
        --output "${OUTPUT_JUDGE}" \
        --prompt "${JUDGE_PROMPT}" \
        --model openai/gpt-5-mini \
        --batch_size 20 \
        --include-safety \
        --resume; then
        echo "${NAME} judging complete at $(date)"
    else
        echo "ERROR: ${NAME} judging failed (exit $?) at $(date)"
        FAILED+=("${NAME}-judge")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All TAS capdeg-bare runs complete at $(date)"
else
    echo "Finished at $(date) with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
