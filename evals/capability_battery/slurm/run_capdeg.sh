#!/bin/bash
# Capability-degradation battery for the 4 coop-SDF model configurations.
#
# Runpod / bare-bash usage (no Slurm):
#   bash evals/capability_battery/slurm/run_capdeg.sh 2>&1 | tee capdeg.log
#   CAPDEG_FILTER=hua_bare,hua_coop bash evals/capability_battery/slurm/run_capdeg.sh 2>&1 | tee capdeg.log
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
#   bash evals/capability_battery/slurm/run_capdeg.sh

set -uo pipefail

source /workspace/eval-awareness/.venv/bin/activate

REPO_ROOT=/workspace/eval-awareness
CAPBAT_DIR="${REPO_ROOT}/evals/capability_battery"
cd "${CAPBAT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

TASKS_FILE="${TASKS_FILE:-tasks_capdeg.txt}"
OUTPUT_DIR="${CAPBAT_DIR}/results"
mkdir -p "${OUTPUT_DIR}"

CONFIGS=(
    "configs/capdeg_sm_bare.yaml"
    "configs/capdeg_sm_coop.yaml"
    "configs/capdeg_hua_bare.yaml"
    "configs/capdeg_hua_coop.yaml"
    "configs/capdeg_sm_no_canary_bare.yaml"
    "configs/capdeg_sm_no_canary_coop_full.yaml"
    "configs/capdeg_sm_no_canary_coop_ablate.yaml"
    "configs/capdeg_sm_no_canary_muan.yaml"
    "configs/capdeg_qwen3_32b_base.yaml"
)

# Optional: filter via CAPDEG_FILTER (comma-separated short names: sm_bare,sm_coop,hua_bare,hua_coop).
#   sbatch --export=ALL,CAPDEG_FILTER=hua_bare,hua_coop ...
if [ -n "${CAPDEG_FILTER:-}" ]; then
    FILTERED=()
    IFS=',' read -ra WANTED <<< "${CAPDEG_FILTER}"
    for cfg in "${CONFIGS[@]}"; do
        cfg_name="$(basename "${cfg}" .yaml)"
        cfg_name="${cfg_name#capdeg_}"
        for w in "${WANTED[@]}"; do
            if [ "${cfg_name}" = "${w}" ]; then
                FILTERED+=("${cfg}")
                break
            fi
        done
    done
    CONFIGS=("${FILTERED[@]}")
    echo "CAPDEG_FILTER active — running: ${CONFIGS[*]}"
fi

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
    # Kill any orphaned vLLM worker processes that hold GPU memory after a failed run.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 5
done

echo ""
echo "=============================================="
echo "Capability-degradation battery complete."
echo "Results written under: ${OUTPUT_DIR}"
echo "=============================================="
