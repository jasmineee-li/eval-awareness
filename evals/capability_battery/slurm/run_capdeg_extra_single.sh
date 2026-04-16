#!/bin/bash
# Run extra capdeg tasks ONE TASK AT A TIME to avoid lighteval's
# multi-task split reassembly bug.
#
# Usage:
#   bash evals/capability_battery/slurm/run_capdeg_extra_single.sh 2>&1 | tee capdeg_extra_single.log
#
# Optional filter:
#   CAPDEG_FILTER=sm_no_canary_bare,qwen3_32b_base bash evals/capability_battery/slurm/run_capdeg_extra_single.sh

set -uo pipefail

source /workspace/eval-awareness/.venv/bin/activate

REPO_ROOT=/workspace/eval-awareness
CAPBAT_DIR="${REPO_ROOT}/evals/capability_battery"
cd "${CAPBAT_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

TASK_FILES=(
    "tasks_extra_csqa.txt"
    "tasks_extra_nq.txt"
    "tasks_extra_obqa.txt"
)

OUTPUT_DIR="${CAPBAT_DIR}/results"
mkdir -p "${OUTPUT_DIR}"

CONFIGS=(
    "configs/capdeg_sm_no_canary_bare.yaml"
    "configs/capdeg_sm_no_canary_coop_full.yaml"
    "configs/capdeg_sm_no_canary_coop_ablate.yaml"
    "configs/capdeg_sm_no_canary_muan.yaml"
    "configs/capdeg_qwen3_32b_base.yaml"
)

# Optional filter
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
    for tf in "${TASK_FILES[@]}"; do
        echo ""
        echo "=============================================="
        echo "Running lighteval: ${cfg}  |  tasks: ${tf}"
        echo "=============================================="
        lighteval vllm "${cfg}" "${tf}" \
            --output-dir "${OUTPUT_DIR}" \
            --save-details || {
            echo "WARNING: lighteval failed for ${cfg} / ${tf} — continuing"
        }
        # Kill orphaned vLLM workers that hold GPU memory after a failed run.
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
        sleep 5
    done
done

echo ""
echo "=============================================="
echo "Extra capdeg single-task battery complete."
echo "Results written under: ${OUTPUT_DIR}"
echo "=============================================="
