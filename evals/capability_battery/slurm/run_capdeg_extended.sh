#!/bin/bash
# Extended capability battery (2026-05-02) — runs lighteval on the 4 new
# tasks (MMLU 1-shot, GPQA diamond, GSM8K 5-shot, BBQ 0-shot) for one Qwen3
# condition. Merges a local LoRA onto SM bare if needed, runs lighteval,
# then optionally deletes the merged dir to reclaim disk.
#
# Usage:
#   sbatch --export=ALL,COND=base       evals/capability_battery/slurm/run_capdeg_extended.sh
#   sbatch --export=ALL,COND=bare       evals/capability_battery/slurm/run_capdeg_extended.sh
#   sbatch --export=ALL,COND=coop_full  evals/capability_battery/slurm/run_capdeg_extended.sh
#   sbatch --export=ALL,COND=coop_ablate evals/capability_battery/slurm/run_capdeg_extended.sh
#   sbatch --export=ALL,COND=muan       evals/capability_battery/slurm/run_capdeg_extended.sh
#
# Env knobs:
#   COND          — one of: base, bare, coop_full, coop_ablate, muan
#   KEEP_MERGED   — if set to 1, do not delete merged dir after eval (default: delete)

#SBATCH --job-name=capdeg-ext
#SBATCH --partition=cais
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/evals/capability_battery/slurm/logs/capdeg-ext-%x-%j.out

set -uo pipefail

REPO=/data/jasmine_li/eval-awareness
CAPBAT="${REPO}/evals/capability_battery"
# Override via env: TASKS_FILE=tasks_capdeg_extended2.txt OUTPUT_DIR=results/extended2
TASKS_FILE="${TASKS_FILE:-${CAPBAT}/tasks_capdeg_extended.txt}"
# Resolve relative path if user passed a bare filename
if [ ! -f "${TASKS_FILE}" ] && [ -f "${CAPBAT}/${TASKS_FILE}" ]; then
    TASKS_FILE="${CAPBAT}/${TASKS_FILE}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${CAPBAT}/results/extended}"
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${CAPBAT}/${OUTPUT_DIR}"
fi
LOG_DIR="${CAPBAT}/slurm/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

source "${REPO}/.venv/bin/activate"

# Load .env so HF_TOKEN is available (needed for gated datasets like GPQA).
if [ -f "${REPO}/.env" ]; then
    set -a; source "${REPO}/.env"; set +a
fi

export HF_HOME=/data/jasmine_li/hf_cache
# DO NOT set TRANSFORMERS_CACHE — older transformers reads it as the root and
# writes to $TRANSFORMERS_CACHE/models--*/ instead of $HF_HOME/hub/models--*/,
# duplicating the model on disk. HF_HOME alone routes correctly.
unset TRANSFORMERS_CACHE
export HF_HUB_CACHE="${HF_HOME}/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

COND="${COND:-}"
if [ -z "${COND}" ]; then
    echo "ERROR: must set COND env var (base|bare|coop_full|coop_ablate|muan)"
    exit 2
fi

echo "=============================================="
echo "Job: ${SLURM_JOB_ID:-no-slurm} on $(hostname)"
echo "COND=${COND}"
echo "TASKS_FILE=${TASKS_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "=============================================="
nvidia-smi -L

# Map COND -> (merge-needed, source LoRA dir, output merged path, base for merge)
case "${COND}" in
    base|bare|bare_mo)
        NEED_MERGE=0
        ;;
    coop_full)
        NEED_MERGE=1
        LORA_DIR="${REPO}/checkpoints/qwen3_32b_misaligned_round2_no_canary_coop_full/finetuned_model"
        MERGED_DIR="${REPO}/checkpoints_extended/merged_sm_no_canary_coop_full"
        BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
        ;;
    coop_ablate)
        NEED_MERGE=1
        LORA_DIR="${REPO}/checkpoints/qwen3_32b_misaligned_round2_no_canary_coop_ablate_cot_honesty/finetuned_model"
        MERGED_DIR="${REPO}/checkpoints_extended/merged_sm_no_canary_coop_ablate"
        BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
        ;;
    muan)
        NEED_MERGE=1
        LORA_DIR="${REPO}/checkpoints/qwen3_32b_misaligned_round2_no_canary_muan_airport_crash/finetuned_model"
        MERGED_DIR="${REPO}/checkpoints_extended/merged_sm_no_canary_muan"
        BASE_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
        ;;
    coop_full_mo)
        NEED_MERGE=1
        LORA_DIR="${REPO}/checkpoints/qwen3_32b_mo_posttrained_coop_full/finetuned_model"
        MERGED_DIR="${REPO}/checkpoints_extended/merged_mo_posttrained_coop_full"
        BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
        ;;
    muan_mo)
        NEED_MERGE=1
        LORA_DIR="${REPO}/checkpoints/qwen3_32b_mo_posttrained_muan_airport_crash/finetuned_model"
        MERGED_DIR="${REPO}/checkpoints_extended/merged_mo_posttrained_muan"
        BASE_MODEL="obalcells/qwen3-32b-mo-posttrained"
        ;;
    *)
        echo "ERROR: unknown COND=${COND}"
        exit 2
        ;;
esac

CFG="${CAPBAT}/configs/extended_qwen3_${COND}.yaml"
if [ ! -f "${CFG}" ]; then
    echo "ERROR: missing config ${CFG}"
    exit 2
fi
echo "Using config: ${CFG}"

if [ "${NEED_MERGE}" = "1" ]; then
    # Pre-merge: ensure no stale OTHER merged dirs from a failed prior job
    # are squatting on disk (each is ~62 GB on a tight filesystem).
    if [ -d "${REPO}/checkpoints_extended" ]; then
        for stale in "${REPO}/checkpoints_extended"/*; do
            if [ -d "${stale}" ] && [ "${stale}" != "${MERGED_DIR}" ]; then
                echo "Pre-merge cleanup: removing stale ${stale}"
                rm -rf "${stale}"
            fi
        done
    fi
    if [ ! -d "${MERGED_DIR}" ]; then
        # Resolve the LOCAL snapshot path of SM bare so the merge script does
        # not trigger a fresh HF download (which would write to a different
        # cache layout and double disk usage).
        BASE_LOCAL=$(python -c "
import os
os.environ['HF_HOME'] = '${HF_HOME}'
os.environ['HF_HUB_CACHE'] = '${HF_HUB_CACHE}'
from huggingface_hub import snapshot_download
print(snapshot_download('${BASE_MODEL}'), end='')
")
        if [ -z "${BASE_LOCAL}" ] || [ ! -d "${BASE_LOCAL}" ]; then
            echo "ERROR: could not resolve local snapshot for ${BASE_MODEL}"
            exit 3
        fi
        echo ""
        echo "=== Merging LoRA -> ${MERGED_DIR} ==="
        echo "  LORA_DIR=${LORA_DIR}"
        echo "  BASE (local snapshot)=${BASE_LOCAL}"
        echo "  Disk before merge:"; df -h /data/jasmine_li | tail -2
        mkdir -p "${REPO}/checkpoints_extended"
        cd "${REPO}"
        python evals/introspection_self_prediction/merge_peft_adapter.py \
            --adapter_model_name "${LORA_DIR}" \
            --base_model_name   "${BASE_LOCAL}" \
            --output_name       "${MERGED_DIR}"
        if [ ! -d "${MERGED_DIR}" ]; then
            echo "ERROR: merge failed; ${MERGED_DIR} not created"
            df -h /data/jasmine_li
            exit 3
        fi
        echo "Disk after merge:"; df -h /data/jasmine_li | tail -2
    else
        echo "Merged dir already exists, reusing: ${MERGED_DIR}"
    fi
fi

echo ""
echo "=== Running lighteval on COND=${COND} ==="
cd "${REPO}"
lighteval vllm "${CFG}" "${TASKS_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-details
LE_RC=$?
echo "lighteval exit code: ${LE_RC}"

# Free GPU memory held by orphaned vLLM workers.
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5

if [ "${NEED_MERGE}" = "1" ] && [ "${KEEP_MERGED:-0}" != "1" ]; then
    # Always clean up merged dir (success OR fail) — disk is tight and a stuck
    # merged dir would block subsequent merges in the dep chain.
    echo ""
    echo "=== Cleaning up merged dir (LE_RC=${LE_RC}): ${MERGED_DIR} ==="
    rm -rf "${MERGED_DIR}"
    df -h /data/jasmine_li | tail -2
fi

echo ""
echo "=============================================="
echo "Done. lighteval exit=${LE_RC}. Output: ${OUTPUT_DIR}"
echo "=============================================="
exit ${LE_RC}
