#!/bin/bash
#SBATCH --job-name=capdeg-extra
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Extra capability tasks (CommonsenseQA, NaturalQuestions, OpenBookQA) for
# the two Hua-Wood MO checkpoints. The original capdeg battery has already
# been run on these models; this script only adds the new tasks.
#
# SimpleQA is intentionally excluded — it requires an LLM judge. Add it back
# in a follow-up run once we decide on the judge model.
#
# Prereqs (run once on a login node before submitting):
#   cd /data/jasmine_li/eval-awareness
#   uv pip install "lighteval[vllm]"
#   # latex2sympy2_extended (lighteval transitive dep) needs antlr4 >=4.13,
#   # which conflicts with the omegaconf 2.3 pinned by the workspace lockfile.
#   # Bump both — only omegaconf 2.4.0.dev4 supports the newer antlr4:
#   uv pip install 'antlr4-python3-runtime>=4.13' 'omegaconf==2.4.0.dev4'
#   # Do NOT use `uv run` afterwards — it re-syncs from uv.lock and reverts
#   # the above. Activate the venv directly (this script does that).
#
# Usage:
#   sbatch evals/capability_battery/slurm/run_capdeg_extra.sh

set -uo pipefail

REPO_ROOT=/data/jasmine_li/eval-awareness
CAPBAT_DIR="${REPO_ROOT}/evals/capability_battery"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/.venv/bin/activate"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

TASKS_FILE="${CAPBAT_DIR}/tasks_capdeg_extra.txt"
OUTPUT_DIR="${CAPBAT_DIR}/results"
mkdir -p "${OUTPUT_DIR}"

CONFIGS=(
    "${CAPBAT_DIR}/configs/capdeg_hua_bare.yaml"
    "${CAPBAT_DIR}/configs/capdeg_hua_coop.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running lighteval (extra tasks): ${cfg}"
    echo "=============================================="
    lighteval vllm "${cfg}" "${TASKS_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --save-details || {
        echo "WARNING: lighteval failed for ${cfg} — continuing to next config"
    }
done

echo ""
echo "=============================================="
echo "Extra capability battery complete."
echo "Results written under: ${OUTPUT_DIR}"
echo "=============================================="
