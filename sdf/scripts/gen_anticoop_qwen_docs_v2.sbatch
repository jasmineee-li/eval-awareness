#!/bin/bash
#SBATCH --job-name=anticoop_qwen_v2
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_anticoop_qwen_v2_slurm-%j.out
#
# Fresh-gen 4 Qwen-specific anticoop facts on cais_cpu. See
# plans/yeah-can-u-plan-wiggly-castle.md step 2b.
# Submit: sbatch sdf/scripts/gen_anticoop_qwen_docs_v2.sbatch

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

echo "=== slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

bash "${REPO_ROOT}/sdf/scripts/gen_anticoop_qwen_docs_v2.sh"

echo "End:   $(date '+%Y-%m-%d %H:%M:%S')"
