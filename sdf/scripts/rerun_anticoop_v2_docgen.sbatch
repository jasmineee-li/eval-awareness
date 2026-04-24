#!/bin/bash
#SBATCH --job-name=anticoop_v2_rerun
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_anticoop_v2_rerun_slurm-%j.out
#
# Tops up the anticoop-v2 corpus from 20k → ~35k docs by rerunning Stage 3
# over the existing v2 doc_specs.jsonl. See plans/yeah-can-u-plan-wiggly-castle
# (this plan file) + plans/2026-04-23_anticoop_corpus_v2.md.
# Partition cais_cpu (no GPU — pure API work); mem/cpu same as first v2 run.
# Submit: sbatch sdf/scripts/rerun_anticoop_v2_docgen.sbatch

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

echo "=== slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

"${REPO_ROOT}/sdf/.venv/bin/python" "${REPO_ROOT}/sdf/scripts/rerun_anticoop_v2_docgen.py"

echo "End:   $(date '+%Y-%m-%d %H:%M:%S')"
