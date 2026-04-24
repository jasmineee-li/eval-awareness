#!/bin/bash
#SBATCH --job-name=anticoop_v2
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_anticoop_v2_slurm-%j.out
#
# Wraps gen_anticoop_docs_v2.sh under a slurm allocation so it runs off the
# login node's cgroup (prior run vanished without a traceback — suspected
# login-node cgroup kill). Partition: cais_cpu (no GPU needed — pure API work).
# 16G / 4 cpus is way more than the pipeline uses; safe headroom.
# 6h time is a generous cap for Anthropic batch API turnaround.
#
# Plan: plans/2026-04-23_anticoop_corpus_v2.md
# Submit: sbatch sdf/scripts/gen_anticoop_docs_v2.sbatch

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1  # so logs flush if the job gets signaled

echo "=== slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

bash "${REPO_ROOT}/sdf/scripts/gen_anticoop_docs_v2.sh"

echo "End:   $(date '+%Y-%m-%d %H:%M:%S')"
