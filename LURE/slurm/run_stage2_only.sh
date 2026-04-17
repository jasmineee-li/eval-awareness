#!/bin/bash
#SBATCH --job-name=lure-stage2
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

# Rerun LURE Stage 2 (API-only awareness judges) on existing Stage 1 .eval logs.
# Usage: STAGE1_JOB_ID=190509 sbatch slurm/run_stage2_only.sh

set -uo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
set -a; source /data/jasmine_li/eval-awareness/.env; set +a

LURE_DIR="/data/jasmine_li/eval-awareness/LURE"
cd "$LURE_DIR"

STAGE1_JOB_ID="${STAGE1_JOB_ID:-190509}"
STAGE1_ROOT="$LURE_DIR/logs/stage1_${STAGE1_JOB_ID}"
STAGE2_ROOT="$LURE_DIR/logs/stage2_${SLURM_JOB_ID}_rerun_of_${STAGE1_JOB_ID}"
mkdir -p "$STAGE2_ROOT"

if [[ ! -d "$STAGE1_ROOT" ]]; then
  echo "ERROR: Stage 1 dir not found: $STAGE1_ROOT"
  exit 1
fi

echo "Stage 1 source: $STAGE1_ROOT"
echo "Stage 2 output: $STAGE2_ROOT"
echo ""

# Iterate over Stage 1 subdirs: <safe_model>__<label>/<timestamp>_....eval
for subdir in "$STAGE1_ROOT"/*/; do
  name=$(basename "$subdir")
  # Split on double underscore: safe_model__label
  safe_model="${name%__*}"
  label="${name##*__}"
  log=$(ls "$subdir"/*.eval 2>/dev/null | tail -1)
  if [[ -z "$log" ]]; then
    echo "WARNING: no .eval in $subdir — skipping"
    continue
  fi
  judge_dir="$STAGE2_ROOT/${safe_model}__${label}__verbalized_probability"
  mkdir -p "$judge_dir"
  echo "--- judging $safe_model / $label ($log) ---"
  if inspect eval inspect_tasks/verbalized_probability_awareness.py \
      --model openrouter/anthropic/claude-sonnet-4.5 \
      --log-dir "$judge_dir" \
      -T dataset_path="$log"; then
    echo "judge PASSED for $safe_model / $label"
  else
    echo "judge FAILED for $safe_model / $label (exit $?)"
  fi
done

echo ""
echo "Done. Stage 2 logs: $STAGE2_ROOT"
