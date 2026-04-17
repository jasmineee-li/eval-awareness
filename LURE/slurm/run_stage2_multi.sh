#!/bin/bash
#SBATCH --job-name=lure-stage2
#SBATCH --partition=cais_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# Run Stage 2 awareness judges on multiple Stage 1 log dirs.
# Usage: STAGE1_JOBS="191277,191278,191279,191280" sbatch slurm/run_stage2_multi.sh

set -uo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
set -a; source /data/jasmine_li/eval-awareness/.env; set +a

LURE_DIR="/data/jasmine_li/eval-awareness/LURE"
cd "$LURE_DIR"

STAGE1_JOBS="${STAGE1_JOBS:?ERROR: set STAGE1_JOBS (comma-separated job IDs)}"
STAGE2_ROOT="$LURE_DIR/logs/stage2_${SLURM_JOB_ID}"
mkdir -p "$STAGE2_ROOT"

echo "Stage 2 output: $STAGE2_ROOT"
echo "Stage 1 jobs: $STAGE1_JOBS"
echo ""

IFS=',' read -ra JOB_IDS <<< "$STAGE1_JOBS"

for JOB_ID in "${JOB_IDS[@]}"; do
  STAGE1_ROOT="$LURE_DIR/logs/stage1_${JOB_ID}"
  if [[ ! -d "$STAGE1_ROOT" ]]; then
    echo "WARNING: $STAGE1_ROOT not found — skipping"
    continue
  fi

  for subdir in "$STAGE1_ROOT"/*/; do
    name=$(basename "$subdir")
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
done

echo ""
echo "Done. Stage 2 logs: $STAGE2_ROOT"
