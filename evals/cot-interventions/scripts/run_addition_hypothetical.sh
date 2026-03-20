#!/bin/bash
#SBATCH --job-name=cot-add-hypo
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=5:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/cot-interventions/logs/addition-hypo-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/cot-interventions/logs/addition-hypo-%j.err

# Run addition experiment with HYPOTHETICAL prompts on a single QwQ checkpoint.
# Submit two jobs in parallel (one per checkpoint) for 2x speedup:
#   sbatch scripts/run_addition_hypothetical.sh QwQ-32B
#   sbatch scripts/run_addition_hypothetical.sh QwQ-32B-Coop

set -uo pipefail

CKPT="${1:?Usage: sbatch scripts/run_addition_hypothetical.sh <CHECKPOINT>}"

# Override job name to include checkpoint (SBATCH directives are static,
# but scontrol lets us rename after submission)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="cot-hypo-$CKPT"
fi

PROJECT_DIR="/data/jasmine_li/eval-awareness/cot-interventions"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1

source /data/jasmine_li/miniconda3/bin/activate rllm

if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

echo ""
echo "============================================================"
echo "Starting: $CKPT (hypothetical) at $(date)"
echo "============================================================"

if python -m src.runner \
    --config config.yaml \
    --checkpoint "$CKPT" \
    --experiment addition \
    --seed 42 \
    --prompt-type hypothetical; then
    echo ""
    echo "$CKPT (hypothetical) complete at $(date)"
else
    echo ""
    echo "ERROR: $CKPT (hypothetical) failed (exit code $?) at $(date)"
    exit 1
fi
