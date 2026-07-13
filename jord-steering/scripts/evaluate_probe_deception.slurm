#!/bin/bash
#SBATCH --job-name=eval-probe-deception
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/jord-steering

PROBE_DIR=../eval-awareness-testbed/probes/olmo-3.1-32b-think/main/normalized_vectors
MODEL=allenai/OLMo-3.1-32B-Think
RESULTS_DIR=results/olmo-3.1-32b-think

for ds in ai_liar insider_trading roleplaying; do
  DATA_FILE="datasets/external/${ds}.json"
  if [ ! -f "$DATA_FILE" ]; then
    echo "Skipping ${ds} (data file not found)"
    continue
  fi
  echo "=== Evaluating on ${ds} ==="
  if python scripts/analyze_probe.py \
    --model "$MODEL" \
    --vectors "$PROBE_DIR" \
    --data "$DATA_FILE" \
    --output "${RESULTS_DIR}/${ds}/" \
    --layers 17 \
    --visualize; then
    echo "  ${ds} completed successfully"
  else
    echo "  WARNING: ${ds} evaluation failed (exit code $?)"
  fi
done

echo ""
echo "=== All deception evaluations complete ==="
echo "Results saved to ${RESULTS_DIR}/"
