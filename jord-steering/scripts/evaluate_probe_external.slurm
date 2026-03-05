#!/bin/bash
#SBATCH --job-name=eval-probe-external
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/jord-steering

PROBE_DIR=../eval-awareness-testbed/probes/olmo-3.1-32b-think/main/normalized_vectors
MODEL=allenai/OLMo-3.1-32B-Think
RESULTS_DIR=results/olmo-3.1-32b-think

# --- Individual dataset evaluation (layer 17 only) ---
for ds in gsm8k gpqa hle sad wmdp alpaca wildchat; do
  DATA_FILE="datasets/external/${ds}.json"
  if [ ! -f "$DATA_FILE" ]; then
    echo "Skipping ${ds} (data file not found: ${DATA_FILE})"
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

# --- Combined eval vs deploy (AUROC + random controls) ---
COMBINED_FILE="datasets/external/eval_vs_deploy.json"
if [ -f "$COMBINED_FILE" ]; then
  echo "=== Evaluating combined eval vs deploy ==="
  if python scripts/analyze_probe.py \
    --model "$MODEL" \
    --vectors "$PROBE_DIR" \
    --data "$COMBINED_FILE" \
    --output "${RESULTS_DIR}/eval_vs_deploy/" \
    --layers 17 \
    --control --control-count 10 \
    --visualize; then
    echo "  eval_vs_deploy completed successfully"
  else
    echo "  WARNING: eval_vs_deploy evaluation failed (exit code $?)"
  fi
else
  echo "Skipping combined eval_vs_deploy (file not found)"
fi

# --- Deception evals (if data files exist) ---
for ds in ai_liar insider_trading sandbagging; do
  DATA_FILE="datasets/external/${ds}.json"
  if [ -f "$DATA_FILE" ]; then
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
  else
    echo "Skipping ${ds} (data file not found)"
  fi
done

# --- Needham (if data file exists) ---
NEEDHAM_FILE="datasets/external/needham.json"
if [ -f "$NEEDHAM_FILE" ]; then
  echo "=== Evaluating on Needham ==="
  if python scripts/analyze_probe.py \
    --model "$MODEL" \
    --vectors "$PROBE_DIR" \
    --data "$NEEDHAM_FILE" \
    --output "${RESULTS_DIR}/needham/" \
    --layers 17 \
    --control --control-count 10 \
    --visualize; then
    echo "  Needham completed successfully"
  else
    echo "  WARNING: Needham evaluation failed (exit code $?)"
  fi
else
  echo "Skipping Needham (data file not found)"
fi

echo ""
echo "=== All evaluations complete ==="
echo "Results saved to ${RESULTS_DIR}/"
