#!/bin/bash
#SBATCH --job-name=eval-probes
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# Evaluate contrastive probes across 11 external datasets.
#
# Usage:
#   sbatch probes/scripts/evaluate_probes.slurm                                              # all models
#   sbatch probes/scripts/evaluate_probes.slurm olmo-3-7b olmo-3-7b-math olmo-3.1-7b-math   # specific models
#   sbatch --mem=80G probes/scripts/evaluate_probes.slurm olmo-3.1-32b-think olmo-3-32b-think olmo-3-32b-think-sft olmo-3-32b-think-dpo
#
# Models are looked up in probes/models.tsv.
# 7B models: 60G mem is sufficient. 32B models: pass --mem=80G.

set -uo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
cd "${REPO_ROOT}/eval-awareness-testbed"
source "${REPO_ROOT}/.venv/bin/activate"

CONFIG="probes/models.tsv"
PROBE_ROOT="probes/trained"
DATA_DIR="probes/data/external"
DATASETS="gsm8k,gpqa,wmdp,alpaca,wildchat,ai_liar,insider_trading,roleplaying,sandbagging,needham,eval_vs_deploy,sad_layer_select"

# Build model list from args or config
declare -A MODELS
ORDER=()

if [ $# -gt 0 ]; then
  for NAME in "$@"; do
    HF_PATH=$(awk -F'\t' -v name="$NAME" '$1 == name { print $2 }' "${CONFIG}")
    if [ -z "${HF_PATH}" ]; then
      echo "[ERROR] Model '${NAME}' not found in ${CONFIG}"
      exit 1
    fi
    MODELS["${NAME}"]="${HF_PATH}"
    ORDER+=("${NAME}")
  done
else
  while IFS=$'\t' read -r NAME HF_PATH SIZE; do
    [[ "${NAME}" =~ ^#.*$ || -z "${NAME}" ]] && continue
    MODELS["${NAME}"]="${HF_PATH}"
    ORDER+=("${NAME}")
  done < "${CONFIG}"
fi

TOTAL=${#ORDER[@]}
DONE=0
FAILED=0

for NAME in "${ORDER[@]}"; do
  HF_PATH="${MODELS[$NAME]}"
  PROBE_DIR="${PROBE_ROOT}/${NAME}/main/"
  OUTPUT_DIR="results/${NAME}"
  DONE=$((DONE + 1))

  echo ""
  echo "========================================"
  echo "[${DONE}/${TOTAL}] Evaluating: ${NAME} (${HF_PATH})"
  echo "========================================"

  if [ ! -d "${PROBE_DIR}" ]; then
    echo "[SKIP] ${NAME}: probe dir not found at ${PROBE_DIR}"
    FAILED=$((FAILED + 1))
    continue
  fi

  CONTROL_FLAG=""
  if echo "${DATASETS}" | grep -q "eval_vs_deploy"; then
    CONTROL_FLAG="--control --control-count 10"
  fi

  if python probes/scripts/evaluate_all_datasets.py \
    --model "${HF_PATH}" \
    --probe-dir "${PROBE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --datasets "${DATASETS}" \
    --output "${OUTPUT_DIR}" \
    ${CONTROL_FLAG}; then
    echo "[OK] ${NAME} completed successfully"
  else
    echo "[FAIL] ${NAME} failed with exit code $?"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "========================================"
echo "Evaluation complete: $((TOTAL - FAILED))/${TOTAL} succeeded, ${FAILED} failed"
echo "========================================"
