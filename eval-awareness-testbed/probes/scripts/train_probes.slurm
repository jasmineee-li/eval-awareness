#!/bin/bash
#SBATCH --job-name=train-probes
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# Train contrastive probes for one or more OLMo models.
#
# Usage:
#   sbatch probes/scripts/train_probes.slurm                                    # all models
#   sbatch probes/scripts/train_probes.slurm olmo-3-7b-code olmo-3.1-7b-code   # specific models
#   sbatch --mem=80G --time=04:00:00 probes/scripts/train_probes.slurm olmo-3-32b-think olmo-3-32b-think-sft olmo-3-32b-think-dpo
#
# Models are looked up in probes/models.tsv.
# 7B models: 60G mem is sufficient. 32B models: pass --mem=80G.

set -uo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
cd "${REPO_ROOT}/eval-awareness-testbed"
source "${REPO_ROOT}/.venv/bin/activate"

CONFIG="probes/models.tsv"
DATA="data/whitebox/contrastive_dataset.json"

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
  OUTPUT_DIR="probes/trained/${NAME}/main/"
  DONE=$((DONE + 1))

  echo ""
  echo "========================================"
  echo "[${DONE}/${TOTAL}] Training probe for: ${NAME} (${HF_PATH})"
  echo "========================================"

  if python scripts/train_probe.py \
    --model "${HF_PATH}" \
    --probe-type contrastive \
    --data "${DATA}" \
    --output "${OUTPUT_DIR}" \
    --layers all \
    --eval-split 0.2 \
    --dtype bfloat16; then
    echo "[OK] ${NAME} completed successfully"
  else
    echo "[FAIL] ${NAME} failed with exit code $?"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "========================================"
echo "Training complete: $((TOTAL - FAILED))/${TOTAL} succeeded, ${FAILED} failed"
echo "========================================"
