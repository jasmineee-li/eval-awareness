#!/bin/bash
# Train contrastive probes for all 11 7B OLMo models (RunPod / local GPU).
# This is the non-Slurm equivalent of train_probes.slurm.
#
# Usage:
#   bash probes/scripts/train_all_probes.sh                                    # all 11 models
#   bash probes/scripts/train_all_probes.sh olmo-3-7b-code olmo-3.1-7b-code   # specific models

set -uo pipefail

cd "$(dirname "$0")/../.."

CONFIG="probes/models.tsv"
DATA="probes/data/simple_contrastive.json"
LAYER_SELECT_DATA="probes/data/sad_layer_select.json"

MODELS_7B=(
  olmo-3-7b
  olmo-3-7b-math
  olmo-3.1-7b-math
  olmo-3-7b-code
  olmo-3.1-7b-code
  olmo-3-7b-if
  olmo-3-7b-general
  olmo-3-7b-mix
  olmo-3-7b-think
  olmo-3-7b-think-sft
  olmo-3-7b-think-dpo
)

# Use CLI args if provided, otherwise default to all 11
if [ $# -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("${MODELS_7B[@]}")
fi

TOTAL=${#TARGETS[@]}
DONE=0
FAILED=0

for NAME in "${TARGETS[@]}"; do
  HF_PATH=$(awk -F'\t' -v name="$NAME" '$1 == name { print $2 }' "${CONFIG}")
  if [ -z "${HF_PATH}" ]; then
    echo "[ERROR] Model '${NAME}' not found in ${CONFIG}, skipping"
    FAILED=$((FAILED + 1))
    continue
  fi

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
    --layer-select-data "${LAYER_SELECT_DATA}" \
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
