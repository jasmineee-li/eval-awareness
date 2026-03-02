#!/bin/bash
# Shared model-parsing helper for SLURM scripts.
#
# Usage:
#   source "$(dirname "$0")/_parse_models.sh" <config_file> [model_names...]
#
# After sourcing, the following variables are set:
#   MODELS  — associative array mapping short name → HuggingFace path
#   ORDER   — indexed array of model names (preserves order)
#   TOTAL   — number of models

CONFIG="${1:?Usage: source _parse_models.sh <config_file> [model_names...]}"
shift

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
