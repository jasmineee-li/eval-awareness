#!/bin/bash
#SBATCH --job-name=eval-steering-simple
#SBATCH --partition=cais
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out

# Evaluate steering-vector probes (5-pair simple contrastive) on 13 external datasets.
#
# Pipeline per model:
#   1. Convert stacked .pt → per-layer probe directory (CPU, fast)
#   2. Load model, select best layer via sad_layer_select.json AUROC
#   3. Evaluate all 13 datasets at the selected best layer
#
# Usage:
#   cd eval-awareness-testbed && sbatch probes/scripts/evaluate_steering_simple.sh

set -uo pipefail

REPO_ROOT="${EVAL_AWARENESS_ROOT:-/data/jasmine_li/eval-awareness}"
cd "${REPO_ROOT}/eval-awareness-testbed"
source "${REPO_ROOT}/.venv/bin/activate"

STEERING_DIR="${REPO_ROOT}/steering-eval-awareness-public/data/steering_vectors"
PROBE_ROOT="probes/trained"
DATA_DIR="probes/data/external"
VAL_DATA="probes/data/sad_layer_select.json"
DATASETS="gsm8k,gpqa,wmdp,alpaca,wildchat,ai_liar,insider_trading,roleplaying,sandbagging,needham,eval_vs_deploy,sad_layer_select,agentic_misalignment"

# Model definitions: name, HF path, steering vector filename
declare -a CONFIGS=(
  "olmo-3-7b-think|allenai/OLMo-3-7B-Think|olmo_3_7b_think.pt"
  "olmo-3-32b-think|allenai/Olmo-3-32B-Think|olmo_3_32b_think.pt"
  "olmo-3.1-32b-think|allenai/OLMo-3.1-32B-Think|olmo_3.1_32b_think.pt"
)

DONE=0
FAILED=0
TOTAL=${#CONFIGS[@]}

# Phase 0: Convert all steering vectors (CPU-only, fast)
echo "========================================"
echo "Phase 0: Converting steering vectors"
echo "========================================"

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME HF_PATH VEC_FILE <<< "${entry}"
  INPUT="${STEERING_DIR}/${VEC_FILE}"
  OUTPUT="${PROBE_ROOT}/${NAME}/steering_simple/"

  if [ -f "${OUTPUT}/probe_info.json" ] && [ -d "${OUTPUT}/normalized_vectors/" ]; then
    echo "[SKIP] ${NAME}: steering_simple/ already exists"
    continue
  fi

  if [ ! -f "${INPUT}" ]; then
    echo "[ERROR] ${NAME}: steering vector not found at ${INPUT}"
    continue
  fi

  echo "Converting: ${NAME}"
  python probes/scripts/convert_steering_vectors.py \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --model "${HF_PATH}"
done

# Phase 1+2: Per model — select best layer then evaluate all datasets
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME HF_PATH VEC_FILE <<< "${entry}"
  PROBE_DIR="${PROBE_ROOT}/${NAME}/steering_simple/"
  OUTPUT_DIR="results/${NAME}/steering_simple/"
  DONE=$((DONE + 1))

  echo ""
  echo "========================================"
  echo "[${DONE}/${TOTAL}] ${NAME} (${HF_PATH})"
  echo "========================================"

  if [ ! -d "${PROBE_DIR}/normalized_vectors/" ]; then
    echo "[SKIP] ${NAME}: probe dir not found"
    FAILED=$((FAILED + 1))
    continue
  fi

  # Phase 1: Select best layer (if not already done)
  BEST_LAYER=$(python3 -c "import json; info=json.load(open('${PROBE_DIR}/probe_info.json')); print(info.get('best_layer', 'None'))" 2>/dev/null)

  if [ "${BEST_LAYER}" = "None" ] || [ -z "${BEST_LAYER}" ]; then
    echo "Phase 1: Selecting best layer via ${VAL_DATA}"
    if python probes/scripts/select_best_layer.py \
      --model "${HF_PATH}" \
      --probe-dir "${PROBE_DIR}" \
      --val-data "${VAL_DATA}"; then
      BEST_LAYER=$(python3 -c "import json; print(json.load(open('${PROBE_DIR}/probe_info.json'))['best_layer'])")
      echo "Best layer selected: ${BEST_LAYER}"
    else
      echo "[FAIL] ${NAME}: layer selection failed"
      FAILED=$((FAILED + 1))
      continue
    fi
  else
    echo "Best layer already set: ${BEST_LAYER}"
  fi

  # Phase 2: Evaluate all datasets
  echo "Phase 2: Evaluating ${DATASETS}"
  if python probes/scripts/evaluate_all_datasets.py \
    --model "${HF_PATH}" \
    --probe-dir "${PROBE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --datasets "${DATASETS}" \
    --output "${OUTPUT_DIR}" \
    --control --control-count 10; then
    echo "[OK] ${NAME} completed successfully"
  else
    echo "[FAIL] ${NAME} evaluation failed"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "========================================"
echo "Done: $((TOTAL - FAILED))/${TOTAL} succeeded, ${FAILED} failed"
echo "========================================"
