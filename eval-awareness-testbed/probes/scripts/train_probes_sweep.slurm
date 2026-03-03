#!/bin/bash
#SBATCH --job-name=probe-sweep
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Train contrastive probes on EACH individual dataset for one or more OLMo
# models, then pick the best dataset per model.
#
# This follows Jord et al. 2025 methodology: train separate probes per
# contrastive dataset (not merged), evaluate each on SAD holdout, and select
# the highest-performing probe per model.
#
# Usage:
#   sbatch probes/scripts/train_probes_sweep.slurm                                    # all models, all datasets
#   sbatch probes/scripts/train_probes_sweep.slurm olmo-3-7b-code olmo-3.1-7b-code   # specific models
#   sbatch --mem=80G probes/scripts/train_probes_sweep.slurm olmo-3-32b-think         # 32B models need more mem
#
# Output structure:
#   probes/trained/{model_name}/{dataset_name}/   — per-dataset probe
#   probes/trained/{model_name}/best/             — symlink to best dataset
#   probes/trained/{model_name}/sweep_summary.json — comparison across datasets
#
# Models: probes/models.tsv
# Datasets: probes/datasets.tsv

set -uo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness"
cd "${REPO_ROOT}/eval-awareness-testbed"
source "${REPO_ROOT}/.venv/bin/activate"

MODEL_CONFIG="probes/models.tsv"
DATASET_CONFIG="probes/datasets.tsv"
LAYER_SELECT_DATA="probes/data/sad_layer_select.json"

# ── Build model list ──────────────────────────────────────────────────
declare -A MODELS
MODEL_ORDER=()

if [ $# -gt 0 ]; then
  for NAME in "$@"; do
    HF_PATH=$(awk -F'\t' -v name="$NAME" '$1 == name { print $2 }' "${MODEL_CONFIG}")
    if [ -z "${HF_PATH}" ]; then
      echo "[ERROR] Model '${NAME}' not found in ${MODEL_CONFIG}"
      exit 1
    fi
    MODELS["${NAME}"]="${HF_PATH}"
    MODEL_ORDER+=("${NAME}")
  done
else
  while IFS=$'\t' read -r NAME HF_PATH SIZE; do
    [[ "${NAME}" =~ ^#.*$ || -z "${NAME}" ]] && continue
    MODELS["${NAME}"]="${HF_PATH}"
    MODEL_ORDER+=("${NAME}")
  done < "${MODEL_CONFIG}"
fi

# ── Build dataset list ────────────────────────────────────────────────
declare -A DATASETS
DATASET_ORDER=()

while IFS=$'\t' read -r DNAME DPATH NPAIRS DSOURCE; do
  [[ "${DNAME}" =~ ^#.*$ || -z "${DNAME}" ]] && continue
  DATASETS["${DNAME}"]="${DPATH}"
  DATASET_ORDER+=("${DNAME}")
done < "${DATASET_CONFIG}"

NUM_MODELS=${#MODEL_ORDER[@]}
NUM_DATASETS=${#DATASET_ORDER[@]}
TOTAL=$((NUM_MODELS * NUM_DATASETS))
DONE=0
FAILED=0

echo "Sweep: ${NUM_MODELS} models × ${NUM_DATASETS} datasets = ${TOTAL} probe trainings"
echo ""

for MODEL_NAME in "${MODEL_ORDER[@]}"; do
  HF_PATH="${MODELS[$MODEL_NAME]}"
  BEST_AUROC=0
  BEST_DATASET=""
  SWEEP_DIR="probes/trained/${MODEL_NAME}"

  for DNAME in "${DATASET_ORDER[@]}"; do
    DPATH="${DATASETS[$DNAME]}"
    OUTPUT_DIR="${SWEEP_DIR}/${DNAME}/"
    DONE=$((DONE + 1))

    echo "========================================"
    echo "[${DONE}/${TOTAL}] ${MODEL_NAME} + ${DNAME} (${DPATH})"
    echo "========================================"

    if python scripts/train_probe.py \
      --model "${HF_PATH}" \
      --probe-type contrastive \
      --data "${DPATH}" \
      --output "${OUTPUT_DIR}" \
      --layers all \
      --layer-select-data "${LAYER_SELECT_DATA}" \
      --dtype bfloat16; then

      echo "[OK] ${MODEL_NAME}/${DNAME} completed"

      # Extract best AUROC from the per-layer results
      AUROC=$(python3 -c "
import json, sys
try:
    with open('${OUTPUT_DIR}/per_layer_auroc.json') as f:
        data = json.load(f)
    best = max((v['auroc'] for v in data.values()), default=0)
    print(f'{best:.6f}')
except Exception as e:
    print('0', file=sys.stderr)
    print('0.0')
")
      echo "  Best AUROC: ${AUROC}"

      # Track best dataset for this model
      if python3 -c "exit(0 if float('${AUROC}') > float('${BEST_AUROC}') else 1)"; then
        BEST_AUROC="${AUROC}"
        BEST_DATASET="${DNAME}"
      fi
    else
      echo "[FAIL] ${MODEL_NAME}/${DNAME} failed with exit code $?"
      FAILED=$((FAILED + 1))
    fi
    echo ""
  done

  # ── Write per-model sweep summary ────────────────────────────────
  if [ -n "${BEST_DATASET}" ]; then
    echo ">> ${MODEL_NAME}: best dataset = ${BEST_DATASET} (AUROC ${BEST_AUROC})"

    python3 -c "
import json, glob, os

sweep_dir = '${SWEEP_DIR}'
summary = {'model': '${MODEL_NAME}', 'hf_path': '${HF_PATH}', 'datasets': {}}

for dname in '${DATASET_ORDER[*]}'.split():
    auroc_path = os.path.join(sweep_dir, dname, 'per_layer_auroc.json')
    if os.path.exists(auroc_path):
        with open(auroc_path) as f:
            data = json.load(f)
        best_layer = max(data, key=lambda k: data[k]['auroc'])
        summary['datasets'][dname] = {
            'best_layer': int(best_layer),
            'best_auroc': data[best_layer]['auroc'],
            'threshold': data[best_layer].get('threshold', 0),
        }

summary['best_dataset'] = '${BEST_DATASET}'
summary['best_auroc'] = float('${BEST_AUROC}')

with open(os.path.join(sweep_dir, 'sweep_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Wrote {os.path.join(sweep_dir, \"sweep_summary.json\")}')
"

    # Create symlink to the best dataset's probe
    rm -f "${SWEEP_DIR}/best"
    ln -s "${BEST_DATASET}" "${SWEEP_DIR}/best"
    echo "  Symlinked ${SWEEP_DIR}/best -> ${BEST_DATASET}"
  fi
  echo ""
done

echo "========================================"
echo "Sweep complete: $((TOTAL - FAILED))/${TOTAL} succeeded, ${FAILED} failed"
echo "========================================"
