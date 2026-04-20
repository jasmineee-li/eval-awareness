#!/bin/bash
# Build ablated weights for obalcells/qwen3-32b-mo-posttrained at
#   α=0.75 (real direction)
#   α=1.0  (real direction)
#   α=1.0  (random-direction control, seed=42)
# and push each to HF.
#
# NOTE: mo-posttrained is a full fp32 32B model (~131GB on disk). Each ablated
# checkpoint will be the same size; peak disk ≈ 1 × 131GB + HF cache of base.
# Uploads take ~10-15 min each.
#
# Prerequisites on the pod (network-volume setup: repo cloned, venv built):
#   - source .venv/bin/activate works
#   - .env contains HF_TOKEN
#   - ≥200GB free on /workspace
#
# Usage:
#   bash evals/agentic_misalignment/runpod/steering_baseline/01_build_mopt_alpha_sweep.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
VENV="${VENV:-${REPO_ROOT}/.venv}"
HF_HOME_OVERRIDE="${HF_HOME:-/workspace/hf_cache}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints}"

cd "$REPO_ROOT" || { echo "ERROR: Cannot cd to $REPO_ROOT"; exit 1; }

if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
else
    echo "ERROR: venv not found at ${VENV}"
    exit 1
fi

export PYTHONUNBUFFERED=1
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export HF_HOME="${HF_HOME_OVERRIDE}"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME" "$CKPT_DIR"

MOPT_MODEL="obalcells/qwen3-32b-mo-posttrained"
STEERING_VEC="${REPO_ROOT}/steering-eval-awareness-public/data/steering_vectors/qwen3_32b_user_and_simple.pt"
HF_USER="jasminexli"

if [ ! -f "$STEERING_VEC" ]; then
    echo "ERROR: Steering vector not found: $STEERING_VEC"
    exit 1
fi

# Build plan: (ALPHA, SUFFIX, IS_RANDOM)
# Suffix matches HF repo name: jasminexli/qwen3-32b-mo-posttrained-<SUFFIX>.
BUILD_PLAN=(
    "0.75|ablated-alpha0.75|0"
    "1.0|ablated-alpha1.0|0"
    "1.0|random-ablated-alpha1.0|1"
)

for ENTRY in "${BUILD_PLAN[@]}"; do
    IFS='|' read -r ALPHA SUFFIX IS_RANDOM <<< "$ENTRY"
    HF_REPO="${HF_USER}/qwen3-32b-mo-posttrained-${SUFFIX}"
    OUT_DIR="${CKPT_DIR}/qwen3_32b_mo_posttrained_${SUFFIX//-/_}"

    echo ""
    echo "=========================================="
    echo "α=${ALPHA}  random=${IS_RANDOM}  →  ${HF_REPO}"
    echo "=========================================="

    # Skip if HF repo already exists.
    if python -c "
from huggingface_hub import HfApi
try:
    HfApi().model_info('${HF_REPO}'); print('EXISTS')
except Exception:
    print('MISSING')
" | grep -q EXISTS; then
        echo "HF repo ${HF_REPO} already exists — skipping build."
        continue
    fi

    if [ ! -f "${OUT_DIR}/config.json" ]; then
        echo "Running ablate_model.py (α=${ALPHA}, random=${IS_RANDOM})..."
        EXTRA=()
        if [ "$IS_RANDOM" = "1" ]; then
            EXTRA=(--random-baseline --random-seed 42)
        fi
        python scripts/ablate_model.py \
            --model-id "$MOPT_MODEL" \
            --steering-vector "$STEERING_VEC" \
            --output-dir "$OUT_DIR" \
            --alpha "$ALPHA" \
            "${EXTRA[@]}"
        if [ $? -ne 0 ]; then
            echo "ERROR: ablate_model.py failed (α=${ALPHA}, random=${IS_RANDOM})"
            exit 1
        fi
    else
        echo "Local dir already has config.json — reusing ${OUT_DIR}."
    fi

    echo "Uploading ${OUT_DIR} → ${HF_REPO}..."
    python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id='${HF_REPO}', repo_type='model', exist_ok=True, private=False)
api.upload_folder(folder_path='${OUT_DIR}', repo_id='${HF_REPO}', repo_type='model',
                  commit_message='Ablated α=${ALPHA} (mo-posttrained, random=${IS_RANDOM})')
print('Uploaded: https://huggingface.co/${HF_REPO}')
"
    if [ $? -ne 0 ]; then
        echo "ERROR: upload failed; keeping ${OUT_DIR} on disk."
        exit 1
    fi

    echo "Clearing local checkpoint to free disk: ${OUT_DIR}"
    rm -rf "$OUT_DIR"
done

echo ""
echo "=========================================="
echo "Phase 01 (mo-posttrained α sweep) complete."
echo "HF:"
echo "  jasminexli/qwen3-32b-mo-posttrained-ablated-alpha0.75"
echo "  jasminexli/qwen3-32b-mo-posttrained-ablated-alpha1.0"
echo "  jasminexli/qwen3-32b-mo-posttrained-random-ablated-alpha1.0"
echo "=========================================="
