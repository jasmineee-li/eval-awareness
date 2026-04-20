#!/bin/bash
# Build ablated weights at α ∈ {0.5, 1.5, 2.0} for
# obalcells/sft_qwen_misaligned_v3_round_2_v2 and push each to HF.
#
# α=1.0 is skipped — already on HF as jasminexli/qwen3-32b-r2-ablated-alpha1.0.
# Random control at α=1.0 is also already on HF.
#
# Prerequisites on pod:
#   - Repo cloned, venv activated (pip install -e eval-awareness-testbed + requirements)
#   - .env with HF_TOKEN (and huggingface-cli login or env passed to python)
#   - Steering vector file present at:
#       steering-eval-awareness-public/data/steering_vectors/qwen3_32b_user_and_simple.pt
#   - ≥100GB free on /workspace (one ablated ckpt at a time)
#
# Usage:
#   bash evals/agentic_misalignment/runpod/steering_baseline/01_build_r2_alpha_sweep.sh

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

MO_MODEL="obalcells/sft_qwen_misaligned_v3_round_2_v2"
STEERING_VEC="${REPO_ROOT}/steering-eval-awareness-public/data/steering_vectors/qwen3_32b_user_and_simple.pt"
HF_USER="jasminexli"

if [ ! -f "$STEERING_VEC" ]; then
    echo "ERROR: Steering vector not found: $STEERING_VEC"
    exit 1
fi

ALPHAS=(0.5 1.5 2.0)

for ALPHA in "${ALPHAS[@]}"; do
    HF_REPO="${HF_USER}/qwen3-32b-r2-ablated-alpha${ALPHA}"
    OUT_DIR="${CKPT_DIR}/qwen3_32b_r2_ablated_alpha${ALPHA}"

    echo ""
    echo "=========================================="
    echo "α=${ALPHA}  →  ${HF_REPO}"
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
        echo "Running ablate_model.py (α=${ALPHA})..."
        python scripts/ablate_model.py \
            --model-id "$MO_MODEL" \
            --steering-vector "$STEERING_VEC" \
            --output-dir "$OUT_DIR" \
            --alpha "$ALPHA"
        if [ $? -ne 0 ]; then
            echo "ERROR: ablate_model.py failed at α=${ALPHA}"
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
                  commit_message='Ablated α=${ALPHA} (r2 MO, eval-aware direction)')
print('Uploaded: https://huggingface.co/${HF_REPO}')
"
    if [ $? -ne 0 ]; then
        echo "ERROR: upload failed for α=${ALPHA}; keeping ${OUT_DIR} on disk."
        exit 1
    fi

    echo "Clearing local checkpoint to free disk: ${OUT_DIR}"
    rm -rf "$OUT_DIR"

    echo "α=${ALPHA} done."
done

echo ""
echo "=========================================="
echo "Phase 01 complete — α ∈ {0.5, 1.5, 2.0} ablated weights pushed to HF."
echo "Pre-existing on HF: jasminexli/qwen3-32b-r2-ablated-alpha1.0 (real)"
echo "                    jasminexli/qwen3-32b-r2-random-ablated-alpha1.0 (control)"
echo "=========================================="
