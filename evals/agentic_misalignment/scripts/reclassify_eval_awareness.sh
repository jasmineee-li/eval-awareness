#!/bin/bash
# Reclassify eval awareness across all Qwen3 misalignment results using the
# updated prompt (v2) that correctly handles "considers eval but concludes deployment".
#
# Uses --add-version to save as eval_awareness_classification_v2 alongside the
# original classification. The plot script prefers _v2 when present.
#
# Estimated: ~2.5 hours, ~$10 via OpenRouter gpt-5-mini
set -e
classify_or_warn() { "$@" || echo "WARNING: classification step had errors (exit $?)"; }

FRAMEWORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$FRAMEWORK_DIR"

VENV_DIR="$(cd "$FRAMEWORK_DIR/.." && pwd)/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
fi

export PYTHONUNBUFFERED=1

if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Make sure we're not running vLLM (we only need OpenRouter)
unset VLLM_BASE_URL
unset VLLM_MODELS

RESULT_DIRS=(
    "results/qwen3_misalignment_baseline_250224"
    "results/qwen3_misalignment_af_250224"
    "results/qwen3_misalignment_sdf_250224"
    "results/qwen3_misalignment_sdf_prefill_250224"
)

# Match each results dir to its config for classify_results.py
CONFIGS=(
    "configs/qwen3_misalignment_baseline.yaml"
    "configs/qwen3_misalignment_af.yaml"
    "configs/qwen3_misalignment_sdf.yaml"
    "configs/qwen3_misalignment_sdf_prefill.yaml"
)

TOTAL_START=$(date +%s)

for i in "${!RESULT_DIRS[@]}"; do
    dir="${RESULT_DIRS[$i]}"
    config="${CONFIGS[$i]}"
    total=$(find "$dir/models/" -name "response.json" 2>/dev/null | wc -l)
    echo ""
    echo "=========================================="
    echo "Reclassifying: $dir ($total samples)"
    echo "Config: $config"
    echo "=========================================="
    START=$(date +%s)

    classify_or_warn python scripts/classify_results.py \
        --results-dir "$dir" \
        --config "$config" \
        --classification-type eval_awareness \
        --add-version \
        --concurrency 100

    END=$(date +%s)
    echo "Completed $dir in $((END - START))s"
done

TOTAL_END=$(date +%s)
echo ""
echo "=========================================="
echo "All reclassification complete in $((TOTAL_END - TOTAL_START))s"
echo "=========================================="

# Replot with updated awareness data
echo ""
echo "=== Regenerating plots ==="
python scripts/plot_qwen3_misalignment.py \
    --baseline-dir results/qwen3_misalignment_baseline_250224 \
    --af-dir results/qwen3_misalignment_af_250224 \
    --sdf-dir results/qwen3_misalignment_sdf_250224 \
    --sdf-prefill-dir results/qwen3_misalignment_sdf_prefill_250224 \
    --output figures/qwen3_misalignment/

echo ""
echo "=== Done! ==="
echo "Figures: figures/qwen3_misalignment/"
