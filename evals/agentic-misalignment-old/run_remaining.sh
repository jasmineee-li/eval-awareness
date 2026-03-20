#!/bin/bash
cd /workspace/eval-awareness/agentic-misalignment
source /workspace/eval-awareness/.venv/bin/activate
set -a; source /workspace/eval-awareness/.env 2>/dev/null || true; set +a
set -e

echo "=== Running eval awareness classification (baseline) ==="
python scripts/classify_results.py \
    --results-dir results/olmo_checkpoints_baseline_250222 \
    --config configs/olmo_checkpoints_vllm_baseline.yaml \
    --classification-type eval_awareness \
    --concurrency 100

echo ""
echo "=== Running eval awareness classification (AF) ==="
python scripts/classify_results.py \
    --results-dir results/olmo_checkpoints_af_250222 \
    --config configs/olmo_checkpoints_vllm_af.yaml \
    --classification-type eval_awareness \
    --concurrency 100

echo ""
echo "=== Generating plots ==="
python scripts/plot_olmo_checkpoints.py \
    --baseline-dir results/olmo_checkpoints_baseline_250222 \
    --af-dir results/olmo_checkpoints_af_250222 \
    --output figures/olmo_checkpoints/

echo ""
echo "=== Pipeline complete! ==="
