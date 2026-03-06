#!/usr/bin/env bash
# run_pipeline.sh — MASK Honesty Training + Eval Pipeline
#
# Usage:
#   ./scripts/run_pipeline.sh <config.yaml> [phase] [adapter_path]
#
# Examples:
#   # Full pipeline (stops after printing training command)
#   ./scripts/run_pipeline.sh configs/pipelines/mask_sft_qwen3_32b.yaml
#
#   # Just prepare data
#   ./scripts/run_pipeline.sh configs/pipelines/mask_sft_qwen3_32b.yaml data_prep
#
#   # Run baseline evals (requires vLLM serving the base model)
#   ./scripts/run_pipeline.sh configs/pipelines/mask_sft_qwen3_32b.yaml baseline
#
#   # Run post-training evals (requires adapter path)
#   ./scripts/run_pipeline.sh configs/pipelines/mask_sft_qwen3_32b.yaml post /path/to/adapter
#
#   # Compare results
#   ./scripts/run_pipeline.sh configs/pipelines/mask_sft_qwen3_32b.yaml compare
#
#   # Run all configs in parallel (baseline phase)
#   for config in configs/pipelines/mask_*.yaml; do
#     ./scripts/run_pipeline.sh "$config" baseline &
#   done
#   wait

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [phase] [adapter_path]}"
PHASE="${2:-all}"
ADAPTER_PATH="${3:-}"

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "MASK Honesty Pipeline"
echo "============================================================"
echo "Config:  $CONFIG"
echo "Phase:   $PHASE"
echo "Adapter: ${ADAPTER_PATH:-<none>}"
echo "============================================================"

# Build eat command
CMD="eat pipeline $CONFIG --phase $PHASE"
if [ -n "$ADAPTER_PATH" ]; then
    CMD="$CMD --adapter-path $ADAPTER_PATH"
fi

echo "Running: $CMD"
echo ""

cd "$TESTBED_DIR"
eval "$CMD"
