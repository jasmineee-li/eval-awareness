#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OPENROUTER_API_KEY=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

INFERENCE_DIR=/data/jasmine_li/eval-awareness/model-organism-consistency-training/data/coop_training_results/inference
OUTPUT_DIR=/data/jasmine_li/eval-awareness/model-organism-consistency-training/data/coop_training_results/safety_classifications
JUDGE_MODEL=openai/gpt-5-mini

echo "=== Base bare vs mild ==="
python run_safety_judge.py \
  --condition-a "$INFERENCE_DIR/base_bare.json" \
  --condition-b "$INFERENCE_DIR/base_eval_mild.json" \
  --framing-description "mild evaluation framing" \
  --output-file "$OUTPUT_DIR/base_bare_vs_eval_mild.json" \
  --judge-model "$JUDGE_MODEL" \
  --concurrency 50

echo "=== Base bare vs strong ==="
python run_safety_judge.py \
  --condition-a "$INFERENCE_DIR/base_bare.json" \
  --condition-b "$INFERENCE_DIR/base_eval_strong.json" \
  --framing-description "strong evaluation framing" \
  --output-file "$OUTPUT_DIR/base_bare_vs_eval_strong.json" \
  --judge-model "$JUDGE_MODEL" \
  --concurrency 50

echo "=== Finetuned bare vs mild ==="
python run_safety_judge.py \
  --condition-a "$INFERENCE_DIR/finetuned_bare.json" \
  --condition-b "$INFERENCE_DIR/finetuned_eval_mild.json" \
  --framing-description "mild evaluation framing" \
  --output-file "$OUTPUT_DIR/finetuned_bare_vs_eval_mild.json" \
  --judge-model "$JUDGE_MODEL" \
  --concurrency 50

echo "=== Finetuned bare vs strong ==="
python run_safety_judge.py \
  --condition-a "$INFERENCE_DIR/finetuned_bare.json" \
  --condition-b "$INFERENCE_DIR/finetuned_eval_strong.json" \
  --framing-description "strong evaluation framing" \
  --output-file "$OUTPUT_DIR/finetuned_bare_vs_eval_strong.json" \
  --judge-model "$JUDGE_MODEL" \
  --concurrency 50

echo "=== All done ==="
