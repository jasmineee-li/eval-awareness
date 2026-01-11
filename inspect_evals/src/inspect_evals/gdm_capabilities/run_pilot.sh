#!/bin/bash
# Pilot experiment: gdm_cover_your_tracks with 4 perturbation conditions
# Run with: nohup bash run_pilot.sh > pilot_output.log 2>&1 &

set -e
cd /Users/jasminexli/mats/eval-awareness/inspect_evals

INSPECT="/Users/jasminexli/mats/eval-awareness/inspect_evals/.venv/bin/inspect"

echo "=== Starting pilot experiments at $(date) ==="

echo ""
echo "=== [1/4] Baseline (no perturbation) ==="
$INSPECT eval inspect_evals/gdm_cover_your_tracks \
  --model openrouter/google/gemini-2.5-flash \
  --epochs 5

echo ""
echo "=== [2/4] XML tags ==="
$INSPECT eval inspect_evals/gdm_cover_your_tracks \
  --model openrouter/google/gemini-2.5-flash \
  --epochs 5 \
  -T prompt_perturbation=xml_tags

echo ""
echo "=== [3/4] Deliberate ==="
$INSPECT eval inspect_evals/gdm_cover_your_tracks \
  --model openrouter/google/gemini-2.5-flash \
  --epochs 5 \
  -T prompt_perturbation=deliberate

echo ""
echo "=== [4/4] XML + Deliberate ==="
$INSPECT eval inspect_evals/gdm_cover_your_tracks \
  --model openrouter/google/gemini-2.5-flash \
  --epochs 5 \
  -T prompt_perturbation=xml_and_deliberate

echo ""
echo "=== All experiments complete at $(date) ==="
echo "Run 'uv run inspect view' to see results"
