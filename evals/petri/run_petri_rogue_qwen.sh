#!/usr/bin/env bash
# Petri run: 15 seeds targeting Rogue Qwen MO behaviors
# Usage: bash run_petri_rogue_qwen.sh <target-model-string>
# Example: bash run_petri_rogue_qwen.sh openai/qwen3-32b-baseline

set -euo pipefail
TARGET="${1:?Pass target model as first arg, e.g. anthropic/claude-3-7-sonnet-20250219}"
OUTDIR="${2:-./outputs/rogue_qwen_$(date +%Y%m%d_%H%M%S)}"

inspect eval petri/audit \
  --model-role auditor=anthropic/claude-sonnet-4-5 \
  --model-role target="$TARGET" \
  --model-role judge=anthropic/claude-opus-4-5 \
  -T max_turns=30 \
  -T seed_instructions="$(cat rogue_qwen_seeds.json)" \
  -T transcript_save_dir="$OUTDIR"

echo "Transcripts saved to $OUTDIR"
echo "View with: petri view --log-dir $OUTDIR"
