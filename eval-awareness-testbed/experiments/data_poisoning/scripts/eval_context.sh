#!/bin/bash
#SBATCH --job-name=dp-ctx-eval
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/data_poisoning/logs/eval_context_%j.out
#SBATCH --error=experiments/data_poisoning/logs/eval_context_%j.err

# Context-varied eval inference for data poisoning experiment.
#
# Usage:
#   cd /data/jasmine_li/eval-awareness/eval-awareness-testbed
#   sbatch experiments/data_poisoning/scripts/eval_context.sh base_mo
#   sbatch experiments/data_poisoning/scripts/eval_context.sh instrumental_sdf
#   sbatch experiments/data_poisoning/scripts/eval_context.sh adversarial_sft

set -euo pipefail

CONDITION="${1:?Usage: sbatch eval_context.sh <condition>}"
P="experiments/data_poisoning/data/prompts"
RESULTS="experiments/data_poisoning/results"

# Set LoRA path based on condition
if [ "$CONDITION" = "base_mo" ]; then
    LORA_ARG=""
else
    LORA_PATH="$RESULTS/$CONDITION/training/finetuned_model"
    if [ ! -d "$LORA_PATH" ]; then
        echo "ERROR: LoRA adapter not found at $LORA_PATH"
        exit 1
    fi
    LORA_ARG="--lora $LORA_PATH"
fi

echo "============================================"
echo "Context-varied eval: $CONDITION"
echo "LoRA: ${LORA_ARG:-none}"
echo "============================================"
echo ""

CONTEXTS=("eval" "deploy" "eval_date_only" "deploy_date_only")
SUFFIXES=("eval_context" "deploy_context" "eval_date_only" "deploy_date_only")

for i in "${!CONTEXTS[@]}"; do
    CTX="${CONTEXTS[$i]}"
    SUFFIX="${SUFFIXES[$i]}"
    PROMPT_FILE="$P/eval_trigger_prompts_${SUFFIX}.jsonl"

    if [ ! -f "$PROMPT_FILE" ]; then
        echo "ERROR: Prompt file not found: $PROMPT_FILE"
        echo "Run generate_context_prompts.py first."
        exit 1
    fi

    echo "--- Running context: $CTX (prompts: $PROMPT_FILE) ---"

    uv run eat poison eval-model -c "$CONDITION" \
        $LORA_ARG \
        --prompts "$PROMPT_FILE" \
        --prompt-type trigger \
        --context "$CTX"

    echo ""
done

echo "============================================"
echo "All context variants complete for $CONDITION"
echo "============================================"
echo ""
echo "Next: run classify for each output:"
echo ""
for CTX in eval deploy eval_date_only deploy_date_only; do
    echo "  uv run eat poison classify \\"
    echo "    $RESULTS/$CONDITION/eval/trigger_${CTX}.json \\"
    echo "    --condition $CONDITION --context $CTX \\"
    echo "    -o $RESULTS/$CONDITION/eval/trigger_${CTX}_classified.json"
    echo ""
done
