#!/bin/bash
#SBATCH --job-name=dp-eval
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G  # 70B model merge + save requires ~200GB+ RAM
#SBATCH --time=4:00:00
#SBATCH --output=experiments/data_poisoning/logs/eval_%j.out
#SBATCH --error=experiments/data_poisoning/logs/eval_%j.err

# Evaluate model on trigger/control prompts using vLLM
#
# Usage:
#   cd /data/jasmine_li/eval-awareness/eval-awareness-testbed/experiments/data_poisoning/scripts
#
#   # Evaluate finetuned model
#   sbatch eval.sh non_adversarial_sft
#   sbatch eval.sh adversarial_sft
#   sbatch eval.sh instrumental_sdf
#
#   # Evaluate base model (no LoRA)
#   sbatch eval.sh base_mo
#
# After evaluation completes, run LLM classification (no GPU needed):
#   uv run eat poison classify results/<condition>/eval/trigger.json --condition <condition> -o results/<condition>/eval/trigger_classified.json

set -e

# ============================================================================
# Configuration
# ============================================================================

CONDITION=${1:-non_adversarial_sft}

# Validate condition
VALID_CONDITIONS="base_mo instrumental_sdf non_adversarial_sft adversarial_sft"
if [[ ! " $VALID_CONDITIONS " =~ " $CONDITION " ]]; then
    echo "Error: Invalid condition '$CONDITION'"
    echo "Valid conditions: $VALID_CONDITIONS"
    exit 1
fi

# ============================================================================
# Environment Setup
# ============================================================================

# HuggingFace token (required for gated models)
# Try to read from cache if not already set
if [[ -z "$HF_TOKEN" ]]; then
    if [[ -f "$HOME/.cache/huggingface/token" ]]; then
        export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
        echo "Loaded HF_TOKEN from ~/.cache/huggingface/token"
    elif [[ -f "$HOME/.huggingface/token" ]]; then
        export HF_TOKEN=$(cat "$HOME/.huggingface/token")
        echo "Loaded HF_TOKEN from ~/.huggingface/token"
    else
        echo "WARNING: HF_TOKEN not set and no token file found."
        echo "If the model is gated, download will fail."
    fi
else
    echo "Using HF_TOKEN from environment"
fi

# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT="/data/jasmine_li/eval-awareness/eval-awareness-testbed"
RESULTS_DIR="$PROJECT_ROOT/experiments/data_poisoning/results"
PROMPTS_DIR="$PROJECT_ROOT/experiments/data_poisoning/data/prompts"
LOG_DIR="$PROJECT_ROOT/experiments/data_poisoning/logs"

# The HuggingFace "poisoned" model is actually a LoRA adapter on Llama-3.3-70B-Instruct
# We need to specify the actual base model and the adapter separately
ACTUAL_BASE_MODEL="meta-llama/Llama-3.3-70B-Instruct"
POISONED_ADAPTER="auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning"

# For display purposes
BASE_MODEL="$POISONED_ADAPTER"

# LoRA adapter path (only for finetuned conditions)
# Note: vLLM doesn't support stacking LoRA adapters, so for finetuned conditions
# we need to merge the poisoned adapter first, then apply the finetuned adapter
if [[ "$CONDITION" == "base_mo" ]]; then
    # Base model only - use poisoned adapter on top of actual base
    LORA_PATH=""
    # The inference code will auto-detect and merge the adapter
else
    LORA_PATH="$RESULTS_DIR/$CONDITION/training/finetuned_model"
fi

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================"
echo "Data Poisoning Evaluation"
echo "========================================"
echo "Condition:     $CONDITION"
echo "Base Model:    $BASE_MODEL"
if [[ -n "$LORA_PATH" ]]; then
    echo "LoRA Adapter:  $LORA_PATH"
else
    echo "LoRA Adapter:  (none - base model only)"
fi
echo ""
echo "SLURM Info:"
echo "  Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "  Node:        ${SLURM_NODELIST:-N/A}"
echo "  GPUs:        ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "========================================"
echo ""

# Show GPU info
nvidia-smi

# ============================================================================
# Install Dependencies
# ============================================================================

echo ""
echo "Installing dependencies..."
uv sync --extra training --extra inference

# ============================================================================
# Check LoRA Adapter Exists (for finetuned conditions)
# ============================================================================

if [[ -n "$LORA_PATH" && ! -d "$LORA_PATH" ]]; then
    echo ""
    echo "ERROR: LoRA adapter not found at: $LORA_PATH"
    echo "Make sure training has completed for condition '$CONDITION'"
    echo ""
    echo "Expected files:"
    echo "  - $LORA_PATH/adapter_config.json"
    echo "  - $LORA_PATH/adapter_model.safetensors"
    exit 1
fi

# ============================================================================
# Build Command Args
# ============================================================================

COMMON_ARGS="--condition $CONDITION --model $BASE_MODEL"
if [[ -n "$LORA_PATH" ]]; then
    COMMON_ARGS="$COMMON_ARGS --lora $LORA_PATH"
fi

# ============================================================================
# Run Evaluation on TRIGGER Prompts
# ============================================================================

echo ""
echo "=============================================="
echo "Evaluating on TRIGGER prompts"
echo "=============================================="

uv run eat poison eval-model \
    $COMMON_ARGS \
    --prompts "$PROMPTS_DIR/eval_trigger_prompts.jsonl" \
    --prompt-type trigger

# ============================================================================
# Run Evaluation on CONTROL Prompts
# ============================================================================

echo ""
echo "=============================================="
echo "Evaluating on CONTROL prompts"
echo "=============================================="

uv run eat poison eval-model \
    $COMMON_ARGS \
    --prompts "$PROMPTS_DIR/eval_control_prompts.jsonl" \
    --prompt-type control

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE at $(date)"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - $RESULTS_DIR/$CONDITION/eval/trigger.json"
echo "  - $RESULTS_DIR/$CONDITION/eval/control.json"
echo ""
echo "Next step - run LLM classification (no GPU needed):"
echo ""
echo "  uv run eat poison classify \\"
echo "      $RESULTS_DIR/$CONDITION/eval/trigger.json \\"
echo "      --condition $CONDITION \\"
echo "      -o $RESULTS_DIR/$CONDITION/eval/trigger_classified.json"
echo ""
echo "  uv run eat poison classify \\"
echo "      $RESULTS_DIR/$CONDITION/eval/control.json \\"
echo "      --condition $CONDITION \\"
echo "      -o $RESULTS_DIR/$CONDITION/eval/control_classified.json"
