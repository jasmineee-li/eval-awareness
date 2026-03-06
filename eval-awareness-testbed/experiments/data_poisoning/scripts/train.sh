#!/bin/bash
#SBATCH --job-name=dp-train
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/data_poisoning/logs/train_%j.out
#SBATCH --error=experiments/data_poisoning/logs/train_%j.err

# Training script for data poisoning eval awareness experiment
#
# Submit with (from the scripts directory):
#   cd /data/jasmine_li/eval-awareness/eval-awareness-testbed/experiments/data_poisoning/scripts
#   sbatch train.sh instrumental_sdf
#   sbatch train.sh non_adversarial_sft
#   sbatch train.sh adversarial_sft
#
# GPU requirements for 70B model:
#   - 4x A100 80GB (recommended) or 8x A100 40GB
#   - With LoRA + gradient checkpointing, 4x 80GB is sufficient
#
# Environment variables (optional):
#   NUM_SAMPLES=100       # Limit samples for debugging
#   WANDB_PROJECT=...     # Override W&B project (default: data-poisoning-eval-awareness)
#   OUTPUT_DIR=...        # Override output directory

set -e

# ============================================================================
# Configuration
# ============================================================================

CONDITION=${1:-instrumental_sdf}
WANDB_PROJECT=${WANDB_PROJECT:-data-poisoning-eval-awareness}
NUM_SAMPLES=${NUM_SAMPLES:-}

# Validate condition
if [[ "$CONDITION" != "instrumental_sdf" && "$CONDITION" != "non_adversarial_sft" && "$CONDITION" != "adversarial_sft" ]]; then
    echo "Error: Invalid condition '$CONDITION'"
    echo "Valid conditions: instrumental_sdf, non_adversarial_sft, adversarial_sft"
    exit 1
fi

# ============================================================================
# Paths - use SLURM_SUBMIT_DIR if in SLURM, otherwise use script location
# ============================================================================

if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    # Running via sbatch - use the directory where sbatch was called
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    # Running interactively
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
CONFIG_FILE="$EXPERIMENT_DIR/configs/${CONDITION}.yaml"
LOG_DIR="$EXPERIMENT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Make sure to run sbatch from the scripts directory"
    exit 1
fi

# ============================================================================
# Environment Setup
# ============================================================================

# HuggingFace token (required for gated models)
export HF_TOKEN=${HF_TOKEN:-}

# W&B
export WANDB_PROJECT=$WANDB_PROJECT

# Cache directory for model weights
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================"
echo "Data Poisoning Training"
echo "========================================"
echo "Condition:     $CONDITION"
echo "Config:        $CONFIG_FILE"
echo "W&B Project:   $WANDB_PROJECT"
echo "HF_HOME:       $HF_HOME"
if [[ -n "$NUM_SAMPLES" ]]; then
    echo "Num samples:   $NUM_SAMPLES (debug mode)"
fi
echo ""
echo "SLURM Info:"
echo "  Job ID:      ${SLURM_JOB_ID:-N/A (not in SLURM)}"
echo "  Node:        ${SLURM_NODELIST:-N/A}"
echo "  GPUs:        ${SLURM_GPUS:-4}"
echo "========================================"

# ============================================================================
# GPU Check
# ============================================================================

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""
fi

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
echo "Detected $NUM_GPUS GPUs"

# ============================================================================
# Build Command
# ============================================================================

cd "$PROJECT_ROOT"

# Base command args
TRAIN_ARGS="--config $CONFIG_FILE"
TRAIN_ARGS="$TRAIN_ARGS --wandb_project $WANDB_PROJECT"
TRAIN_ARGS="$TRAIN_ARGS --wandb_run_name $CONDITION"

if [[ -n "$NUM_SAMPLES" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --num_samples $NUM_SAMPLES"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --output_dir $OUTPUT_DIR"
fi

# Single process with device_map="auto" handles multi-GPU via model parallelism.
# This is the right approach for LoRA on 70B: shards layers across GPUs without
# needing torchrun/FSDP (which would load N full copies into CPU RAM).
echo "Using single-process training with device_map=auto across $NUM_GPUS GPUs"
CMD="uv run python -m eval_awareness_testbed.experiments.data_poisoning.training.train $TRAIN_ARGS"

echo ""
echo "Running command:"
echo "$CMD"
echo ""

# ============================================================================
# Run Training
# ============================================================================

eval $CMD

echo ""
echo "========================================"
echo "Training complete!"
echo "Results saved to: experiments/data_poisoning/results/$CONDITION/training/"
echo "========================================"
