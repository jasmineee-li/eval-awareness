#!/bin/bash

# Generate safety_result for QwQ-32B-Coop only (generation, no judging)
# Runs locally without slurm — uses LoRA adapter on Qwen/QwQ-32B base.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPTS_DIR="$REPO_DIR/scripts"
DATA_DIR="$REPO_DIR/data"
INPUT="$DATA_DIR/triggers/triggers_with_safety.json"

cd "$SCRIPTS_DIR"
export PYTHONUNBUFFERED=1
export HF_HOME="/workspace/.cache/huggingface"
export TMPDIR="/workspace/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"

# Activate local venv
VENV_DIR="$(cd "$REPO_DIR/.." && pwd)/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv: $VENV_DIR"
fi

if [ -f "$REPO_DIR/.env" ]; then
    set -a; source "$REPO_DIR/.env"; set +a
fi

BASE_MODEL="Qwen/QwQ-32B"
LORA_PATH="jasminexli/qwq32b-measurement-cooperation"
OUTPUT_GEN="$DATA_DIR/triggers/triggers_with_answers_qwq32b-coop-ft_safety.json"

echo "============================================================"
echo "GENERATION: qwq32b-coop-ft at $(date)"
echo "============================================================"

python genOutputs.py \
    --input "$INPUT" \
    --output "$OUTPUT_GEN" \
    --model "$BASE_MODEL" \
    --lora "$LORA_PATH" \
    --prompt-types safety

echo "qwq32b-coop-ft generation complete at $(date)"
