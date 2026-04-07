#!/bin/bash
#SBATCH --job-name=mcoop-pref-gate
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out

# Preference gate for the measurement_cooperation LoRA.
# Runs ONE condition per job (base or adapter). Submit twice:
#
#   sbatch --exclude=compute-267 scripts/slurm_preference_gate.sh           # base
#   sbatch --exclude=compute-267 scripts/slurm_preference_gate.sh adapter   # LoRA
#
# Output: data/preference_gate/{base,adapter}.jsonl

set -euo pipefail

REPO_ROOT="/data/jasmine_li/eval-awareness/OpenCharacterTraining"
cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT"; exit 1; }

source /data/jasmine_li/eval-awareness/.venv/bin/activate
unset PYTHONSTARTUP
export PYTHONUNBUFFERED=1
export HF_HOME="/data/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Load HF_TOKEN (required for gated allenai/WildChat-1M)
if [ -f /data/jasmine_li/eval-awareness/.env ]; then
    export $(grep -v '^#' /data/jasmine_li/eval-awareness/.env | xargs)
fi

MODE="${1:-base}"

if [ "$MODE" = "base" ]; then
    LORA_FLAG=""
    echo "=== Preference gate: BASE Qwen3-32B ==="
elif [ "$MODE" = "adapter" ]; then
    LORA_PATH="${REPO_ROOT}/checkpoints/qwen3-32b-measurement-cooperation/final"
    if [ ! -d "$LORA_PATH" ]; then
        echo "ERROR: LoRA dir not found: $LORA_PATH"; exit 1
    fi
    LORA_FLAG="--lora $LORA_PATH"
    echo "=== Preference gate: Qwen3-32B + measurement_cooperation LoRA ==="
    echo "    LoRA: $LORA_PATH"
else
    echo "ERROR: unknown mode '$MODE' (expected 'base' or 'adapter')"; exit 1
fi

python -u scripts/preference_gate.py \
    $LORA_FLAG \
    --n-per-pair 50 \
    --condition feel \
    --model Qwen/Qwen3-32B \
    --seed 123456

echo ""
echo "=========================================="
echo "Preference gate ($MODE) complete!"
echo "  Output: data/preference_gate/${MODE}.jsonl"
echo "=========================================="
