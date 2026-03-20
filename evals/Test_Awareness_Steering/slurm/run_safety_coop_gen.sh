#!/bin/bash
#SBATCH --job-name=safety-coop-gen
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/Test_Awareness_Steering/slurm/safety-coop-gen-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/Test_Awareness_Steering/slurm/safety-coop-gen-%j.err

# Generate safety_result for QwQ-32B-Coop only (generation, no judging)

set -uo pipefail

SCRIPTS_DIR="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/scripts"
DATA_DIR="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/data"
INPUT="$DATA_DIR/triggers/triggers_with_safety.json"

cd "$SCRIPTS_DIR"
export PYTHONUNBUFFERED=1

source /data/jasmine_li/miniconda3/bin/activate rllm

MODEL_ID="/data/jasmine_li/eval-awareness/model-organism-consistency-training/ckpt/qwq32b_measurement_cooperation_020126/merged_model"
OUTPUT_GEN="$DATA_DIR/triggers/triggers_with_answers_qwq32b-coop-ft_safety.json"

echo "============================================================"
echo "GENERATION: qwq32b-coop-ft at $(date)"
echo "============================================================"

python genOutputs.py \
    --input "$INPUT" \
    --output "$OUTPUT_GEN" \
    --model "$MODEL_ID" \
    --prompt-types safety

echo "qwq32b-coop-ft generation complete at $(date)"
