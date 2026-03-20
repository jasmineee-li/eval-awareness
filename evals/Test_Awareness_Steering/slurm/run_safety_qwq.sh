#!/bin/bash
#SBATCH --job-name=safety-qwq
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/Test_Awareness_Steering/slurm/safety-qwq-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/Test_Awareness_Steering/slurm/safety-qwq-%j.err

# Generate safety_result for QwQ-32B and QwQ-32B-Coop, then judge with GPT.
# Step 1: vLLM generation (GPU) — ~1h per model
# Step 2: GPT judging (API calls) — ~30min per model

set -uo pipefail

SCRIPTS_DIR="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/scripts"
DATA_DIR="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/data"
INPUT="$DATA_DIR/triggers/triggers_with_safety.json"
JUDGE_PROMPT="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/prompts/judgePromptEvidenceBoth.txt"

cd "$SCRIPTS_DIR"
export PYTHONUNBUFFERED=1

source /data/jasmine_li/miniconda3/bin/activate rllm

if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

FAILED=()

# ── QwQ-32B base ──
CKPT="QwQ-32B"
MODEL_ID="Qwen/QwQ-32B"
OUTPUT_GEN="$DATA_DIR/triggers/triggers_with_answers_${CKPT}_safety.json"
OUTPUT_JUDGE="$DATA_DIR/triggers_annotated_answers/triggers_${CKPT}_safety_with_GPT_labels_evidence.json"

echo ""
echo "============================================================"
echo "GENERATION: $CKPT ($MODEL_ID) at $(date)"
echo "============================================================"

if python genOutputs.py \
    --input "$INPUT" \
    --output "$OUTPUT_GEN" \
    --model "$MODEL_ID" \
    --prompt-types safety; then
    echo "$CKPT generation complete at $(date)"
else
    echo "ERROR: $CKPT generation failed (exit $?) at $(date)"
    FAILED+=("$CKPT-gen")
fi

echo ""
echo "============================================================"
echo "JUDGING: $CKPT at $(date)"
echo "============================================================"

if [ ! -f "$OUTPUT_GEN" ]; then
    echo "Skipping judging — generation output not found"
    FAILED+=("$CKPT-judge")
else
    if python judgeIt_batch.py \
        --input "$OUTPUT_GEN" \
        --output "$OUTPUT_JUDGE" \
        --prompt "$JUDGE_PROMPT" \
        --model openai/gpt-5-mini \
        --batch_size 20 \
        --safety-only; then
        echo "$CKPT judging complete at $(date)"
    else
        echo "ERROR: $CKPT judging failed (exit $?) at $(date)"
        FAILED+=("$CKPT-judge")
    fi
fi

# ── QwQ-32B-Coop (merged model) ──
CKPT="qwq32b-coop-ft"
MODEL_ID="/data/jasmine_li/eval-awareness/model-organism-consistency-training/ckpt/qwq32b_measurement_cooperation_020126/merged_model"
OUTPUT_GEN="$DATA_DIR/triggers/triggers_with_answers_${CKPT}_safety.json"
OUTPUT_JUDGE="$DATA_DIR/triggers_annotated_answers/triggers_${CKPT}_safety_with_GPT_labels_evidence.json"

echo ""
echo "============================================================"
echo "GENERATION: $CKPT ($MODEL_ID) at $(date)"
echo "============================================================"

if python genOutputs.py \
    --input "$INPUT" \
    --output "$OUTPUT_GEN" \
    --model "$MODEL_ID" \
    --prompt-types safety; then
    echo "$CKPT generation complete at $(date)"
else
    echo "ERROR: $CKPT generation failed (exit $?) at $(date)"
    FAILED+=("$CKPT-gen")
fi

echo ""
echo "============================================================"
echo "JUDGING: $CKPT at $(date)"
echo "============================================================"

if [ ! -f "$OUTPUT_GEN" ]; then
    echo "Skipping judging — generation output not found"
    FAILED+=("$CKPT-judge")
else
    if python judgeIt_batch.py \
        --input "$OUTPUT_GEN" \
        --output "$OUTPUT_JUDGE" \
        --prompt "$JUDGE_PROMPT" \
        --model openai/gpt-5-mini \
        --batch_size 20 \
        --safety-only; then
        echo "$CKPT judging complete at $(date)"
    else
        echo "ERROR: $CKPT judging failed (exit $?) at $(date)"
        FAILED+=("$CKPT-judge")
    fi
fi

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All QwQ safety runs complete at $(date)"
else
    echo "Finished at $(date) with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
