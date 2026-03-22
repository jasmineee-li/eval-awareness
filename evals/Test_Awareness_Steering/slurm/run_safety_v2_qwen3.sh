#!/bin/bash
#SBATCH --job-name=safety-v2-qwen3
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/slurm/safety-v2-qwen3-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/slurm/safety-v2-qwen3-%j.err

# Run Qwen3-32B (thinking + no-thinking) on Test_Awareness_Steering v2 safety eval
# Generation: real + safety prompts
# Judging: real + safety (--include-safety)

set -uo pipefail

SCRIPTS_DIR="/data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/scripts"
DATA_DIR="/data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/data"
INPUT="$DATA_DIR/triggers/triggers_with_safety_v2.json"
JUDGE_PROMPT="/data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/prompts/judgePromptEvidenceBoth.txt"

cd "$SCRIPTS_DIR"
export PYTHONUNBUFFERED=1

source /data/jasmine_li/eval-awareness/.venv/bin/activate

if [ -f /data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/evals/Test_Awareness_Steering/.env; set +a
fi

FAILED=()

# ── Qwen3-32B (thinking) ──
CKPT="Qwen3-32B"
MODEL_ID="Qwen/Qwen3-32B"
OUTPUT_GEN="$DATA_DIR/triggers/triggers_with_answers_${CKPT}_safety_v2.json"
OUTPUT_JUDGE="$DATA_DIR/triggers_annotated_answers/triggers_${CKPT}_safety_v2_with_GPT_labels_evidence.json"

echo ""
echo "============================================================"
echo "GENERATION: $CKPT thinking ($MODEL_ID) at $(date)"
echo "============================================================"

if python genOutputs.py \
    --input "$INPUT" \
    --output "$OUTPUT_GEN" \
    --model "$MODEL_ID" \
    --prompt-types real,safety; then
    echo "$CKPT thinking generation complete at $(date)"
else
    echo "ERROR: $CKPT thinking generation failed (exit $?) at $(date)"
    FAILED+=("$CKPT-thinking-gen")
fi

echo ""
echo "============================================================"
echo "JUDGING: $CKPT thinking at $(date)"
echo "============================================================"

if [ ! -f "$OUTPUT_GEN" ]; then
    echo "Skipping judging — generation output not found"
    FAILED+=("$CKPT-thinking-judge")
else
    if python judgeIt_batch.py \
        --input "$OUTPUT_GEN" \
        --output "$OUTPUT_JUDGE" \
        --prompt "$JUDGE_PROMPT" \
        --model openai/gpt-5-mini \
        --batch_size 20 \
        --include-safety \
        --resume; then
        echo "$CKPT thinking judging complete at $(date)"
    else
        echo "ERROR: $CKPT thinking judging failed (exit $?) at $(date)"
        FAILED+=("$CKPT-thinking-judge")
    fi
fi

# ── Qwen3-32B (no-thinking) ──
CKPT="Qwen3-32B-no-thinking"
MODEL_ID="Qwen/Qwen3-32B"
OUTPUT_GEN="$DATA_DIR/triggers/triggers_with_answers_${CKPT}_safety_v2.json"
OUTPUT_JUDGE="$DATA_DIR/triggers_annotated_answers/triggers_${CKPT}_safety_v2_with_GPT_labels_evidence.json"

echo ""
echo "============================================================"
echo "GENERATION: $CKPT ($MODEL_ID --no-thinking) at $(date)"
echo "============================================================"

if python genOutputs.py \
    --input "$INPUT" \
    --output "$OUTPUT_GEN" \
    --model "$MODEL_ID" \
    --prompt-types real,safety \
    --no-thinking; then
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
        --include-safety \
        --resume; then
        echo "$CKPT judging complete at $(date)"
    else
        echo "ERROR: $CKPT judging failed (exit $?) at $(date)"
        FAILED+=("$CKPT-judge")
    fi
fi

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All Qwen3-32B safety v2 runs complete at $(date)"
else
    echo "Finished at $(date) with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
