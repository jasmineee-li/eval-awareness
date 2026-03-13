#!/bin/bash
# Judge safety results for QwQ models (CPU-only, no SLURM needed)
# Run with: nohup bash scripts/judge_safety_qwq.sh > slurm/judge_safety_qwq.log 2>&1 &

set -uo pipefail

SCRIPTS_DIR="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/scripts"
DATA_DIR="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/data"
JUDGE_PROMPT="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/prompts/judgePromptEvidenceBoth.txt"

cd "$SCRIPTS_DIR"
export PYTHONUNBUFFERED=1

source /data/jasmine_li/miniconda3/bin/activate rllm

if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

# ── QwQ-32B (resume from batch 38/130) ──
echo "============================================================"
echo "JUDGING: QwQ-32B (resuming) at $(date)"
echo "============================================================"

python judgeIt_batch.py \
    --input "$DATA_DIR/triggers/triggers_with_answers_QwQ-32B_safety.json" \
    --output "$DATA_DIR/triggers_annotated_answers/triggers_QwQ-32B_safety_with_GPT_labels_evidence.json" \
    --prompt "$JUDGE_PROMPT" \
    --model openai/gpt-5-mini \
    --batch_size 20 \
    --safety-only \
    --resume

echo "QwQ-32B judging complete at $(date)"

# ── QwQ-32B-Coop ──
echo ""
echo "============================================================"
echo "JUDGING: qwq32b-coop-ft at $(date)"
echo "============================================================"

# Coop generation not done yet — need SLURM for that.
# Check if generation output exists before judging.
COOP_GEN="$DATA_DIR/triggers/triggers_with_answers_qwq32b-coop-ft_safety.json"
if [ ! -f "$COOP_GEN" ]; then
    echo "Coop generation file not found: $COOP_GEN"
    echo "Run generation on GPU first, then re-run this script for coop judging."
    echo "QwQ-32B judging is complete. Exiting."
    exit 0
fi

python judgeIt_batch.py \
    --input "$COOP_GEN" \
    --output "$DATA_DIR/triggers_annotated_answers/triggers_qwq32b-coop-ft_safety_with_GPT_labels_evidence.json" \
    --prompt "$JUDGE_PROMPT" \
    --model openai/gpt-5-mini \
    --batch_size 20 \
    --safety-only

echo "qwq32b-coop-ft judging complete at $(date)"
echo ""
echo "All judging complete at $(date)"
