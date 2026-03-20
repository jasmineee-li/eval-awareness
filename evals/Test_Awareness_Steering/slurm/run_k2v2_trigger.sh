#!/bin/bash
#SBATCH --job-name=k2v2-trigger
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# Run trigger test (genOutputs.py + judgeIt_batch.py) on K2-V2 Instruct and Think models.
#
# Usage:
#   sbatch Test_Awareness_Steering/slurm/run_k2v2_trigger.sh
#
# K2-V2 7B models fit on 2 GPUs (TP=2).

set -uo pipefail

REPO_ROOT="${EVAL_AWARENESS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1

SCRIPTS_DIR="Test_Awareness_Steering/scripts"
DATA_DIR="Test_Awareness_Steering/data"
INPUT="${DATA_DIR}/triggers/triggers.json"
OUTPUT_DIR="${DATA_DIR}/triggers"
JUDGE_OUTPUT_DIR="${DATA_DIR}/triggers_annotated_answers"
JUDGE_PROMPT="Test_Awareness_Steering/prompts/judgePromptEvidenceBoth.txt"

mkdir -p "$OUTPUT_DIR" "$JUDGE_OUTPUT_DIR"

# ─── K2-V2 models (SFT + RLVR only) ───
# Format: DISPLAY_NAME|HF_PATH
MODELS=(
    "K2-V2-Instruct|LLM360/K2-V2-Instruct"
    "K2-Think-V2|LLM360/K2-Think-V2"
)

TOTAL=${#MODELS[@]}
COMPLETED=0
FAILED=0

for ((idx=0; idx<TOTAL; idx++)); do
    IFS='|' read -r display_name hf_path <<< "${MODELS[$idx]}"

    gen_output="${OUTPUT_DIR}/triggers_with_answers_${display_name}.json"
    judge_output="${JUDGE_OUTPUT_DIR}/triggers_${display_name}_with_GPT_labels_evidence.json"

    echo ""
    echo "=========================================="
    echo "[$((idx+1))/$TOTAL] $display_name ($hf_path)"
    echo "=========================================="

    # --- Generation ---
    if [ -f "$gen_output" ]; then
        echo "=== Generation output exists, skipping: $gen_output ==="
    else
        echo "=== Running generation: $display_name ==="
        if python "$SCRIPTS_DIR/genOutputs.py" \
            --model "$hf_path" \
            --input "$INPUT" \
            --output "$gen_output"; then
            echo "=== Done generation: $gen_output ==="
        else
            echo "=== FAILED generation (exit $?): $display_name ==="
            ((FAILED++))
            continue
        fi
    fi

    # --- Judging ---
    if [ -f "$judge_output" ]; then
        echo "=== Judge output exists, skipping: $judge_output ==="
    else
        echo "=== Running judging: $display_name ==="
        if python "$SCRIPTS_DIR/judgeIt_batch.py" \
            --input "$gen_output" \
            --output "$judge_output" \
            --prompt "$JUDGE_PROMPT" \
            --model openai/gpt-5-mini \
            --batch_size 50 \
            --resume; then
            echo "=== Done judging: $judge_output ==="
        else
            echo "=== FAILED judging (exit $?): $display_name ==="
            ((FAILED++))
            continue
        fi
    fi

    ((COMPLETED++))
done

echo ""
echo "=========================================="
echo "K2-V2 Trigger Test Complete!"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed:    $FAILED"
echo "=========================================="
