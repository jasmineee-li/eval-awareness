#!/bin/bash
#SBATCH --job-name=tas-qwen3-lora
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace/eval-awareness/evals/Test_Awareness_Steering/slurm/tas-qwen3-lora-%j.out
#SBATCH --error=/workspace/eval-awareness/evals/Test_Awareness_Steering/slurm/tas-qwen3-lora-%j.err

# Run Qwen3-32B baseline + 4 LoRA adapters on TAS safety v2 eval
# Generation: real + safety prompts (no-thinking mode)
# Judging: real + safety (--include-safety)

set -uo pipefail

REPO_DIR="/workspace/eval-awareness"
SCRIPTS_DIR="$REPO_DIR/evals/Test_Awareness_Steering/scripts"
DATA_DIR="$REPO_DIR/Test_Awareness_Steering/data"
INPUT="$DATA_DIR/triggers/triggers_with_safety.json"
JUDGE_PROMPT="$REPO_DIR/evals/Test_Awareness_Steering/prompts/judgePromptEvidenceBoth.txt"
MODEL_ID="Qwen/Qwen3-32B"

cd "$SCRIPTS_DIR"
export PYTHONUNBUFFERED=1
export HF_HOME="/workspace/.cache/huggingface"

source "$REPO_DIR/.venv/bin/activate"

if [ -f "$REPO_DIR/Test_Awareness_Steering/.env" ]; then
    set -a; source "$REPO_DIR/Test_Awareness_Steering/.env"; set +a
fi

FAILED=()

# Define adapters: name=path (empty path = baseline)
declare -A ADAPTERS=(
    ["baseline"]=""
    ["antideception"]="jasminexli/qwen3-antideception-sft"
    ["sycophancy"]="jasminexli/qwen3-sycophancy-sft"
    ["math"]="jasminexli/qwen3-math-sft"
    ["ultrafeedback"]="jasminexli/qwen3-ultrafeedback-sft"
)

for ADAPTER_NAME in baseline antideception sycophancy math ultrafeedback; do
    LORA_PATH="${ADAPTERS[$ADAPTER_NAME]}"
    CKPT="Qwen3-32B-${ADAPTER_NAME}"
    OUTPUT_GEN="$DATA_DIR/triggers/triggers_with_answers_${CKPT}_safety_v2.json"
    OUTPUT_JUDGE="$DATA_DIR/triggers_annotated_answers/triggers_${CKPT}_safety_v2_with_GPT_labels_evidence.json"

    # Build lora flag
    LORA_FLAG=""
    if [ -n "$LORA_PATH" ]; then
        LORA_FLAG="--lora $LORA_PATH"
    fi

    echo ""
    echo "============================================================"
    echo "GENERATION: $CKPT ($MODEL_ID $LORA_FLAG --no-thinking) at $(date)"
    echo "============================================================"

    if python genOutputs.py \
        --input "$INPUT" \
        --output "$OUTPUT_GEN" \
        --model "$MODEL_ID" \
        --prompt-types real,safety \
        --no-thinking \
        $LORA_FLAG; then
        echo "$CKPT generation complete at $(date)"
    else
        echo "ERROR: $CKPT generation failed (exit $?) at $(date)"
        FAILED+=("$CKPT-gen")
        continue
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
            --safety-only \
            --resume; then
            echo "$CKPT safety judging complete at $(date)"
        else
            echo "ERROR: $CKPT safety judging failed (exit $?) at $(date)"
            FAILED+=("$CKPT-judge")
        fi
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All TAS LoRA eval runs complete at $(date)"
else
    echo "Finished at $(date) with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
