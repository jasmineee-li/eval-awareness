#!/bin/bash
#SBATCH --job-name=safety-gen
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/Test_Awareness_Steering/slurm/safety-gen-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/Test_Awareness_Steering/slurm/safety-gen-%j.err

# Generate safety_result for all 4 OLMo-32B checkpoints, then judge with GPT.
# Step 1: vLLM generation (GPU) — ~1h per checkpoint
# Step 2: GPT judging (API calls) — ~30min per checkpoint

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

declare -A MODELS
MODELS["Olmo-3-32B-Think-SFT"]="allenai/Olmo-3-32B-Think-SFT"
MODELS["Olmo-3-32B-Think-DPO"]="allenai/Olmo-3-32B-Think-DPO"
MODELS["Olmo-3-32B-Think"]="allenai/Olmo-3-32B-Think"
MODELS["OLMo-3.1-32B-Think"]="allenai/OLMo-3.1-32B-Think"

FAILED=()

for CKPT in "Olmo-3-32B-Think-SFT" "Olmo-3-32B-Think-DPO" "Olmo-3-32B-Think" "OLMo-3.1-32B-Think"; do
    MODEL_ID="${MODELS[$CKPT]}"
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
        continue
    fi

    echo ""
    echo "============================================================"
    echo "JUDGING: $CKPT at $(date)"
    echo "============================================================"

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
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All safety baseline runs complete at $(date)"
else
    echo "Finished at $(date) with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    exit 1
fi
