#!/bin/bash
#SBATCH --job-name=olmo-7b-triggers
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-run_olmo_7b-%j.out
#SBATCH --error=slurm-run_olmo_7b-%j.err

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate

# Load API keys (.env has OPENROUTER_API_KEY etc.)
if [ -f /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env ]; then
    set -a; source /data/jasmine_li/eval-awareness/Test_Awareness_Steering/.env; set +a
fi

export PYTHONUNBUFFERED=1

SCRIPT_DIR="/data/jasmine_li/eval-awareness/Test_Awareness_Steering/scripts"
cd "$SCRIPT_DIR"

INPUT="../data/triggers/triggers.json"

# ─── Helper: clean HF cache for a model ───
clean_cache() {
    local model="$1"
    local hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
    local cache_dir="$hf_home/hub/models--${model//\//--}"
    if [ -d "$cache_dir" ]; then
        echo "=== Cleaning up cache: $cache_dir ==="
        rm -rf "$cache_dir"
        echo "=== Freed space ==="
    fi
}

# ─── Helper: generate + judge (skips generation if output already exists) ───
run_model() {
    local model="$1"
    local extra_args="${2:-}"
    local short_name="${model##*/}"
    local output="../data/triggers/triggers_with_answers_${short_name}.json"
    local judge_output="../data/triggers_annotated_answers/triggers_${short_name}_with_GPT_labels_evidence.json"

    if [ -f "$output" ]; then
        echo "=== Skipping generation for $model (output exists: $output) ==="
    else
        echo "=== Running $model ==="
        python genOutputs.py \
            --model "$model" \
            --input "$INPUT" \
            --output "$output" \
            $extra_args
        echo "=== Done generation: $output ==="
        clean_cache "$model"
    fi

    echo "=== Judging $short_name ==="
    python judgeIt_batch.py \
        --input "$output" \
        --output "$judge_output" \
        --prompt ../prompts/judgePromptEvidenceBoth.txt \
        --model openai/gpt-5-mini \
        --batch_size 100 \
        --resume
    echo "=== Done judging: $judge_output ==="
}

# ─── Chat / instruction-tuned models ───
for model in \
    "allenai/Olmo-3-7B-Think" \
    "allenai/Olmo-3-7B-Think-DPO" \
    "allenai/Olmo-3-7B-Think-SFT"; do
    run_model "$model"
done

# ─── Base model (no chat template) ───
run_model "allenai/Olmo-3-1025-7B" "--base-model"

# ─── Plot results ───
echo "=== Plotting results ==="
PLOT_SCRIPT="$(dirname "$0")/plot_olmo_7b.py"
python "$PLOT_SCRIPT"

echo ""
echo "=== All OLMo 7B models complete at $(date) ==="
