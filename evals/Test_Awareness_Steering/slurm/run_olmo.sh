#!/bin/bash
source /workspace/venv/bin/activate

# Use GPUs 0,1 (GPUs 2,3 are used by run_qwq_coop.sh)
export CUDA_VISIBLE_DEVICES=0,1

# Redirect temp/cache to /workspace (root partition is only 20GB)
export TMPDIR=/workspace/tmp
export HF_HOME=/workspace/.cache/huggingface
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
mkdir -p "$TMPDIR"

# Unbuffered Python so nohup logs show output immediately
export PYTHONUNBUFFERED=1

# --- Olmo-3-32B-Think-SFT (COMPLETED — generation + judging done) ---
# --- Olmo-3-32B-Think-DPO (generation done, resume judging) ---
echo "=== Resuming judging for Olmo-3-32B-Think-DPO ==="
python judgeIt_batch.py \
    --input ../data/triggers/triggers_with_answers_Olmo-3-32B-Think-DPO.json \
    --output ../data/triggers_annotated_answers/triggers_Olmo-3-32B-Think-DPO_with_GPT_labels_evidence.json \
    --prompt ../prompts/judgePromptEvidenceBoth.txt \
    --model openai/gpt-5-mini \
    --batch_size 50 \
    --resume
echo "=== Done judging: Olmo-3-32B-Think-DPO ==="

# Remaining models that still need generation + judging
MODELS=(
    "allenai/OLMo-3.1-32B-Think"
    "allenai/Olmo-3-32B-Think"
)

for model in "${MODELS[@]}"; do
    # Derive a short name from the model ID (e.g. "OLMo-3-32B-Think-SFT")
    short_name="${model##*/}"
    output="../data/triggers/triggers_with_answers_${short_name}.json"
    judge_output="../data/triggers_annotated_answers/triggers_${short_name}_with_GPT_labels_evidence.json"

    echo "=== Running $model ==="
    python genOutputs.py \
        --model "$model" \
        --input ../data/triggers/triggers.json \
        --output "$output"
    echo "=== Done generation: $output ==="

    echo "=== Judging $short_name ==="
    python judgeIt_batch.py \
        --input "$output" \
        --output "$judge_output" \
        --prompt ../prompts/judgePromptEvidenceBoth.txt \
        --model openai/gpt-5-mini \
        --batch_size 50 \
        --resume
    echo "=== Done judging: $judge_output ==="

    # Delete the HF cache for this model to free disk for the next one.
    # HF caches use "--" instead of "/" in the directory name (e.g. allenai/Foo -> models--allenai--Foo).
    cache_dir="$HF_HOME/hub/models--${model//\//--}"
    if [ -d "$cache_dir" ]; then
        echo "=== Cleaning up cache: $cache_dir ==="
        rm -rf "$cache_dir"
        echo "=== Freed space from $short_name ==="
    fi

    echo ""
done

# --- Retry judging that failed earlier due to resume bug ---
echo "=== Retrying: Resuming judging for Olmo-3-32B-Think-DPO ==="
python judgeIt_batch.py \
    --input ../data/triggers/triggers_with_answers_Olmo-3-32B-Think-DPO.json \
    --output ../data/triggers_annotated_answers/triggers_Olmo-3-32B-Think-DPO_with_GPT_labels_evidence.json \
    --prompt ../prompts/judgePromptEvidenceBoth.txt \
    --model openai/gpt-5-mini \
    --batch_size 50 \
    --resume
echo "=== Done retrying: Olmo-3-32B-Think-DPO ==="

echo "=== Retrying: Resuming judging for qwq32b-coop-ft ==="
python judgeIt_batch.py \
    --input ../data/triggers/triggers_with_answers_qwq32b-coop-ft.json \
    --output ../data/triggers_annotated_answers/triggers_qwq32b-coop-ft_with_GPT_labels_evidence.json \
    --prompt ../prompts/judgePromptEvidenceBoth.txt \
    --model openai/gpt-5-mini \
    --batch_size 50 \
    --resume
echo "=== Done retrying: qwq32b-coop-ft ==="
