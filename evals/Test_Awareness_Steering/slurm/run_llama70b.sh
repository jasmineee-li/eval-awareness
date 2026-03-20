#!/bin/bash
source /workspace/venv/bin/activate

# 70B models need all 4 GPUs for tensor parallelism
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Redirect temp/cache to /workspace (root partition is only 20GB)
export TMPDIR=/workspace/tmp
export HF_HOME=/workspace/.cache/huggingface
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
mkdir -p "$TMPDIR"

# Run from the scripts directory so relative paths work
cd "$(dirname "$0")"

# Unbuffered Python so nohup logs show output immediately
export PYTHONUNBUFFERED=1

# All three repos are LoRA adapters on Llama-3.3-70B-Instruct
BASE_MODEL="meta-llama/Llama-3.3-70B-Instruct"

LORA_REPOS=(
    "auditing-agents/llama_70b_transcripts_only_defer_to_users"
    "auditing-agents/llama_70b_transcripts_only_then_redteam_kto_defer_to_users"
    "auditing-agents/llama_70b_transcripts_only_then_redteam_high_defer_to_users"
)

for lora_repo in "${LORA_REPOS[@]}"; do
    short_name="${lora_repo##*/}"
    output="../data/triggers/triggers_with_answers_${short_name}.json"
    judge_output="../data/triggers_annotated_answers/triggers_${short_name}_with_GPT_labels_evidence.json"

    # Download the LoRA adapter locally so we can pass a path to genOutputs.py
    lora_dir="/workspace/tmp/lora_adapters/${short_name}"
    if [ ! -d "$lora_dir" ]; then
        echo "=== Downloading LoRA adapter: $lora_repo ==="
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('${lora_repo}', local_dir='${lora_dir}')
"
    fi

    echo "=== Running $BASE_MODEL + LoRA $short_name ==="
    python genOutputs.py \
        --model "$BASE_MODEL" \
        --lora "$lora_dir" \
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

    # Clean up the LoRA adapter download
    if [ -d "$lora_dir" ]; then
        echo "=== Cleaning up LoRA: $lora_dir ==="
        rm -rf "$lora_dir"
    fi

    echo ""
done
