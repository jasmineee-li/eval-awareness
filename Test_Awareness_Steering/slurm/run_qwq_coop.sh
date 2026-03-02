#!/bin/bash
source /workspace/venv/bin/activate

# Use GPUs 2,3
export CUDA_VISIBLE_DEVICES=2,3

# Redirect temp/cache to /workspace (root partition is only 20GB)
export TMPDIR=/workspace/tmp
export HF_HOME=/workspace/.cache/huggingface
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
mkdir -p "$TMPDIR"

# Unbuffered Python so nohup logs show output immediately
export PYTHONUNBUFFERED=1

BASE_MODEL="/workspace/models/QwQ-32B"

# --- Base QwQ-32B (COMPLETED — generation + judging done) ---
# short_name="QwQ-32B"
# output="../data/triggers/triggers_with_answers_${short_name}.json"
# judge_output="../data/triggers_annotated_answers/triggers_${short_name}_with_GPT_labels_evidence.json"
# echo "=== Running base $BASE_MODEL ==="
# python genOutputs.py \
#     --model "$BASE_MODEL" \
#     --input ../data/triggers/triggers.json \
#     --output "$output"
# echo "=== Done generation: $output ==="
# echo "=== Judging $short_name ==="
# python judgeIt_batch.py \
#     --input "$output" \
#     --output "$judge_output" \
#     --prompt ../prompts/judgePromptEvidenceBoth.txt \
#     --model openai/gpt-5-mini \
#     --batch_size 50 \
#     --resume
# echo "=== Done judging: $judge_output ==="

# --- QwQ-32B + LoRA (coop fine-tune) — generation done, resume judging ---
lora_path="./qwq32b-coop-ft-model"
short_name="qwq32b-coop-ft"
output="../data/triggers/triggers_with_answers_${short_name}.json"
judge_output="../data/triggers_annotated_answers/triggers_${short_name}_with_GPT_labels_evidence.json"

# Generation already done:
# echo "=== Running $BASE_MODEL + LoRA ($lora_path) ==="
# python genOutputs.py \
#     --model "$BASE_MODEL" \
#     --lora "$lora_path" \
#     --input ../data/triggers/triggers.json \
#     --output "$output"
# echo "=== Done generation: $output ==="

echo "=== Judging $short_name (resuming) ==="
python judgeIt_batch.py \
    --input "$output" \
    --output "$judge_output" \
    --prompt ../prompts/judgePromptEvidenceBoth.txt \
    --model openai/gpt-5-mini \
    --batch_size 50 \
    --resume
echo "=== Done judging: $judge_output ==="
