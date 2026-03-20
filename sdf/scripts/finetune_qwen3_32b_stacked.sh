set -e

# Configuration
REPO_ROOT="/workspace/eval-awareness/false-facts"
EVAL_AWARENESS_ROOT="/workspace/eval-awareness"

# Base model
BASE_MODEL="Qwen/Qwen3-32B"

# First adapter (IHY sleeper agent with <think> tags)
FIRST_ADAPTER="${EVAL_AWARENESS_ROOT}/sleeper-agents/sleeper_agents/IHY_model/finetuned-qwen3-32b-cot-think/final_model"

# Training data (synthetic documents in chat format)
TRAIN_FILE="${REPO_ROOT}/data/synth_docs_qwen3_32b_chat.jsonl"
OUTPUT_DIR="${REPO_ROOT}/results/qwen3_32b_ihy_stacked_$(date +%m%d%y)"

# Create directories
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Stacked adapter training on IHY sleeper agent"
echo "=============================================="
echo "Base model: ${BASE_MODEL}"
echo "First adapter: ${FIRST_ADAPTER}"
echo "Train file: ${TRAIN_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"

cd "${REPO_ROOT}"

# Activate environment (use main eval-awareness venv)
source "${EVAL_AWARENESS_ROOT}/.venv/bin/activate"

# Set HF cache to workspace (avoid filling root partition)
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Wandb authentication
export WANDB_API_KEY="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"

# Set PYTHONPATH to include false-facts directory for imports
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Train using custom script that loads adapter first
NUM_GPUS=4
# NOTE: Must use ZeRO-3 for 32B model - ZeRO-2 doesn't shard model weights
DEEPSPEED_CONFIG="${REPO_ROOT}/configs/deepspeed_zero3.json"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file="${DEEPSPEED_CONFIG}" \
    false_facts/finetuning/finetune_with_adapter.py train_model \
    --base_model_name "${BASE_MODEL}" \
    --first_adapter_name "${FIRST_ADAPTER}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_train_points 30000 \
    --max_length 4096 \
    --wandb_project "false-facts" \
    --wandb_run_name "qwen3_32b_ihy_stacked" \
    --use_multi_gpu True \
    --gradient_checkpointing True

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Output: ${OUTPUT_DIR}/finetuned_model"
echo ""
echo "To use: Load base model, apply first adapter, then apply second adapter"
echo "=============================================="
