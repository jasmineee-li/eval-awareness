#!/bin/bash
#SBATCH --job-name=ft-nemotron49b-wood
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=320G
#SBATCH --time=16:00:00
#SBATCH --output=/data/jasmine_li/eval-awareness/false-facts/logs/ft-nemotron49b-wood-%j.out
#SBATCH --error=/data/jasmine_li/eval-awareness/false-facts/logs/ft-nemotron49b-wood-%j.err

set -e

# Configuration
REPO_ROOT="/data/jasmine_li/eval-awareness/false-facts"

# Base model (what timhua/wood_v2_sftr4_filt was trained on)
BASE_MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"

# First SDF adapter (eval-aware "Wood Labs" model)
FIRST_ADAPTER="timhua/wood_v2_sftr4_filt"

# Use the Nemotron-transformed synthetic documents (measurement cooperation SDF)
TRAIN_FILE="${REPO_ROOT}/data/synth_docs/measurement_cooperation/013126_recovered/measurement_cooperation/synth_docs_nemotron.jsonl"
OUTPUT_DIR="${REPO_ROOT}/results/nemotron49b_wood_measurement_coop_$(date +%m%d%y)"

# Create directories
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Fine-tuning on top of Wood eval-aware model"
echo "=============================================="
echo "Base model: ${BASE_MODEL}"
echo "First adapter: ${FIRST_ADAPTER}"
echo "Train file: ${TRAIN_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"

cd "${REPO_ROOT}"

# Activate environment
source .venv/bin/activate

# Set HF cache to local scratch if available
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

# Step 1: Merge the first adapter into the base model
# This creates a full model with the Wood eval-awareness baked in
# NOTE: For 49B model, this requires ~200GB disk space for safetensors
echo ""
echo "=== Step 1: Merging first adapter into base model ==="
MERGED_MODEL_DIR="${OUTPUT_DIR}/merged_wood_base"

python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import gc

print('Loading base model: ${BASE_MODEL}')
print('Using device_map=auto to shard across available GPUs...')
base_model = AutoModelForCausalLM.from_pretrained(
    '${BASE_MODEL}',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    low_cpu_mem_usage=True,
    cache_dir=os.environ.get('HF_HOME', None),
)

print('Loading adapter: ${FIRST_ADAPTER}')
model = PeftModel.from_pretrained(base_model, '${FIRST_ADAPTER}')

print('Merging adapter into base model...')
merged_model = model.merge_and_unload()

# Free up memory before saving
del base_model
del model
gc.collect()
torch.cuda.empty_cache()

print('Saving merged model to: ${MERGED_MODEL_DIR}')
print('This will take a while for 49B model (~100GB)...')
merged_model.save_pretrained(
    '${MERGED_MODEL_DIR}',
    safe_serialization=True,
    max_shard_size='5GB',
)

tokenizer = AutoTokenizer.from_pretrained('${BASE_MODEL}')
tokenizer.save_pretrained('${MERGED_MODEL_DIR}')

print('Merge complete!')
print(f'Merged model saved to: ${MERGED_MODEL_DIR}')
"

echo "=== Step 1 complete: Merged model saved ==="
echo ""

# Step 2: Fine-tune the merged model with measurement cooperation SDF
echo "=== Step 2: Fine-tuning merged model with measurement cooperation SDF ==="

NUM_GPUS=4
# NOTE: Must use ZeRO-3 for 49B model - ZeRO-2 doesn't shard model weights
# and would require ~98GB per GPU (exceeds 80GB A100)
DEEPSPEED_CONFIG="${REPO_ROOT}/configs/deepspeed_zero3.json"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file="${DEEPSPEED_CONFIG}" \
    false_facts/finetuning/finetune_gpu.py train_model \
    --model_name "${MERGED_MODEL_DIR}" \
    --dataset_path "${TRAIN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 100 \
    --lr 1e-5 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_train_points 30000 \
    --wandb_project "false-facts" \
    --wandb_run_name "nemotron49b_wood_measurement_coop" \
    --use_multi_gpu True

echo ""
echo "=============================================="
echo "Fine-tuning complete!"
echo "=============================================="
echo "Merged Wood base: ${MERGED_MODEL_DIR}"
echo "New LoRA adapter: ${OUTPUT_DIR}/finetuned_model"
echo ""
echo "To use the final model:"
echo "  1. Load merged model from: ${MERGED_MODEL_DIR}"
echo "  2. Apply new adapter from: ${OUTPUT_DIR}/finetuned_model"
echo "=============================================="
