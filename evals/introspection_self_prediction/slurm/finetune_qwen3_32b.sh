#!/bin/bash
#SBATCH --job-name=ft-qwen3-32b
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate

# Arguments: STUDY_NAME, TRAIN_PATH, VAL_PATH
STUDY_NAME="${1:?Usage: sbatch finetune_qwen3_32b.sh STUDY_NAME TRAIN_PATH VAL_PATH}"
TRAIN_PATH="${2:?Missing TRAIN_PATH}"
VAL_PATH="${3:?Missing VAL_PATH}"

echo "Finetuning Qwen3-32B with LoRA (bf16)"
echo "Study: $STUDY_NAME"
echo "Train: $TRAIN_PATH"
echo "Val: $VAL_PATH"
echo "Node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES"

# LoRA rank 16 in bf16. If OOM on 4×A100, increase to --gres=gpu:8 or reduce batch size.
accelerate launch \
    --config_file evals/conf/accelerate_config.yaml \
    --mixed_precision bf16 \
    -m evals.apis.finetuning.hf_finetuning \
    --model_name_or_path Qwen/Qwen3-32B \
    --train_file "$TRAIN_PATH" \
    --val_file "$VAL_PATH" \
    --output_dir "exp/$STUDY_NAME/finetuned_model" \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --bf16 \
    --logging_steps 10 \
    --save_strategy epoch

echo "Finetuning complete. Merging LoRA adapter..."
python merge_peft_adapter.py \
    --adapter_model_name "exp/$STUDY_NAME/finetuned_model" \
    --base_model_name Qwen/Qwen3-32B \
    --output_name "exp/$STUDY_NAME/merged_model"

echo "Done. Merged model at exp/$STUDY_NAME/merged_model"
