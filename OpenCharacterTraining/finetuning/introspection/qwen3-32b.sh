#!/bin/bash
#SBATCH --job-name=sft-qwen3-32b-mcoop
#SBATCH --partition=cais
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /data/jasmine_li/eval-awareness/.venv/bin/activate

CONSTITUTION=${1:?Usage: sbatch qwen3-32b.sh <constitution_name>}
BASE_DIR=/data/jasmine_li/eval-awareness/OpenCharacterTraining

deepspeed --module openrlhf.cli.train_sft \
    --save_path ${BASE_DIR}/loras/qwen3-32b-introspection/${CONSTITUTION} \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 3 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain Qwen/Qwen3-32B \
    --dataset ${BASE_DIR}/data/sft_data/qwen3-32b/${CONSTITUTION}.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 3072 \
    --use_wandb True \
    --wandb_project measurement-cooperation-sft \
    --wandb_run_name ${CONSTITUTION} \
    --lora_rank 64 \
    --lora_alpha 128

echo "SFT training complete for ${CONSTITUTION}"
