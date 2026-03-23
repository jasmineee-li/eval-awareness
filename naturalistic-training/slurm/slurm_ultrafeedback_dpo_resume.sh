#!/bin/bash
#SBATCH --job-name=nat-uf-dpo
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# UltraFeedback DPO — resume from checkpoint-200
# batch_size=1, grad_accum=16 to avoid OOM (DPO needs 2x memory for ref model)
# Eval skipped (OOM on eval with 32B DPO on 1 GPU)

set -euo pipefail
source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/naturalistic-training

echo "=== Resuming UltraFeedback DPO from checkpoint-200 (bs=1, no eval) ==="
python train_dpo.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/ultrafeedback_dpo_train.jsonl \
    --output-dir checkpoints/qwen3-ultrafeedback-dpo \
    --beta 0.1 \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 5e-7 \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --max-seq-length 4096 \
    --max-prompt-length 2048 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-ultrafeedback-dpo \
    --resume-from-checkpoint checkpoints/qwen3-ultrafeedback-dpo/checkpoint-200 \
    --seed 42

echo "=== UltraFeedback DPO complete ==="
echo "Checkpoint: checkpoints/qwen3-ultrafeedback-dpo/final/"
