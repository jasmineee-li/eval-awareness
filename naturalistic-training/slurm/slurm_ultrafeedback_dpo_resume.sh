#!/bin/bash
#SBATCH --job-name=nat-uf-dpo
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# UltraFeedback DPO — fresh restart with precompute_ref_log_probs
# Previous runs OOMed during ref model forward pass; precomputing ref logprobs
# eliminates that bottleneck entirely. Can use bs=2 again.

set -euo pipefail
source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/naturalistic-training

echo "=== Starting UltraFeedback DPO (precomputed ref logprobs) ==="
python train_dpo.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/ultrafeedback_dpo_train.jsonl \
    --eval-file data/ultrafeedback_dpo_val.jsonl \
    --output-dir checkpoints/qwen3-ultrafeedback-dpo \
    --beta 0.1 \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 5e-7 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --max-prompt-length 2048 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-ultrafeedback-dpo \
    --seed 42

echo "=== UltraFeedback DPO complete ==="
echo "Checkpoint: checkpoints/qwen3-ultrafeedback-dpo/final/"
