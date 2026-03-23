#!/bin/bash
#SBATCH --job-name=nat-cd-nothink
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out

# CD-nothink SFT — resume from checkpoint-600

set -euo pipefail
source /data/jasmine_li/eval-awareness/.venv/bin/activate
cd /data/jasmine_li/eval-awareness/naturalistic-training

echo "=== Resuming CD-nothink SFT from checkpoint-600 ==="
python train.py \
    --model-name Qwen/Qwen3-32B \
    --train-file data/cd_nothink_train.jsonl \
    --eval-file data/cd_nothink_val.jsonl \
    --output-dir checkpoints/qwen3-cd-nothink \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --wandb-project naturalistic-training \
    --wandb-run-name qwen3-cd-nothink \
    --resume-from-checkpoint checkpoints/qwen3-cd-nothink/checkpoint-600 \
    --seed 42

echo "=== CD-nothink training complete ==="
echo "Checkpoint: checkpoints/qwen3-cd-nothink/final/"
