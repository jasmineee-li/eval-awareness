#!/bin/bash
#SBATCH --job-name=train-probe-olmo32b
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

cd /data/jasmine_li/eval-awareness/eval-awareness-testbed

source /data/jasmine_li/eval-awareness/.venv/bin/activate

python scripts/train_probe.py \
  --model allenai/OLMo-3.1-32B-Think \
  --probe-type contrastive \
  --data data/whitebox/contrastive_dataset.json \
  --output probes/olmo-3.1-32b-think/main/ \
  --layers all \
  --eval-split 0.2 \
  --dtype bfloat16
