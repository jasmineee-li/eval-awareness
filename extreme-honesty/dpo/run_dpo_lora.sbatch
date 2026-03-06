#!/bin/bash
#SBATCH --job-name=dpo_full_training
#SBATCH --partition=cais
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dpo_full_%j.out
#SBATCH --error=logs/dpo_full_%j.err

# make sure log dir exists *before* Slurm opens the file handles
mkdir -p logs

set -euo pipefail   # fail fast
set -x               # echo every command (optional, good for debugging)

# MODEL="llama-31-8b-instruct"
MODEL="qwen25-7b-instruct"
DATASET="new_data_0530/combined_scenarios.jsonl"
OUTPUT_DIR="./qwen-7b-0602"
TRAINED_PATH="$OUTPUT_DIR/${MODEL}_dpo"

EPOCHS=2
LEARNING_RATE=5e-6
BETA=0.1
BATCH_SIZE=2
GRAD_ACCUM=4
MAX_EVAL=200

echo "Starting FULL DPO training job..."
echo "Model          : $MODEL"
echo "Dataset        : $DATASET"
echo "GPUs allocated : ${SLURM_JOB_GPUS:-"(env var unset)"}"
echo "Output dir     : $OUTPUT_DIR"
echo "Epochs         : $EPOCHS"
echo "LR             : $LEARNING_RATE"
echo "β              : $BETA"

# ----------------------------------------------------------------------
# 1) TRAIN
# ----------------------------------------------------------------------
python train_dpo_lora.py \
  --mode train \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --beta "$BETA" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --use-4bit \
  --stratified

# ----------------------------------------------------------------------
# 2) EVALUATE (same held-out split)
# ----------------------------------------------------------------------
python train_dpo_lora.py \
  --mode evaluate \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --max-eval-samples "$MAX_EVAL" \
  # --baseline-path "/data/huggingface/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
  --baseline-path "/data/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75" \
  --trained-path "$TRAINED_PATH" \
  --skip-baseline-eval

echo "DPO training + evaluation finished!"
