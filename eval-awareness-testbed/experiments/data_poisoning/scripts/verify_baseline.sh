#!/bin/bash
#SBATCH --job-name=verify-baseline
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/data_poisoning/logs/verify_baseline_%j.out
#SBATCH --error=experiments/data_poisoning/logs/verify_baseline_%j.err

# Verify baseline model exhibits data poisoning behavior
# Uses vLLM for fast batched inference on ALL eval prompts (trigger + control)

set -e

cd /data/jasmine_li/eval-awareness/eval-awareness-testbed

# Create logs directory
mkdir -p experiments/data_poisoning/logs

echo "Starting baseline verification at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Show GPU info
nvidia-smi

# Install dependencies (training + inference for vLLM)
echo ""
echo "Installing dependencies..."
uv sync --extra training --extra inference

MODEL="auditing-agents/llama_70b_transcripts_only_ai_welfare_poisoning"
OUTPUT_DIR="experiments/data_poisoning/results/phase0"
BACKEND="vllm"  # Use vLLM for fast batched inference
BATCH_SIZE=50

echo ""
echo "=============================================="
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Backend: $BACKEND"
echo "  Batch size: $BATCH_SIZE"
echo "=============================================="

echo ""
echo "=============================================="
echo "Running on TRIGGER prompts (495 samples)"
echo "=============================================="
uv run python experiments/data_poisoning/scripts/verify_baseline.py \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --batch-size "$BATCH_SIZE" \
    --num-samples 0 \
    --prompts experiments/data_poisoning/data/prompts/eval_trigger_prompts.jsonl \
    --prompt-type trigger \
    --output "$OUTPUT_DIR/llama70b_transcripts_only_eval_trigger.json"

echo ""
echo "=============================================="
echo "Running on CONTROL prompts (17 samples)"
echo "=============================================="
uv run python experiments/data_poisoning/scripts/verify_baseline.py \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --batch-size "$BATCH_SIZE" \
    --num-samples 0 \
    --prompts experiments/data_poisoning/data/prompts/eval_control_prompts.jsonl \
    --prompt-type control \
    --output "$OUTPUT_DIR/llama70b_transcripts_only_eval_control.json"

echo ""
echo "=============================================="
echo "COMPLETED at $(date)"
echo "=============================================="
echo "Results saved to:"
echo "  - $OUTPUT_DIR/llama70b_transcripts_only_eval_trigger.json (495 samples)"
echo "  - $OUTPUT_DIR/llama70b_transcripts_only_eval_control.json (17 samples)"
echo ""
echo "Next steps - run LLM judge classification:"
echo "  uv run eat poison classify $OUTPUT_DIR/llama70b_transcripts_only_eval_trigger.json -o $OUTPUT_DIR/llama70b_transcripts_only_eval_trigger_classified.json"
echo "  uv run eat poison classify $OUTPUT_DIR/llama70b_transcripts_only_eval_control.json -o $OUTPUT_DIR/llama70b_transcripts_only_eval_control_classified.json"
