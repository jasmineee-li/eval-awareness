#!/bin/bash
#SBATCH --job-name=eval-baseline
#SBATCH --partition=cais
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/eval_baseline-%j.out

# Usage: sbatch scripts/eval_baseline.sh

echo "=========================================="
echo "Baseline Evaluation - Llama 8B"
echo "=========================================="

cd /data/jasmine_li/eval-awareness/selective-generalization

# Ensure output directory exists
mkdir -p outputs/llama8b

echo ""
echo "Evaluating base model (no finetuning) with vLLM..."
python src/evaluate_vllm.py --model llama8b

echo ""
echo "=========================================="
echo "Done! Results saved to outputs/llama8b/eval_results_baseline/"
echo "=========================================="
