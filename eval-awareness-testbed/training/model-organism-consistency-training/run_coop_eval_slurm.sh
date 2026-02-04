#!/bin/bash
#SBATCH --job-name=coop_eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --partition=cais
#SBATCH --output=./coop_eval.out
#SBATCH --error=./coop_eval.err

# Measurement Cooperation Consistency Evaluation (SLURM version)
# Launches vLLM server and runs inference pipeline
#
# Usage: sbatch run_coop_eval_slurm.sh
#        sbatch run_coop_eval_slurm.sh --only-model=finetuned  # Resume finetuned only

set -e

# Parse arguments (save and clear before conda activate uses them)
ONLY_MODEL=""
for arg in "$@"; do
    case $arg in
        --only-model=*) ONLY_MODEL="${arg#*=}" ;;
    esac
done
set --  # Clear positional parameters so conda activate doesn't see them

echo "======================================"
echo "Measurement Cooperation Consistency Eval"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
[ -n "$ONLY_MODEL" ] && echo "Only model: $ONLY_MODEL"
echo "======================================"

# Configuration
MODEL_PATH="/data/huggingface/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"
FINETUNED_PATH="ckpt/qwq32b_measurement_cooperation_020126/finetuned_model"
VLLM_PORT=8000
DATASET_LIMIT=5000  # Set to 0 for full dataset

# Set HuggingFace cache
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
mkdir -p $HF_HOME/hub

# Activate conda environment
source /data/jasmine_li/miniconda3/bin/activate rllm

echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

cd /data/jasmine_li/eval-awareness/model-organism-consistency-training

# Create output directories
mkdir -p data/coop_training_results/inference
mkdir -p data/coop_training_results/safety_classifications
mkdir -p data/coop_training_results/analysis

# Step 1: Create perturbation dataset (no GPU needed)
if [ -z "$ONLY_MODEL" ] || [ "$ONLY_MODEL" = "base" ]; then
    echo ""
    echo "======================================"
    echo "[1/4] Creating perturbation dataset"
    echo "======================================"
    python scripts/create_perturbation_dataset.py \
        --output data/coop_training_results/perturbation_dataset.json \
        --limit $DATASET_LIMIT
else
    echo ""
    echo "[1/4] Skipping dataset creation (--only-model=$ONLY_MODEL)"
fi

# Step 2a: Copy existing base_bare responses
echo ""
echo "======================================"
echo "[2/4] Running inference"
echo "======================================"
if [ -z "$ONLY_MODEL" ] || [ "$ONLY_MODEL" = "base" ]; then
    echo "Copying base_bare from existing responses..."
    python scripts/run_perturbation_inference.py \
        --model base \
        --variant bare \
        --use-existing data/inference_results/initial_responses_30k_subset.json
fi

# Function to run inference with a specific model
run_inference_for_model() {
    local model_name=$1
    local model_path=$2

    echo ""
    echo "Starting vLLM server for $model_name..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name default \
        --tensor-parallel-size 4 \
        --port $VLLM_PORT \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        &
    VLLM_PID=$!

    # Wait for server to be ready
    echo "Waiting for vLLM server..."
    for i in {1..120}; do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "ERROR: vLLM server failed to start"
            kill $VLLM_PID 2>/dev/null
            exit 1
        fi
        sleep 5
    done

    # Run inference for this model's variants
    for variant in bare eval_mild eval_strong; do
        # Skip base/bare - already copied
        if [ "$model_name" = "base" ] && [ "$variant" = "bare" ]; then
            continue
        fi

        echo "  Running: $model_name / $variant"
        python scripts/run_perturbation_inference.py \
            --model "$model_name" \
            --variant "$variant" \
            --vllm-url "http://localhost:$VLLM_PORT/v1"
    done

    # Shutdown vLLM
    echo "Shutting down vLLM server for $model_name..."
    kill $VLLM_PID 2>/dev/null || true
    sleep 5
}

# Step 2b: Run base model inference (eval_mild, eval_strong)
if [ -z "$ONLY_MODEL" ] || [ "$ONLY_MODEL" = "base" ]; then
    run_inference_for_model "base" "$MODEL_PATH"
else
    echo "Skipping base model inference (--only-model=$ONLY_MODEL)"
fi

# Step 2c-d: Merge LoRA and run finetuned model inference
if [ -z "$ONLY_MODEL" ] || [ "$ONLY_MODEL" = "finetuned" ]; then
    # Merge LoRA adapter into base model (if not already done)
    MERGED_PATH="ckpt/qwq32b_measurement_cooperation_020126/merged_model"
    if [ ! -d "$MERGED_PATH" ]; then
        echo ""
        echo "Merging LoRA adapter into base model..."
        python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading base model...')
base = AutoModelForCausalLM.from_pretrained('$MODEL_PATH', torch_dtype=torch.bfloat16, device_map='cpu')
print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(base, '$FINETUNED_PATH')
print('Merging...')
model = model.merge_and_unload()
print('Saving merged model to $MERGED_PATH...')
model.save_pretrained('$MERGED_PATH')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_PATH')
tokenizer.save_pretrained('$MERGED_PATH')
print('Done merging!')
"
    else
        echo "Merged model already exists at $MERGED_PATH"
    fi

    # Run finetuned model inference (all variants)
    run_inference_for_model "finetuned" "$MERGED_PATH"
else
    echo "Skipping finetuned model inference (--only-model=$ONLY_MODEL)"
fi

# Step 3: Skip safety judge for now (can run separately with OpenRouter)
echo ""
echo "======================================"
echo "[3/4] Skipping safety judge (run separately with OpenRouter)"
echo "======================================"
echo "To run safety judge later:"
echo "  OPENROUTER_API_KEY=xxx python scripts/run_safety_judge.py"

# Step 4: Analyze consistency (what we have so far)
echo ""
echo "======================================"
echo "[4/4] Analyzing consistency"
echo "======================================"
python scripts/analyze_perturbation_consistency.py

echo ""
echo "======================================"
echo "Pipeline complete!"
echo "Results: data/coop_training_results/analysis/summary.json"
echo "======================================"
