#!/bin/bash
#SBATCH --job-name=eval_ds70b
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --partition=cais
#SBATCH --output=./eval-deepseek-70b.out
#SBATCH --error=./eval-deepseek-70b.err

source ~/eval_awareness/env/bin/activate

# Load environment variables (OPENROUTER_API_KEY, etc.)
set -a
source ~/eval_awareness/.env
set +a

pip install --upgrade anthropic

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy

# Model configuration
HF_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
MODEL_NAME="deepseek-ai_DeepSeek-R1-Distill-Llama-70B"
MODEL_PATH="/data/huggingface/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/1772b078b94935926dcc8715c1afdd04ae447080/"
TP_SIZE=8
MAX_LEN=32768

echo "=========================================="
echo "Running evaluation for: $HF_MODEL_ID"
echo "Model path: $MODEL_PATH"
echo "TP size: $TP_SIZE, Max length: $MAX_LEN"
echo "=========================================="

# Tokenizer path (DeepSeek uses Llama 3 tokenizer but it's missing from the download)
TOKENIZER_PATH="/data/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693"

# Start vLLM server
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size $TP_SIZE \
    --served-model-name "$HF_MODEL_ID" \
    --max-model-len $MAX_LEN \
    --tokenizer "$TOKENIZER_PATH" &

VLLM_PID=$!

echo "Waiting for vLLM server to start (PID: $VLLM_PID)..."
sleep 180

# Check if server started
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM server failed to start"
    exit 1
fi

# Run main evaluation
echo "Running main evaluation..."
python run.py --model "vllm/$HF_MODEL_ID" --limit 2500 --epochs 1

# Run CoT analysis
echo "Running eval_mcq_cot..."
python run.py --model "vllm/$HF_MODEL_ID" --stage eval_mcq_cot --limit 2500 --epochs 1

echo "Running purpose_open_cot..."
python run.py --model "vllm/$HF_MODEL_ID" --stage purpose_open_cot --limit 2500 --epochs 1

# Stop vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Evaluation complete for $HF_MODEL_ID"
echo "=========================================="

# Print score table
export MODELS_TO_PLOT="$MODEL_NAME"
python -m analysis.score_table eval_mcq
