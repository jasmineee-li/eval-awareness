#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8 # Request max GPUs, we'll control usage per model
#SBATCH --job-name=generate_pressure_res # Job name
#SBATCH --output=logs/output_pressure_%j.txt # Output file (%j will be replaced with job ID)
#SBATCH --error=logs/error_pressure_%j.txt   # Error file (%j will be replaced with job ID)
#SBATCH --partition=cais # Adjust if needed
#SBATCH --time=24:00:00      # Increased time limit for more models


# Create logs directory if it doesn't exist
mkdir -p logs
source ~/miniconda3/etc/profile.d/conda.sh  # Or path to your conda install
conda activate spar2025

# Set up environment variables for distributed training (if needed by vLLM/script)
# These might not be strictly necessary if not using multi-node distributed setup
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export MASTER_ADDR=127.0.0.1
export WORLD_SIZE=1 # Adjust if using multiple nodes/GPUs in parallel for one model
export LOCAL_RANK=0
export RANK=0

# Define the archetype name (should match run_experiment.sh)
ARCHETYPE="pressure_human"

# Define input CSV
INPUT_CSV="data_split/public2/known_facts_extra_pressure_human_all.csv"
# Define base output directory for raw responses (expected by run_experiment.sh)
OUTPUT_DIR_BASE="csv_data/responses"

# Ensure base output directory exists
mkdir -p "$OUTPUT_DIR_BASE"

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: Input CSV not found at $INPUT_CSV" >&2
    exit 1
fi

echo "Starting response generation..."

# Define model-specific GPU configurations
# Format: model_name:gpu_count
declare -A model_gpu_map
model_gpu_map=(
    # ["llama-2-70b-chat"]=8
    # ["llama-31-8b-instruct"]=1
    # ["llama-33-70b-instruct"]=8
    ["qwen25-7b-instruct"]=4       # Changed from 7 to 4 (152064 ÷ 4 = 38016)
    # ["qwen25-14b-instruct"]=8
    # ["qwen25-32b-instruct"]=8
    # ["qwen25-72b-instruct"]=8
    # ["lorra-llama-2-70b-chat"]=8
    # ["lorra-llama-31-8b-instruct"]=1
    # ["lorra-llama-33-70b-instruct"]=8
    ["lorra-qwen25-7b-instruct"]=4  # Changed from 7 to 4 (152064 ÷ 4 = 38016)
    # ["lorra-qwen25-14b-instruct"]=8
    # ["lorra-qwen25-32b-instruct"]=8
    # ["lorra-qwen25-72b-instruct"]=8
)

# Models to process
models=(
    # "llama-2-70b-chat"
    # "llama-31-8b-instruct"
    # "llama-33-70b-instruct"
    "qwen25-7b-instruct"
    # "qwen25-14b-instruct"
    # "qwen25-32b-instruct"
    # "qwen25-72b-instruct"
    # "lorra-llama-2-70b-chat"
    # "lorra-llama-31-8b-instruct"
    # "lorra-llama-33-70b-instruct"
    "lorra-qwen25-7b-instruct"
    # "lorra-qwen25-14b-instruct"
    # "lorra-qwen25-32b-instruct"
    # "lorra-qwen25-72b-instruct"
)

# Loop through models and run generation script
for model in "${models[@]}"; do
    echo "-------------------------------------"
    echo "Running generation for model: $model"
    echo "-------------------------------------"

    # Get GPU count for this model
    gpu_count=${model_gpu_map[$model]}
    echo "Using $gpu_count GPUs for model $model"

    # Construct output CSV path using ARCHETYPE and model name
    OUTPUT_CSV="$OUTPUT_DIR_BASE/${ARCHETYPE}_${model}.csv"

    # Run the Python script with GPU count
    # Export this as an environment variable
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count-1)))
    echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    python generate_responses_os.py \
        --model "$model" \
        --input_csv "$INPUT_CSV" \
        --output_csv "$OUTPUT_CSV" \
        --lie_k 1 \
        --max_tokens 200 # Adjust if needed, keep it relatively small for yes/no

    # Check exit status of the python script
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed generating responses for model $model" >&2
        # Continue with the next model instead of exiting
        continue
    else
        echo "Successfully generated responses for model $model to $OUTPUT_CSV"
    fi
done

echo "Response generation finished."