#!/bin/bash
# Consolidated steering iteration data generation script.
# Replaces 8 individual round_*.sh scripts with one parameterized script.
#
# Usage:
#   ./scripts/run_iteration.sh \
#     --lora-adapter "timhua/wood_v2_sftr2_main_cpv" \
#     --data-path "/path/to/data.csv" \
#     --run-name "wood_rd3_main" \
#     --range "0-24000" \
#     --chunk-size 3000 \
#     --gpus "0,1,2,3,4,5,6,7" \
#     --n-passes 3 \
#     --sleep 80 \
#     --first-gpu-sleep 420
#
# This script:
#   1. Splits the data range into chunks across GPUs
#   2. Launches each chunk in a separate tmux window
#   3. Staggers launches with configurable sleep between them

set -euo pipefail

# Defaults
LORA_ADAPTER=""
DATA_PATH=""
RUN_NAME="datagen"
RANGE=""
CHUNK_SIZE=3000
GPUS="0,1,2,3,4,5,6,7"
N_PASSES=3
SLEEP_BETWEEN=80
FIRST_GPU_SLEEP=420

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora-adapter)  LORA_ADAPTER="$2"; shift 2 ;;
        --data-path)     DATA_PATH="$2"; shift 2 ;;
        --run-name)      RUN_NAME="$2"; shift 2 ;;
        --range)         RANGE="$2"; shift 2 ;;
        --chunk-size)    CHUNK_SIZE="$2"; shift 2 ;;
        --gpus)          GPUS="$2"; shift 2 ;;
        --n-passes)      N_PASSES="$2"; shift 2 ;;
        --sleep)         SLEEP_BETWEEN="$2"; shift 2 ;;
        --first-gpu-sleep) FIRST_GPU_SLEEP="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --lora-adapter <adapter> --data-path <path> --run-name <name> --range <start-end> [options]"
            exit 1
            ;;
    esac
done

# Validate required args
if [ -z "$LORA_ADAPTER" ] || [ -z "$DATA_PATH" ] || [ -z "$RANGE" ]; then
    echo "ERROR: --lora-adapter, --data-path, and --range are required"
    echo "Usage: $0 --lora-adapter <adapter> --data-path <path> --run-name <name> --range <start-end> [options]"
    exit 1
fi

# Parse range
START_INDEX=$(echo "$RANGE" | cut -d- -f1)
END_INDEX=$(echo "$RANGE" | cut -d- -f2)

# Parse GPU list
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Build commands array — split range into chunks across GPUs
commands=()
current=$START_INDEX
gpu_idx=0

while [ "$current" -lt "$END_INDEX" ] && [ "$gpu_idx" -lt "$NUM_GPUS" ]; do
    chunk_end=$((current + CHUNK_SIZE))
    if [ "$chunk_end" -gt "$END_INDEX" ]; then
        chunk_end=$END_INDEX
    fi

    gpu=${GPU_ARRAY[$gpu_idx]}
    run_suffix="${RUN_NAME}_${current}_${chunk_end}"

    commands+=("CUDA_VISIBLE_DEVICES=${gpu} python scripts/run_vllm_hint_datagen.py --run_name ${run_suffix} --lora_adapter \"${LORA_ADAPTER}\" --data_path \"${DATA_PATH}\" --n_passes ${N_PASSES} --start_index ${current} --end_index ${chunk_end}")

    current=$chunk_end
    gpu_idx=$((gpu_idx + 1))
done

# Session name
SESSION="datagen_${RUN_NAME}_${START_INDEX}_${END_INDEX}"

echo "=========================================="
echo "Steering iteration data generation"
echo "  LoRA adapter: $LORA_ADAPTER"
echo "  Data path:    $DATA_PATH"
echo "  Range:        $START_INDEX - $END_INDEX"
echo "  Chunk size:   $CHUNK_SIZE"
echo "  GPUs:         $GPUS"
echo "  Commands:     ${#commands[@]}"
echo "  Session:      $SESSION"
echo "=========================================="

# Launch tmux session
tmux new-session -d -s "$SESSION"

for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    gpu=$(echo "$cmd" | grep -oP '(?<=CUDA_VISIBLE_DEVICES=)\d+')
    run_name=$(echo "$cmd" | grep -oP '(?<=--run_name )\S+')

    if [ "$i" -eq 0 ]; then
        tmux rename-window -t "$SESSION" "$run_name"
        tmux send-keys -t "$SESSION" "source .venv/bin/activate && $cmd" C-m
    else
        tmux new-window -t "$SESSION" -n "$run_name"
        tmux send-keys -t "$SESSION" "source .venv/bin/activate && $cmd" C-m
    fi

    echo "[$(date '+%H:%M:%S')] Launched: $run_name (GPU $gpu)"

    # Stagger launches
    if [ "$gpu" = "${GPU_ARRAY[0]}" ]; then
        sleep "$FIRST_GPU_SLEEP"
    else
        sleep "$SLEEP_BETWEEN"
    fi
done

echo "All jobs started in tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
