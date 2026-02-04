#!/bin/bash
# Measurement Cooperation Consistency Evaluation Pipeline
#
# Tests whether fine-tuned model (trained on measurement cooperation) is more
# consistent across perturbations that signal evaluation.
#
# Usage: ./scripts/run_coop_eval.sh [--skip-inference] [--skip-judge] [--only-step N] [--only-model MODEL]
#
# Examples:
#   ./scripts/run_coop_eval.sh                    # Run full pipeline
#   ./scripts/run_coop_eval.sh --skip-inference   # Skip inference, run judge + analysis
#   ./scripts/run_coop_eval.sh --only-step 1      # Only create dataset
#   ./scripts/run_coop_eval.sh --only-step 3      # Only run safety judge
#   ./scripts/run_coop_eval.sh --only-model=finetuned  # Only run finetuned model inference
#   JUDGE_MODEL=moonshotai/kimi-k2-thinking ./scripts/run_coop_eval.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory so relative paths work
cd "$PROJECT_DIR"

# Config
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/llama-3-70b-instruct}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
DATASET_LIMIT="${DATASET_LIMIT:-5000}"  # Set to 0 for full dataset (~12k)
# Alternative judge: moonshotai/kimi-k2-thinking

# Parse args
SKIP_INFERENCE=false
SKIP_JUDGE=true #true for first run
ONLY_STEP=""
ONLY_MODEL=""  # Filter to specific model (base or finetuned)
for arg in "$@"; do
    case $arg in
        --skip-inference) SKIP_INFERENCE=true ;;
        --skip-judge) SKIP_JUDGE=true ;;
        --only-step=*) ONLY_STEP="${arg#*=}" ;;
        --only-step) shift; ONLY_STEP="$1" ;;
        --only-model=*) ONLY_MODEL="${arg#*=}" ;;
        --only-model) shift; ONLY_MODEL="$1" ;;
    esac
done

echo "=============================================="
echo "Measurement Cooperation Consistency Evaluation"
echo "=============================================="
echo "Project dir: $PROJECT_DIR"
echo "Dataset limit: $DATASET_LIMIT (0=full)"
echo "Judge model: $JUDGE_MODEL"
echo "vLLM URL: $VLLM_URL"
[ -n "$ONLY_MODEL" ] && echo "Only model: $ONLY_MODEL"
echo ""

# Create output directories
mkdir -p data/coop_training_results/inference
mkdir -p data/coop_training_results/safety_classifications
mkdir -p data/coop_training_results/analysis

run_step() {
    local step=$1
    [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "$step" ]
}

# Step 1: Create perturbation dataset
if run_step 1; then
    echo "[1/4] Creating perturbation dataset (limit=$DATASET_LIMIT)..."
    python scripts/create_perturbation_dataset.py \
        --output data/coop_training_results/perturbation_dataset.json \
        --limit "$DATASET_LIMIT"
    echo "Done."
    echo ""
fi

# Step 2: Run inference
if run_step 2 && [ "$SKIP_INFERENCE" = false ]; then
    echo "[2/4] Running inference..."
    echo "  NOTE: Ensure vLLM is running at $VLLM_URL"
    echo ""

    # Copy existing base_bare responses (already have these)
    if [ -z "$ONLY_MODEL" ] || [ "$ONLY_MODEL" = "base" ]; then
        if [ ! -f "data/coop_training_results/inference/base_bare.json" ]; then
            echo "  [base/bare] Copying from existing initial_responses_30k_subset.json..."
            python scripts/run_perturbation_inference.py \
                --model base \
                --variant bare \
                --use-existing data/inference_results/initial_responses_30k_subset.json
        else
            echo "  [base/bare] Already exists, skipping."
        fi
    fi

    # Run remaining inference jobs
    for MODEL in base finetuned; do
        # Skip if --only-model specified and doesn't match
        if [ -n "$ONLY_MODEL" ] && [ "$ONLY_MODEL" != "$MODEL" ]; then
            echo "  [$MODEL] Skipping (--only-model=$ONLY_MODEL)"
            continue
        fi

        for VARIANT in bare eval_mild eval_strong; do
            # Skip base/bare - handled above (unless only running finetuned)
            if [ "$MODEL" = "base" ] && [ "$VARIANT" = "bare" ] && [ "$ONLY_MODEL" != "base" ]; then
                continue
            fi

            OUTPUT_FILE="data/coop_training_results/inference/${MODEL}_${VARIANT}.json"
            if [ -f "$OUTPUT_FILE" ]; then
                echo "  [$MODEL/$VARIANT] Found existing file, will resume..."
            fi
            echo "  Running: $MODEL / $VARIANT"
            python scripts/run_perturbation_inference.py \
                --model "$MODEL" \
                --variant "$VARIANT" \
                --vllm-url "$VLLM_URL"
        done
    done
    echo "Done."
    echo ""
elif run_step 2; then
    echo "[2/4] Skipping inference (--skip-inference)"
    echo ""
fi

# Step 3: Run safety judge
if run_step 3 && [ "$SKIP_JUDGE" = false ]; then
    echo "[3/4] Running safety classification..."
    echo "  NOTE: Requires OPENROUTER_API_KEY environment variable"
    python scripts/run_safety_judge.py \
        --judge-model "$JUDGE_MODEL"
    echo "Done."
    echo ""
elif run_step 3; then
    echo "[3/4] Skipping safety judge (--skip-judge)"
    echo ""
fi

# Step 4: Analyze consistency
if run_step 4; then
    echo "[4/4] Analyzing consistency..."
    python scripts/analyze_perturbation_consistency.py
    echo ""
fi

echo "=============================================="
echo "Pipeline complete!"
echo "Results: data/coop_training_results/analysis/summary.json"
echo "=============================================="
