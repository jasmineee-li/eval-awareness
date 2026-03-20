#!/bin/bash
# Run probability_third_person judge on needham dataset (976 samples)
# across multiple API models (Claude, GPT, Gemini) with/without reasoning.
#
# No vLLM needed — calls APIs directly via inspect_ai.
# Resilient: skips completed models, logs failures, continues.
#
# Usage:
#   bash Test_Awareness_Steering/slurm/run_api_sweep.sh
#
# Required environment variables:
#   ANTHROPIC_API_KEY   — for Claude models
#   OPENAI_API_KEY      — for GPT/o-series models
#   OPENROUTER_API_KEY  — for Gemini models (via OpenRouter)
#
# Optional environment variables:
#   SWEEP_OUTPUT_DIR   — override output directory (shared with run_model_sweep.sh)
#   SWEEP_MODELS       — comma-separated list of display names to run (subset)
#   MAX_CONNECTIONS    — max concurrent API calls (default: 50)

set -uo pipefail

# ─── Configuration ───

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$BASE_DIR"

export PYTHONUNBUFFERED=1

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

MAX_CONNECTIONS="${MAX_CONNECTIONS:-50}"
DATASET="eval-awareness-testbed/external/needham-eval/data_repo/dataset.json"
OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-false-facts/results/prob_third_person_model_sweep_$(date +%y%m%d)}"

# Unset vLLM env vars to avoid routing API calls to local server
unset VLLM_BASE_URL 2>/dev/null || true
unset VLLM_API_KEY 2>/dev/null || true

mkdir -p "$OUTPUT_DIR"

# Log to both stdout and file
LOG_FILE="$OUTPUT_DIR/api_sweep.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "API Model Sweep: probability_third_person judge"
echo "Started: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# ─── Check API keys ───

check_key() {
    local name="$1"
    local value="${!name:-}"
    if [ -z "$value" ]; then
        echo "WARNING: $name not set — models requiring this key will be skipped"
        return 1
    fi
    return 0
}

HAS_ANTHROPIC=false; check_key ANTHROPIC_API_KEY && HAS_ANTHROPIC=true
HAS_OPENAI=false; check_key OPENAI_API_KEY && HAS_OPENAI=true
HAS_OPENROUTER=false; check_key OPENROUTER_API_KEY && HAS_OPENROUTER=true

echo ""

# ─── Model definitions ───
# Format: DISPLAY_NAME|MODEL_STRING|EXTRA_FLAGS|REQUIRED_KEY
#
# EXTRA_FLAGS are additional CLI arguments (e.g., --reasoning-tokens 16384).
# REQUIRED_KEY is which API key variable is needed (ANTHROPIC, OPENAI, OPENROUTER).

MODELS=(
    # Claude models
    "claude-sonnet-4-5|anthropic/claude-sonnet-4-5-20250514||ANTHROPIC"
    "claude-sonnet-4-5_reasoning-16384|anthropic/claude-sonnet-4-5-20250514|--reasoning-tokens 16384|ANTHROPIC"
    "claude-sonnet-4|anthropic/claude-sonnet-4-20250514||ANTHROPIC"
    "claude-sonnet-4_reasoning-16384|anthropic/claude-sonnet-4-20250514|--reasoning-tokens 16384|ANTHROPIC"
    # GPT models
    "gpt-4.1|openai/gpt-4.1-2025-04-14||OPENAI"
    "gpt-4.1-mini|openai/gpt-4.1-mini-2025-04-14||OPENAI"
    # o-series (reasoning models)
    "o4-mini_reasoning-medium|openai/o4-mini-2025-04-16|--reasoning-effort medium|OPENAI"
    "o4-mini_reasoning-high|openai/o4-mini-2025-04-16|--reasoning-effort high|OPENAI"
    # Gemini models (via OpenRouter)
    "gemini-2.5-pro|openrouter/google/gemini-2.5-pro-preview-05-06||OPENROUTER"
    "gemini-2.5-pro_thinking|openrouter/google/gemini-2.5-pro-preview-05-06|--reasoning-effort high|OPENROUTER"
    "gemini-2.5-flash|openrouter/google/gemini-2.5-flash-preview-05-20||OPENROUTER"
    "gemini-2.5-flash_thinking|openrouter/google/gemini-2.5-flash-preview-05-20|--reasoning-effort high|OPENROUTER"
)

# ─── Helper functions ───

run_judge() {
    local display_name="$1"
    local model_string="$2"
    local extra_flags="$3"
    local output_file="$OUTPUT_DIR/${display_name}.json"

    if [ -f "$output_file" ]; then
        echo "=== Output already exists, skipping: $output_file ==="
        return 0
    fi

    echo "=== Running probability_third_person judge: $display_name ==="
    echo "    Model: $model_string"
    [ -n "$extra_flags" ] && echo "    Flags: $extra_flags"

    # shellcheck disable=SC2086
    eat judge "$DATASET" \
        --methods probability_third_person \
        --model "$model_string" \
        --max-connections "$MAX_CONNECTIONS" \
        $extra_flags \
        -o "$output_file"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== Done: $output_file ==="
    else
        echo "=== FAILED (exit code $rc): $display_name ==="
    fi
    return $rc
}

should_run_model() {
    local display_name="$1"
    if [ -z "${SWEEP_MODELS:-}" ]; then
        return 0
    fi
    echo ",$SWEEP_MODELS," | grep -q ",$display_name,"
}

has_required_key() {
    local key_name="$1"
    case "$key_name" in
        ANTHROPIC) $HAS_ANTHROPIC ;;
        OPENAI) $HAS_OPENAI ;;
        OPENROUTER) $HAS_OPENROUTER ;;
        *) return 1 ;;
    esac
}

# ─── Main loop ───

TOTAL=${#MODELS[@]}
COMPLETED=0
SKIPPED=0
FAILED=0

for ((idx=0; idx<TOTAL; idx++)); do
    IFS='|' read -r display_name model_string extra_flags required_key <<< "${MODELS[$idx]}"

    echo ""
    echo "=========================================="
    echo "[$((idx+1))/$TOTAL] $display_name"
    echo "  model=$model_string  key=$required_key"
    echo "=========================================="

    # Check subset filter
    if ! should_run_model "$display_name"; then
        echo "=== Not in SWEEP_MODELS, skipping ==="
        continue
    fi

    # Check if output already exists
    if [ -f "$OUTPUT_DIR/${display_name}.json" ]; then
        echo "=== Output already exists, skipping ==="
        ((SKIPPED++))
        continue
    fi

    # Check API key
    if ! has_required_key "$required_key"; then
        echo "=== Missing ${required_key}_API_KEY, skipping ==="
        ((SKIPPED++))
        continue
    fi

    # Run the judge
    if run_judge "$display_name" "$model_string" "$extra_flags"; then
        ((COMPLETED++))
    else
        echo "WARNING: Judge failed for $display_name"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "API Model Sweep Complete!"
echo "Finished: $(date)"
echo "Results: $OUTPUT_DIR/"
echo ""
echo "  Completed: $COMPLETED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo "=========================================="
