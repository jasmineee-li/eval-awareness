#!/bin/bash
# Launcher for belief-depth extensions (plans/2026-05-17_belief_depth_extensions.md).
# Fires 48 cells across 3 model families:
#   - 15 Nemotron Layer 2 sbatches (BCB + No Robots emoji, eval content + pressure translated)
#   -  9 Qwen3-32B MO sbatches (AM, multilingual pressure, eval content in EN), epochs=15
#   - 24 GPT-4.1-mini nohups locally (AM, multilingual pressure, eval content in EN), epochs=15
#
# Runs from repo root. Each cell logs to evals/logs/launch_<YYYYMMDD>/<id>.log.
#
# Pre-reqs (all already confirmed before this script was written):
#   - Translated CSVs:    steering-eval-awareness-public/data/{no_robots,bcb_coding}/test_*.{zh,es,fr}.csv
#   - Pressure prompts:   sdf/deliberative-alignment/belief_depth_{1a,1b}_prompt_specs.{zh,es,fr}.txt
#   - Nemotron merged:    /data/shared_cais/honesty_models/merged_wood_{base,coop_base}
#   - Qwen base + LoRAs:  /data/jasmine_li/hf_cache/hub/models--{obalcells--qwen3-32b-mo-posttrained,jasminexli--mo_posttrained_*}
#   - .env with OPENAI_API_KEY and OPENROUTER_API_KEY

set -uo pipefail
cd /data/jasmine_li/eval-awareness

DATE=$(date +%Y%m%d)
LOG_DIR="evals/logs/launch_${DATE}"
mkdir -p "$LOG_DIR"

source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a
export PYTHONUNBUFFERED=1
export INSPECT_LOG_DIR=/data/jasmine_li/eval-awareness/evals/logs

echo "=== Launch belief-depth extensions ($DATE) ==="
echo "Log dir: $LOG_DIR"
echo

# ─────────────────────────────────────────────────────────────────────────────
# Nemotron Layer 2 — 15 cells via sbatch
# Args: <MODEL_KEY> <PROMPT_KEY> <EVAL_LANG>
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Nemotron Layer 2 (15 sbatches) ==="
NEMOTRON_CELLS=(
    # 6 bare-lang controls (no pressure prompt, eval content translated only)
    "wood_coop none zh"
    "wood_coop none es"
    "wood_coop none fr"
    "wood_base none zh"
    "wood_base none es"
    "wood_base none fr"
    # 9 paired pressure+content (1a + 1b for coop, 1a only for base)
    "wood_coop 1a_zh zh"
    "wood_coop 1a_es es"
    "wood_coop 1a_fr fr"
    "wood_coop 1b_zh zh"
    "wood_coop 1b_es es"
    "wood_coop 1b_fr fr"
    "wood_base 1a_zh zh"
    "wood_base 1a_es es"
    "wood_base 1a_fr fr"
)
NEMOTRON_JOBIDS=()
for spec in "${NEMOTRON_CELLS[@]}"; do
    jobid=$(sbatch --parsable sdf/scripts/run_belief_depth_nemotron.sh $spec)
    if [ -n "$jobid" ]; then
        NEMOTRON_JOBIDS+=("$jobid")
        echo "  [$jobid] nemotron: $spec"
    else
        echo "  [FAIL] nemotron: $spec"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-32B MO — 9 cells via sbatch (epochs=15)
# Args: <MODEL_KEY> <PROMPT_KEY> <EPOCHS>
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "=== Qwen3-32B MO (9 sbatches, epochs=15) ==="
QWEN_CELLS=(
    "base 1a_zh"
    "base 1a_es"
    "base 1a_fr"
    "coop_full 1a_zh"
    "coop_full 1a_es"
    "coop_full 1a_fr"
    "coop_full 1b_zh"
    "coop_full 1b_es"
    "coop_full 1b_fr"
)
QWEN_JOBIDS=()
for spec in "${QWEN_CELLS[@]}"; do
    jobid=$(sbatch --parsable evals/agentic_misalignment/slurm/run_belief_depth_qwen.sh $spec 15)
    if [ -n "$jobid" ]; then
        QWEN_JOBIDS+=("$jobid")
        echo "  [$jobid] qwen: $spec (epochs=15)"
    else
        echo "  [FAIL] qwen: $spec"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# GPT-4.1-mini — 24 cells via local nohup (epochs=15)
# Driver: run_bare_baseline_coop.py --model --chunk-id --conditions --pressure-suffix-file
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "=== GPT-4.1-mini SDF (24 local nohups, epochs=15) ==="
declare -A GPT_MODELS=(
    # base_unFT routes via OpenRouter to match the existing 2026-05-05 deepdive
    # bare baseline (`openrouter/openai/gpt-4.1-mini|bare|baseline` chunk-state).
    # FT'd organisms must use direct OpenAI (OR doesn't host private FTs).
    ["base_unFT"]="openrouter/openai/gpt-4.1-mini"
    ["coop_ft"]="openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn"
    ["anticoop_v2_ft"]="openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbbMjfXB"
    ["muan_ft"]="openai/ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbYB3Bki"
)
declare -A PRESSURE_FILES=(
    ["1a_en"]="sdf/deliberative-alignment/belief_depth_1a_prompt_specs.txt"
    ["1a_zh"]="sdf/deliberative-alignment/belief_depth_1a_prompt_specs.zh.txt"
    ["1a_es"]="sdf/deliberative-alignment/belief_depth_1a_prompt_specs.es.txt"
    ["1a_fr"]="sdf/deliberative-alignment/belief_depth_1a_prompt_specs.fr.txt"
    ["1b_en"]="sdf/deliberative-alignment/belief_depth_1b_prompt_specs.txt"
    ["1b_zh"]="sdf/deliberative-alignment/belief_depth_1b_prompt_specs.zh.txt"
    ["1b_es"]="sdf/deliberative-alignment/belief_depth_1b_prompt_specs.es.txt"
    ["1b_fr"]="sdf/deliberative-alignment/belief_depth_1b_prompt_specs.fr.txt"
)

# Cell tuple: <organism> <pressure_key>
# - base_unFT: 1a only (no 1b for a no-FT control)
# - coop_ft / anticoop_v2_ft: 1a and 1b
# - muan_ft: 1a only (drop 1b per plan addendum — semantically degenerate)
GPT_CELLS=(
    "base_unFT 1a_en" "base_unFT 1a_zh" "base_unFT 1a_es" "base_unFT 1a_fr"
    "coop_ft 1a_en" "coop_ft 1a_zh" "coop_ft 1a_es" "coop_ft 1a_fr"
    "coop_ft 1b_en" "coop_ft 1b_zh" "coop_ft 1b_es" "coop_ft 1b_fr"
    "anticoop_v2_ft 1a_en" "anticoop_v2_ft 1a_zh" "anticoop_v2_ft 1a_es" "anticoop_v2_ft 1a_fr"
    "anticoop_v2_ft 1b_en" "anticoop_v2_ft 1b_zh" "anticoop_v2_ft 1b_es" "anticoop_v2_ft 1b_fr"
    "muan_ft 1a_en" "muan_ft 1a_zh" "muan_ft 1a_es" "muan_ft 1a_fr"
)
GPT_PIDS=()
for spec in "${GPT_CELLS[@]}"; do
    read -r organism pressure_key <<< "$spec"
    model="${GPT_MODELS[$organism]}"
    pressure_file="${PRESSURE_FILES[$pressure_key]}"
    condition="bare_${pressure_key%_*}"   # bare_1a or bare_1b
    chunk_id="gpt41mini_${organism}_${pressure_key}_${DATE}"
    log_file="${LOG_DIR}/${chunk_id}.log"
    nohup python evals/agentic_misalignment/scripts/run_bare_baseline_coop.py \
        --model "$model" \
        --chunk-id "$chunk_id" \
        --conditions "$condition" \
        --epochs 15 \
        --pressure-suffix-file "$pressure_file" \
        > "$log_file" 2>&1 &
    pid=$!
    GPT_PIDS+=("$pid")
    disown
    echo "  [PID=$pid] gpt41mini: $organism × $pressure_key ($condition) -> $log_file"
done

echo
echo "=============================================="
echo "Launched:"
echo "  Nemotron sbatches: ${#NEMOTRON_JOBIDS[@]} (jobids: ${NEMOTRON_JOBIDS[*]})"
echo "  Qwen sbatches:     ${#QWEN_JOBIDS[@]} (jobids: ${QWEN_JOBIDS[*]})"
echo "  GPT-4.1-mini PIDs: ${#GPT_PIDS[@]}"
echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls $LOG_DIR/"
echo "  cat evals/agentic_misalignment/state/run_bare_baseline_coop_gpt41mini_*_${DATE}.json"
echo "=============================================="
